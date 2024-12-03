import argparse
import json
from typing import Any
from collections import defaultdict
from beartype import beartype
import os
import random
import numpy as np
import torch
import transformers
import re
import html
import math
from lxml import html as lxml_html
from lxml import etree

from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from agent import Agent
from agent.gpt_agent import GPTAgent
import sys
MAX_RECURSION = 16000
sys.setrecursionlimit(MAX_RECURSION)
IP_ADDR = os.environ.get("IP_ADDR", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_GPT_LEN = 50000


class WorkflowAgent(Agent):

    @beartype
    def __init__(
        self,
        output_dir: str, instruction_path: str, max_context_len: int = 32768, max_text_len: int = 100, dom_window_size: int = 50
    ) -> None:
        super().__init__()
        
        config = transformers.AutoConfig.from_pretrained(
    		"Qwen/Qwen2-7B-Instruct",
    		token=HF_TOKEN
    	)
        model = transformers.AutoModelForCausalLM.from_pretrained(
    		"Qwen/Qwen2-7B-Instruct",
    		config=config,
    		torch_dtype=torch.bfloat16,
    		token=HF_TOKEN
    	)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
    		"Qwen/Qwen2-7B-Instruct",
    		model_max_length=max_context_len,
    		padding_side="left",
    		token=HF_TOKEN
    	)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model.load_adapter(output_dir)
        model.load_state_dict(torch.load(f"{output_dir}/model.pt"), strict=False)
        model.eval().cuda()

        self.model = model
        self.tokenizer = tokenizer
        self.prepend = "Help achieve the objective by generating the next step."
        with open(instruction_path) as f:
            self.prompt_dict = json.load(f)
        
        self.intent = None
        self.domain = None
        self.start_url = ""
        self.last_action = ""
        self.prev_actions = ""
        self.prev_actions_acc = ""
        self.num_prev_actions = 0

        self.note = ""
        self.subsequent_actions = []        

        self.max_context_len = max_context_len
        self.max_text_len = max_text_len  
        self.dom_window_size = dom_window_size    


    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    ) -> Action:
        print("\n\n" + "="*25,"NEW STEP","="*25) 

        self.dom = trajectory[-1]["info"]["observation_metadata"]["text"]["html"]
        self.full_dom = trajectory[-1]["info"]["observation_metadata"]["text"]["obs_nodes_info_html"]["full_html"]
        self.acc = trajectory[-1]["info"]["observation_metadata"]["text"]["obs_nodes_info_html"]["acc"]
        self.url = trajectory[-1]["info"]["page"].url
        self.full_id_map = trajectory[-1]["info"]["observation_metadata"]["text"]["obs_nodes_info_html"]["full_id_map"]
        self.cleaned_id_map = trajectory[-1]["info"]["observation_metadata"]["text"]["obs_nodes_info_html"]["cleaned_id_map"]
        self.acc_id_map = trajectory[-1]["info"]["observation_metadata"]["text"]["obs_nodes_info_html"]["acc_id_map"]
        self.obs_nodes_info = trajectory[-1]["info"]["observation_metadata"]["text"]["obs_nodes_info"]  

        if self.intent is None:
            self.start_url = self.url
        self.set_domain()
        self.dom = self.clean_string(self.dom)
        self.acc = self.clean_string(self.acc, is_axtree=True)
        self.replace_urls()
        self.truncate_text()

        self.user = GPTAgent(self.prompt_dict[self.domain], self.note)
        if self.intent is None:
            self.revise_intent(intent)
        self.task_meta_info = "Objective: " + self.intent + "\nURL: " + self.url + "\n"   
    
        action = self.execute_remaining_actions()
        if action is not None:
            return action           
        
        if self.num_prev_actions > 0:
            stop_prompt = self.task_meta_info + "Accessibility tree:\n" + self.acc + "\nUser's actions:\n" + self.prev_actions_acc
            should_stop, ans = self.user.reflect(stop_prompt, "stop")
            self.note = self.user.note

            if should_stop:
                print("[ISSUE STOP]", ans)                
                return create_id_based_action("stop" + "[" + ans + "]")
                
        doms = self.chunk_dom()

        response = ""
        num_try = 0
        while len(response) == 0 and num_try < 5:
            response = self.eval(doms, seed=num_try)
            num_try += 1

        if len(response) == 0:
            return create_id_based_action("scroll [down]")
       
        parsed_response = self.parse(response)
        if parsed_response == "stop":
            stop_prompt = self.task_meta_info + "Accessibility tree:\n" + self.acc + "\nUser's actions:\n" + self.prev_actions_acc
            should_stop, ans = self.user.reflect(stop_prompt, "stop")
            self.note = self.user.note

            if should_stop:
                print("[ISSUE STOP]", ans)
                return create_id_based_action("stop" + "[" + ans + "]")

        print("[EXECUTE ACTION]", parsed_response)
        return create_id_based_action(parsed_response)


    def eval(self, doms, seed = 0, max_try = 10):
        target_predictions = defaultdict(int)
        responses = defaultdict(list)
        
        for chunkidx, raw_obs in enumerate(doms):
            self.set_deterministic(seed)

            raw_inp = self.task_meta_info + "Observation: " + raw_obs + "\nStep-by-step guide:\n" + self.prev_actions
            print("[DOM CHUNK", chunkidx + 1, "OUT OF", len(doms), "]")
            #print(raw_inp)

            messages = [
                    {"role": "system", "content": self.prepend},
                    {"role": "user", "content": raw_inp}
                ]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_text = self.tokenizer(input_text, return_tensors="pt")
            model_inputs = {}
            for key, value in input_text.items():
                model_inputs[key] = input_text[key].to(self.model.device).reshape(1, -1)
            input_len = model_inputs["input_ids"].shape[1]      
            
            if self.num_prev_actions >= 1 and "hasPopup: menu expanded: False" in self.last_action:
                action = str(self.num_prev_actions+1) + ".\nDescription: click dropdown item"
                action_inputs = self.tokenizer(action, add_special_tokens=False, return_tensors="pt")
                for key, value in model_inputs.items():
                    model_inputs[key] = torch.cat([model_inputs[key], action_inputs[key].to(self.model.device)], -1)

            num_try = 0
            while num_try < max_try:            
                try:
                    generated_ids = self.model.generate(**model_inputs, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.6, pad_token_id=self.tokenizer.eos_token_id)
                except:
                    print("[OOM]")
                    num_try += 1
                    continue
                    
                num_try += 1
                generated_ids = [generated_ids[0][input_len:]]
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                match = re.search(r"Description: ([^\n]+)\nAction: ([^\n]+)\nNode: ([^\n]+)\nTarget: (.+)", generated_text)
                if not match:
                    continue
                generated_text = str(self.num_prev_actions+1) +".\n" + re.sub(r"\s+", " ", generated_text[re.search("Description: ", generated_text).start():re.search("\nAction: ", generated_text).start()]) + generated_text[re.search("\nAction: ", generated_text).start():]
                generated_text = generated_text[:re.search("Target: ", generated_text).start()] + re.sub(r"\s+", " ", generated_text[re.search("Target: ", generated_text).start():])
                generated_text = generated_text.replace("</s>", "").strip() +"\n"
                print("[CANDIDATE " + str(num_try) + "]")
                print(generated_text)

                sidx = re.search("Node: ", generated_text).end()
                tid = generated_text[sidx:]
                eidx = re.search(r"[ \n]", tid).start()
                tid = tid[:eidx]
                if not tid.isdigit():
                    sidx = re.search("node=\"", generated_text).end()
                    alt_tid = generated_text[sidx:]
                    eidx = re.search("\"", alt_tid).start()
                    tid = alt_tid[:eidx]
                    if not tid.isdigit():
                        continue

                target_predictions[int(tid)] +=1
                responses[int(tid)].append(generated_text)

                if target_predictions[int(tid)] >= max_try // 2: break

        if len(target_predictions) == 0:
            return ""

        if len(target_predictions) == 1:
            responses = sorted(responses[int(tid)], key=lambda x: len(x)) 
            return responses[-1]

        sorted_target_predictions = sorted(target_predictions.items(), key=lambda x: x[1]) 
        prompt = self.task_meta_info +"Accessbility tree:\n" + self.acc[:MAX_GPT_LEN] + "\nPrevious steps:\n" + (self.prev_actions if len(self.prev_actions) > 0 else "None\n") + "Proposed next steps:\n" 
        
        idx_id_map = {}
        for candidate_idx, (tid, _) in enumerate(sorted_target_predictions):
            candidate = sorted(responses[tid], key=lambda x: len(x))[-1]
            prompt += "Candidate No. " + str(candidate_idx + 1) + ":\n" + candidate[re.search("Description", candidate).start():] + "\n"
            idx_id_map[candidate_idx + 1] = tid
        selected_candidate, _ = self.user.reflect(prompt, "reflect")  
        
        if selected_candidate < 0 or selected_candidate > len(idx_id_map):
            return ""
        
        target_id_pred = idx_id_map[selected_candidate]
        print("[USER PICKED ACTION]", target_id_pred)    
        responses = sorted(responses[target_id_pred], key=lambda x: len(x)) 
        return responses[-1]


    def parse(self, response):
        inv_full_id_map = {v: k for k, v in self.full_id_map.items()}
        inv_acc_id_map = {v: k for k, v in self.acc_id_map.items()}
        inv_cleaned_id_map = {v: k for k, v in self.cleaned_id_map.items()}
        tree = lxml_html.fromstring(self.dom)

        dsidx = re.search("\nDescription: ", response).end()
        deidx = re.search("\nAction: ", response).start()
        action = response[re.search("\nAction: ", response).end():re.search("\nNode: ", response).end()]
        target = response[re.search("\nNode: ", response).end():]
        target = target[:re.search(" ", target).start()]

        parsed_response = ""
        if "click" in action:
            parsed_response += "click"
        elif "sequence" in action or "type" in response[:deidx].lower():
            parsed_response += "type"
        else:
            parsed_response += "press"
        
        all_nodes = sorted(self.cleaned_id_map.keys(), key=lambda x: int(x)) 
        if target not in all_nodes:
            snippet = self.dom
        else:
            target_list_idx = all_nodes.index(target)
            target_low = max(0, target_list_idx - self.dom_window_size)
            target_high = min(len(all_nodes) - 1, target_list_idx + self.dom_window_size)

            dom_start = re.search("node=\"" + all_nodes[target_high] + "\"", self.dom).start()
            dom_end = re.search("node=\"" + all_nodes[target_low] + "\"", self.dom).end()
            snippet = self.dom[dom_start:dom_end]

        prompt = self.task_meta_info + "HTML snippit:\n" + snippet + "\nAccessibility tree:\n" + self.acc[:MAX_GPT_LEN] + "\nPrevious steps:\n" + (self.prev_actions_acc if len(self.prev_actions_acc) > 0 else "None\n") + "\nProposed next step:\n" + response[re.search("Description", response).start():]                
        _, action = self.user.reflect(prompt, "map")
        print("[USER MAPPED ACTION]", action)

        if action == "go back":
            self.prev_actions += str(self.num_prev_actions + 1) + ".\nDescription: This web page does not contain useful information. Go back to the starting page\nAction: mouse_click_action\nNode: 1 1 1 1 1\nTarget: <div node=\"1\">Go Back</div>\n"
            self.num_prev_actions += 1
            self.prev_actions_acc += "This web page does not contain useful information. Go back to the starting page\n"
            self.last_action = action
            return "goto [" + self.start_url + "]"
        
        if action in ["scroll [down]", "stop"] or not re.findall(r'\d+', action):
            self.prev_actions += response.strip() + "\n"
            self.num_prev_actions += 1
            self.prev_actions_acc += response[dsidx:deidx].strip() + "\n"
            self.last_action = action
            return action
        
        target_acc = re.findall(r'\d+', action)[0]

        if not re.search("\\[" + target_acc + "\\]", self.acc) or inv_acc_id_map[target_acc] not in inv_full_id_map:
            self.prev_actions += response.strip() + "\n"
            self.num_prev_actions += 1
            self.prev_actions_acc += response[dsidx:deidx].strip() + "\n"
            self.last_action = action
            return "scroll [down]"
            
        # rewrite user action to agent format

        target_acc_des = self.acc[re.search("\\[" + target_acc + "\\]", self.acc).end():]      
        if "\n" in target_acc_des:
            target_acc_des = target_acc_des[:re.search("\n", target_acc_des).start()].strip()  
        target_acc_des = re.sub(r"\s+", " ", target_acc_des)

        if action.split()[0] == "type":
            revise_des = "Type in " + target_acc_des + ". Enter the content: " + action[re.search("\\] \\[", action).end():-1]
        else:
            revise_des = action.split()[0] + " " + target_acc_des

        revise_des_acc = revise_des + " (Target HTML: " + response[re.search("Target: ", response).end():].strip() + ")"
        revise_action = "keyboard_sequence_action" if action[:4] == "type" else "mouse_click_action"

        if inv_acc_id_map[target_acc] in inv_cleaned_id_map.keys():
            revise_target = inv_cleaned_id_map[inv_acc_id_map[target_acc]]
        else:
            full_tree = lxml_html.fromstring(self.full_dom)
            revise_target = inv_full_id_map[inv_acc_id_map[target_acc]]
            selected_node = full_tree.xpath(f"//*[@node=\"{revise_target}\"]")[0]
            revise_target_rep = print_without_children(selected_node)
            cscores = []
            cids = []
            for child in selected_node.getchildren():
                child_backend = full_id_map[child.attrib["node"]]
                if child_backend in inv_cleaned_id_map:
                    child_id = inv_cleaned_id_map[child_backend]
                    child_id_rep = print_without_children(tree.xpath(f"//*[@node=\"{child_id}\"]")[0])
                    cids.append(child_id)
                    cscores.append(calculate_overlap_percentage(child_id_rep, revise_target_rep))
            revise_target = cids[np.argmax(cscores)] if len(cscores) > 0 else target
                
        revise_target_rep = print_without_children(tree.xpath(f"//*[@node=\"{revise_target}\"]")[0])
        revise_target_rep = revise_target_rep[:revise_target_rep.find(">")+1]
        revise_step = str(self.num_prev_actions + 1) + ".\nDescription: " + revise_des + "\nAction: " + revise_action +"\nNode: " + (revise_target + " ") * 4 + revise_target + "\nTarget: " + revise_target_rep 
        
        self.prev_actions += revise_step.strip() + "\n"
        self.num_prev_actions += 1
        self.prev_actions_acc += revise_des_acc + "\n"
        self.last_action = revise_des

        if re.findall(r'\d+', action):
            scroll_action = self.add_scroll(re.findall(r'\d+', action)[0])
            if len(scroll_action) > 0:
                self.subsequent_actions.insert(0, action)
                action = scroll_action

        return action


    def reset(self, test_config_file: str) -> None:
        self.intent = None
        self.domain = None
        self.start_url = ""
        self.last_action = ""
        self.prev_actions = ""
        self.prev_actions_acc = ""
        self.num_prev_actions = 0

        self.note = ""
        self.subsequent_actions = []        


    def set_domain(self):
        if ":9999" in self.url:
            self.domain = "reddit"
        elif ":7770" in self.url:
            self.domain = "shopping"
        elif ":8023" in self.url:
            self.domain = "git"
        elif ":7780" in self.url:
            self.domain = "admin"
        else:
            self.domain = "map"


    def replace_urls(self):
        for attr in ['dom', 'full_dom', 'url', 'acc']:
            setattr(self, attr, getattr(self, attr).replace("http://metis.lti.cs.cmu.edu", IP_ADDR))

        url_dict = {"reddit": {(IP_ADDR + ":9999"): "http://reddit.com"},
                    "shopping": {(IP_ADDR + ":7770"): "http://onestopmarket.com"},
                    "git": {(IP_ADDR + ":8023"): "http://gitlab.com"}, 
                    "admin": {(IP_ADDR + ":7780"): "http://luma.com", "/admin/admin": "/admin"},
                    "map": {(IP_ADDR + ":3000"): "http://openstreetmap.org"}}
        
        for attr in ['dom', 'full_dom', 'url', 'acc']:
            for url_src, url_tgt in url_dict[self.domain].items():
                setattr(self, attr, getattr(self, attr).replace(url_src, url_tgt))


    def revise_intent(self, intent):
        if intent[-1].isalnum():
            intent += "."
        self.ori_intent = intent
        prompt = intent + " Here's the current webpage for reference:\n" + self.acc
        _, intent = self.user.reflect(prompt, "intent")

        if intent[-1].isalnum():
            intent += "."
        intent = self.ori_intent + " Specifically: " + intent
        self.intent = intent
        print("[REVISE INTENT]", self.intent)


    def add_scroll(self, action_target):
        if action_target not in self.obs_nodes_info:
            return ""

        node_info = self.obs_nodes_info[action_target]
        node_bound = node_info["union_bound"]
        x, y, width, height = node_bound
        center_x = x + width / 2
        center_y = y + height / 2
        newy = int(center_y / 720.0)
        action = ""

        if newy >= 1:
            action = "scroll [down]" if newy == 1 else "scroll [down"+str(newy)+"]"   
        elif newy < 0:
            newy = -newy + 1
            action = "scroll [up]" if newy == 1 else "scroll [up"+str(newy)+"]"

        return action


    def execute_remaining_actions(self):
        if len(self.subsequent_actions) > 0:
            parsed_response = self.subsequent_actions[0]
            breakflag = False

            if "scroll" not in parsed_response and re.search(r'\d+', parsed_response):
                action_target = re.findall(r'\d+', parsed_response)[0]
                
                if action_target not in self.obs_nodes_info.keys():
                    try:
                        model_selected_acc = self.acc_id_map[self.cleaned_id_map[action_target]]
                        parsed_response = parsed_response.replace(action_target, model_selected_acc)
                        self.subsequent_actions[0] = parsed_response
                        action_target = re.findall(r'\d+', parsed_response)[0]
                    except:
                        self.subsequent_actions = self.subsequent_actions[1:]
                        breakflag = True

                if not breakflag:
                    scroll_action = self.add_scroll(action_target)
                    if len(scroll_action) > 0:
                        self.subsequent_actions.insert(0, scroll_action)

            if not breakflag:
                parsed_response = self.subsequent_actions[0]
                self.subsequent_actions = self.subsequent_actions[1:]
                print("[EXECUTE REMAINING ACTION]", parsed_response)
                return create_id_based_action(parsed_response)
            
        return None


    def truncate_text(self):
        if self.max_text_len != -1 and self.domain != "map":
            alllines = re.split("(\"> )", self.dom)
            lines = []
            dom_trunc = ""
            for lidx in range(len(alllines)):
                has_text = re.search("<", alllines[lidx])
                if has_text:
                    text_part = alllines[lidx][:has_text.start()]
                    if len(text_part) > self.max_text_len:
                        text_part = text_part[:self.max_text_len]
                        dom_trunc += text_part + alllines[lidx][has_text.start() - 1:] 
                        continue
                dom_trunc += alllines[lidx] 
            self.dom = re.sub(r"\s+", " ", dom_trunc.strip())

            alllines = self.acc.split("\n")
            acc_trunc = ""
            for lidx in range(len(alllines)):
                has_text = re.search("\'", alllines[lidx])
                if has_text:
                    line = alllines[lidx][has_text.end():]
                    if re.search("\'", line):
                        text_part = line[:re.search("\'", line).start()].strip()
                        if len(text_part) > self.max_text_len:
                            text_part = text_part[:self.max_text_len]
                            acc_trunc += alllines[lidx][:has_text.end()] + text_part + line[re.search("\'", line).start():] + "\n"
                            continue
                acc_trunc += alllines[lidx] + "\n"
            self.acc = acc_trunc.strip()


    def chunk_dom(self):
        cur_len = len(self.tokenizer(self.task_meta_info + self.prev_actions)["input_ids"])
        windowsize = self.max_context_len - cur_len
        obs_len = len(self.tokenizer(self.dom)["input_ids"])

        if obs_len > windowsize:
            doms = []
            alllines = re.split("(</[a-z]+> <[a-z])", self.dom)                        
            lines = []
            for lidx in range(len(alllines)):
                if lidx % 2 == 1:
                    lines.append(alllines[lidx - 1] + alllines[lidx])
                else:
                    if lidx == len(alllines) - 1:
                        lines.append(alllines[lidx])
            
            num_iter = 0
            prev_remaining = ""
            dom = ""
            while len(lines) > 0 and num_iter < MAX_RECURSION:
                num_iter += 1

                if lines[0][-4:-1] == "> <":
                    line_to_add = lines[0][:-3]
                    remaining = lines[0][-2:]
                else:
                    line_to_add = lines[0]
                    remaining = ""

                dom_new = dom + prev_remaining + line_to_add
                new_len = len(self.tokenizer(dom_new)["input_ids"]) 
                if new_len > windowsize:
                    if dom == "":
                        lines = lines[1:]
                        rev_remaining = remaining
                    else:
                        doms.append(dom)
                        dom = ""
                else:
                    dom = dom_new
                    lines = lines[1:]
                    prev_remaining = remaining

            if len(dom) > 0:
                doms.append(dom)

        else:
            doms = [self.dom]
            
        print("[CHUNK DOM INTO", len(doms), "PIECES]")
        return doms

    
    def clean_string(self, target_string, is_axtree=False):
        target_string = html.unescape(target_string)
        try:
            target_string = bytes(target_string, "utf-8").decode("unicode_escape")
        except:
            pass
        target_string = target_string.replace("–", '-').replace("•", '-').replace("’", '\'').replace("‹", '<').replace("×", '*').replace("·", '.').replace("”","\"").replace("＋", '+')
        target_string = target_string.replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")
        target_string = re.sub(r'[^\x00-\x7F]+',' ', target_string)
        target_string = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', ' ', target_string)    
        pattern = re.compile(r'[\ue000-\uf8ff]')
        target_string = pattern.sub('', target_string)
        if is_axtree:
            target_string = re.sub(r"\n([^\n]+)StaticText \'\'\n", "\n", target_string)
            target_string = re.sub(r"\n([^\n]+)LineBreak \'\n\'\n", "\n", target_string)
            target_string = re.sub(r"'\s*([^\[\]\n]+)\s*'", r"'\1'", target_string)
        else:
            target_string = re.sub(r"require([^<>\n]*);", "", target_string)
            target_string = re.sub(r"//\s*<!\[CDATA([^<>\n]*)>", "", target_string)
            target_string = re.sub(r"\s+", " ", target_string)

        return target_string


    def set_deterministic(self, seed):
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed) 
        torch.cuda.manual_seed_all(seed)


def construct_workflow_agent(args: argparse.Namespace) -> Agent:
    agent = WorkflowAgent(output_dir=args.model_endpoint, instruction_path=args.instruction_path)
    
    return agent

##### Helper Funcs #####

def print_without_children(element):
    element_string = f'<{element.tag}'
    for name, value in element.attrib.items():
        element_string += f' {name}="{value}"'
    element_string += '>'

    # Optionally, add element's text if it's not None or empty
    if element.text and element.text.strip():
        element_string += element.text.strip()

    element_string += f'</{element.tag}>'
    return element_string


def clean_str(text):
    for symbol in ["*","/","'","\"","(",")","[","]","\\","#","&",".",",",":","?","!", "<", ">", "=", "\"", "'", "-", "_"]:
        text = text.replace(symbol, ' ')
    return text


def calculate_overlap_percentage(sentence1, sentence2):
    # Tokenize the sentences into sets of words, converting to lowercase to ensure case-insensitive comparison
    sentence1 = clean_str(sentence1)
    sentence2 = clean_str(sentence2)
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())
    
    # Find the common words between the two sets
    common_words = words1.intersection(words2)
    
    # Calculate the total number of unique words across both sentences
    total_unique_words = len(words2)
    
    # Calculate the percentage of overlap
    if total_unique_words > 0:  # Prevent division by zero
        overlap_percentage = (len(common_words) / total_unique_words) 
    else:
        overlap_percentage = 0
    
    return overlap_percentage

