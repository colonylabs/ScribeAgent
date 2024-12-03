import json
import re
import string
from collections import defaultdict
from beartype.typing import Any, TypedDict, Union
import html
from bs4 import BeautifulSoup, Tag, Comment
from lxml import html as lxml_html
from lxml.html import InputElement
import numpy as np
from functools import lru_cache
import transformers
import torch
import sys
import random
from pathlib import Path
import os
import argparse
import math
from bs4.element import NavigableString
from vllm import LLM, SamplingParams
from utils import load_model, print_attr, calculate_f1

MAX_RECURSION = 16000
sys.setrecursionlimit(MAX_RECURSION)

valid_tags = {
'div', 'body', 'span', 'svg', 'input', 'img', 'p', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'i', 'u', 'strong', 'em', 'abbr', 'cite', 'q', 'code', 'ins', 'var', 'area', 'ul', 'li', 'ol', 'dl', 'dt', 'dd', 'form', 'button', 'col', 'textarea', 'path', 'lightning-primitive-icon', 'select', 'label', 'td', 'canvas', 'circle', 'i18n-string', 'table', 'tr', 'image', 'footer', 'use', 'option', 'rect', 'mark', 'section', 'th', 'polygon', 'aside', 'main', 'header', 'pre', 'figure'
}

code_elements_to_decompose = {
    'style', 'script'
}

salient_attributes = {
    "alt",
    "aria-role",
    "aria-label",
    "option_selected",
    "placeholder",
    "role",
    "type",
    "node",
    "desc",
    "label",
    "input",
    "name",
    "title",
    "text",
    "value",
    "href",
    "expanded",
    "required",
    "selected",
    "id",
    "class"
}

class Processor():
    def __init__(self, tokenizer, clean_dropdown_limit = 20, max_attr_len = 32, max_text_len = 100):
        self.tokenizer = tokenizer
        self.clean_dropdown_limit = clean_dropdown_limit
        self.max_attr_len = max_attr_len
        self.max_text_len = max_text_len

    @lru_cache(maxsize=2**12)
    def token_ratio(self, target_string):
        return float(len(target_string)) / (len(self.tokenizer(target_string, add_special_tokens=False)["input_ids"]) + 1e-5)

    def collect_tags(self, tag, tags):
        if isinstance(tag, Tag):
            tags.append(tag)
            for child in tag.children:
                self.collect_tags(child, tags)

    def clean_string(self, target_string):        
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
        target_string = re.sub(r"\s+", " ", target_string)

        return target_string

    def process_html(self, html_content, target_ids_backend=None):
        html_content = self.clean_string(html_content)
        soup = BeautifulSoup(html_content, "html.parser")

        all_tags = []
        self.collect_tags(soup, all_tags)

        full_map = {}
        target_nodes = []
        for i, tag in enumerate(all_tags[::-1]):
            tag["node"] = int(i)
            try:
                full_map[str(i)] = tag["backend_node_id"]
                if tag["backend_node_id"] in target_ids_backend:
                    target_nodes.append(i)
                del tag["backend_node_id"]
            except:
                pass
        if len(target_nodes) == 0:
            return None, None, None, None, None

        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

        full_html = soup.prettify()
        full_html = re.sub(r"\s+", " ", full_html)

        num_op_tag = 0
        for tag in all_tags[1:]:
            if tag.name in code_elements_to_decompose:
                tag.decompose()                
            elif tag.name not in valid_tags:
                tag.unwrap()
            else:
                if tag.name == "option" and tag.text.isdigit():
                    num_op_tag += 1

        if num_op_tag > self.clean_dropdown_limit:
            for tag in all_tags[1:]:
                if tag.name == "option" and tag.text.isdigit():
                    tag.decompose()
 
        for tag in all_tags:
            if tag.attrs is None:
                continue                    

            for attr in list(tag.attrs):
                
                if attr.lower() not in salient_attributes:
                    del tag[attr]
                    continue

                if len(str(tag[attr])) > self.max_attr_len and self.token_ratio(str(tag[attr])) < 2:
                    del tag[attr]
                    continue
                
                if tag[attr] in ["", "none"]:
                    del tag[attr]
                    continue

                if tag.name == "iframe" and attr != "node":
                    del tag[attr]
        
        cleaned_html = soup.prettify()
        cleaned_html = re.sub(r"\s+", " ", cleaned_html)
        try:
            cleaned_html = cleaned_html[re.search("<body", cleaned_html).start():re.search("</body>", cleaned_html).end()]
        except:
            pass

        return full_html, full_map, cleaned_html, full_map, target_nodes


def generate(model, tokenizer, task_meta_info, previous_actions, prompt_prefix, doms, maxtry = 5, seed = 1):
    
    target_predictions = defaultdict(int)
    responses = defaultdict(list)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=200)

    for dom in doms:
        raw_inp = task_meta_info + "Observation: " + dom + "\nStep-by-step guide:\n" + previous_actions
        
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed) 
        torch.cuda.manual_seed_all(seed)
    
        messages = [
                        {"role": "system", "content": prompt_prefix},
                        {"role": "user", "content": raw_inp}
                    ]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
        try:
            generated_ids = model.generate(
    			[input_text] * maxtry,
    			sampling_params=sampling_params,
    			use_tqdm=False
        		)
        except:
            print("[OOM]")
            continue

        for idx in range(maxtry):
            response = generated_ids[idx].outputs[0].text        
            match = re.search(r".\nDescription: ([^\n]+)\nAction: ([^\n]+)\nNode: ([^\n]+)\nTarget: (.+)", response)
            if not match:
                print("[PATTERN NOT MATCH]")
                continue
                
            sidx = re.search("Node: ", response).end()
            tid = response[sidx:]
            eidx = re.search(r"[ \n]", tid).start()
            tid = tid[:eidx]
            
            try:
                int(tid)
            except:
                try:
                    sidx = re.search("node=\"", response).end()
                    alt_tid = response[sidx:]
                    eidx = re.search("\"", alt_tid).start()
                    tid = int(alt_tid[:eidx])
                except:
                    print("[INVALID GENERATED ID]")
                    continue

            target_predictions[int(tid)]+=1
            responses[int(tid)].append(response)

            if idx == maxtry // 2 and len(target_predictions) == 1:
                break

    if len(target_predictions) == 0:
        return None, None

    sorted_target_predictions = sorted(target_predictions.items(), key=lambda x: x[1]) 
    max_vote = sorted_target_predictions[-1][-1]
    target_id_preds = []
    for k, v in sorted_target_predictions:
        if v == max_vote:
            target_id_preds.append(k)
    target_id_pred = max(target_id_preds)
    responses = sorted(responses[target_id_pred], key=lambda x: len(x)) 
    response = responses[-1]
    print("[PRED]", response)

    return target_id_pred, response


def evaluate(task, model, tokenizer, max_context_len, prompt_prefix, processor, output_name, refined_eval, start=0):
    dirname = 'data/' + task
    os.makedirs("results/" + task, exist_ok = True)

    filenames = Path(dirname).rglob("*.json")
    filenames = sorted([str(s) for s in filenames])[start:]
    
    print("*** Task:", task)
    print("*** Num files to evaluate:", len(filenames))
    print()

    for filename in filenames:

        with open(filename, 'r') as file:
            print("*** Start evaluating file", filename)

            data = json.load(file)
            responses = {}

            task_srs = []
            step_srs = []
            element_accs = []
            f1s = []
            valid_steps = []
            
            for wid, datapoint in enumerate(data):
                url = datapoint["website"]
                if "." not in url:
                    url = url + ".com"
                if "https://www." not in url:
                    url = "https://www." + url
                obj = datapoint["confirmed_task"]
                action_reprs = datapoint["action_reprs"]

                print()
                print("*** Workflow", wid)
                print("*** URL:", url)
                print("*** Objective:", obj)
                print("*** Action sequence:", action_reprs)

                responses[wid] = {}
                step_id = 1
                previous_actions = ""
                all_correct = 1

                for sid, step in enumerate(datapoint["actions"]):
                    print("-"*20, "step", sid + 1, "-"*20)

                    operation = step["operation"]
                    action = operation["op"]
                    action_repr = processor.clean_string(action_reprs[sid])

                    if len(step["pos_candidates"]) == 0:
                        target_ids_backend = [step['action_uid']]
                    else:
                        target_ids_backend = [target["backend_node_id"] for target in step["pos_candidates"]]

                    html_content = step["raw_html"]
                    full_html, full_map, cleaned_html, full_map, target_ids = processor.process_html(html_content, target_ids_backend)

                    if full_html is None:
                        print("[TARGET NOT IN OBSERVATION]")
                        valid_steps.append(0)
                        step_srs.append(0)
                        element_accs.append(0)
                        f1s.append(0)
                        all_correct = 0
                        continue

                    inv_full_map = {v: k for k, v in full_map.items()}
                    cleaned_tree = lxml_html.fromstring(cleaned_html)
               
                    found = False
                    for candidate in target_ids_backend:
                        if str(candidate) not in inv_full_map.keys():
                            continue

                        target_id = inv_full_map[str(candidate)]
                        if "node=\"" + str(target_id) not in cleaned_html:
                            continue

                        try:
                            selected_element = cleaned_tree.xpath(f"//*[@node=\"{target_id}\"]")[0]
                        except:
                            continue
                        found = True
                        break

                    if not found:
                        print("[TARGET PRUNED]")
                        if "Action:" in previous_actions:
                            insertidx = previous_actions.rfind("\nAction:")
                            previous_actions = previous_actions[:insertidx] + " " + action.lower().strip() + " " + action_reprs[sid].split("->")[0].strip() + "." + previous_actions[insertidx:]        
                        valid_steps.append(1)
                        step_srs.append(0)
                        element_accs.append(0)
                        f1s.append(0)
                        all_correct = 0
                        continue

                    task_meta_info = "Objective: " + obj + "\n" + "URL: " + url + "\n" 
                    cur_len = len(tokenizer(task_meta_info + previous_actions)["input_ids"])
                    windowsize = max_context_len - cur_len
                    obs_len = len(tokenizer(cleaned_html)["input_ids"])

                    if processor.max_text_len != -1:
                        alllines = re.split("(\"> )", cleaned_html)
                        lines = []
                        cleaned_html_trunc = ""
                        for lidx in range(len(alllines)):
                            has_text = re.search("<", alllines[lidx])
                            if has_text:
                                text_part = alllines[lidx][:has_text.start()]
                                if len(text_part) > processor.max_text_len:
                                    text_part = text_part[:processor.max_text_len]
                                    cleaned_html_trunc += text_part + alllines[lidx][has_text.start() - 1:] 
                                    continue
                            cleaned_html_trunc += alllines[lidx] 
                        cleaned_html = re.sub(r"\s+", " ", cleaned_html_trunc.strip())
                        obs_len = len(tokenizer(cleaned_html)["input_ids"])

                    if obs_len > windowsize:
                        doms = []
                        alllines = re.split("(</[a-z]+> <[a-z])", cleaned_html)                        
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
                            new_len = len(tokenizer(dom_new)["input_ids"]) 
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
                        print("[CHUNK DOM INTO", len(doms), "PIECES]")

                    else:
                        doms = [cleaned_html]

                    if action in ["CLICK", "SELECT"]:
                        action_new = "mouse_click_action"
                        sidx = re.search("] ", action_repr).end()
                        eidx = re.search(" ->", action_repr).start()
                        description = action[0] + action[1:].lower()
                        description_ = action_repr[sidx:eidx].strip()
                        if len(description_) > 0:
                            description += " \"" + description_ + "\""
                        else:
                            description_ = "here"
                            description += " " + description    
                    else:
                        action_new = "keyboard_sequence_action"
                        description_ = step["operation"]["value"].strip()
                        description = "Type \"" + description_ + "\""
                        
                    cur_step = str(step_id) + ".\nDescription: " + description + "\nAction: " + action_new +  "\nNode: " + str(target_id) + (" " + str(target_id)) * 4 + "\nTarget: " + print_attr(selected_element) + "\n"
                        
                    target_id_pred, response = generate(model, tokenizer, task_meta_info, previous_actions, prompt_prefix, doms)
                    
                    if response is None:
                        print("[FAIL TO GENERATE]")
                        valid_steps.append(1)
                        step_srs.append(0)
                        element_accs.append(0)
                        f1s.append(0)
                        all_correct = 0
                        previous_actions += cur_step
                        step_id += 1
                        continue

                    print("[GT]", cur_step)
                    selected_element_pred = cleaned_tree.xpath(f"//*[@node=\"{target_id_pred}\"]")
                    if len(selected_element_pred) == 0:
                        selected_element_pred = response[re.search("\nTarget: ", response).end():]
                    else:
                        selected_element_pred = lxml_html.tostring(selected_element_pred[0], pretty_print=True, encoding=str)     
                    
                    if str(target_id_pred) in full_map:
                        target_id_pred_backend = full_map[str(target_id_pred)]
                    else:
                        target_id_pred_backend = None
                    element_acc = target_id_pred_backend in target_ids_backend or target_id_pred in target_ids

                    description_pred = response[re.search("Description: ", response).end():re.search("\nAction:", response).start()] 
                    action_pred = response[re.search("Action: ", response).end():re.search("\nNode:", response).start()] 
                    cleaned_description_pred = re.sub(r"\s+", " ", re.sub(r'[^\w\d\s]+', ' ', description_pred).lower())
                    cleaned_description = re.sub(r"\s+", " ", re.sub(r'[^\w\d\s]+', ' ', description).lower())
                    
                    if refined_eval:
                        
                        if selected_element.text and selected_element.text.strip() and isinstance(selected_element_pred, InputElement):                            
                            function_match = selected_element.text == selected_element_pred.text
                        else:
                            function_match = description_.lower() in description_pred.lower() or cleaned_description in cleaned_description_pred
                        element_acc = int(function_match or element_acc)

                    f1, step_sr = 0, 0
        
                    if (action in ["CLICK", "SELECT"] and action_pred.split("_")[0] == "mouse") or (action == "TYPE" and action_pred.split("_")[0] == "keyboard"):
                        if element_acc:
                            f1, step_sr = 1, 1
                        else:
                            f1 = calculate_f1(action_repr[re.search("]", action_repr).end():], action_repr.split(" -> ")[-1] + " " + description_pred)
                    
                    if f1 != 1:
                        f1 = max(f1, calculate_f1(action_repr[re.search("]", action_repr).end():], description_pred), calculate_f1(cleaned_description_pred, cleaned_description))

                    element_accs.append(element_acc)
                    step_srs.append(step_sr)
                    f1s.append(f1)
                    valid_steps.append(1)
                    all_correct *= element_acc
                        
                    print("[EA/F1/SR]", element_acc, f1, step_sr)
                    print("[SUMMARY][EA]", np.mean(element_accs), "[F1]", np.mean(f1s), "[STEP SR]", np.mean(step_srs), "[TASK SR]",np.mean(task_srs), "[NUM STEPS]", len(f1s))
                    
                    responses[wid][step_id] = {"prediction": response, "label": cur_step, "backend_pred": str(target_id_pred_backend), "backend_label": tuple(target_ids_backend), "node_pred": str(target_id_pred), "node_label": tuple(target_ids)}
                    
                    if element_acc == 1 and step_sr == 1 and f1 == 1:
                        previous_actions +=response
                    else:
                        previous_actions += cur_step
                        
                    step_id += 1  
                    
                task_srs.append(all_correct)

            with open(filename.replace("data/", "results/").replace(".json", "_" + output_name + ("_refined" if refined_eval else "") + "_response.json"), "w") as f:
                json.dump(responses, f)

            fstats = {"valid_steps": valid_steps, "element_accs": element_accs, "step_srs": step_srs, "f1s": f1s, "task_srs": task_srs}

            with open(filename.replace("data/", "results/").replace(".json", "_" + output_name + ("_refined" if refined_eval else "") +"_stats.json"), "w") as f:
                json.dump(fstats, f)

            print("*** Finish evaluating", filename)
            print("[SUMMARY][EA]", np.mean(element_accs), "[F1]", np.mean(f1s), "[STEP SR]", np.mean(step_srs), "[TASK SR]",np.mean(task_srs), "[NUM STEPS]", len(f1s))
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M2W Evaluation - Direct Generation')
    parser.add_argument('--task', type=str, default="test_domain", help='task, test_domain or test_task or test_website')
    parser.add_argument('--model_name_or_path',type=str, default="")
    parser.add_argument('--output_name',type=str, default="qwen32b")
    parser.add_argument('--refined_eval',type=bool, default=False)
    parser.add_argument('--scale_context',type=int, default=3)
    parser.add_argument('--start',type=int, default=0)

    args = parser.parse_args()

    model, tokenizer, max_context_len, prompt_prefix = load_model(args.model_name_or_path, args.scale_context)
    processor = Processor(tokenizer)
    evaluate(args.task, model, tokenizer, max_context_len, prompt_prefix, processor, args.output_name, args.refined_eval, start=args.start)


