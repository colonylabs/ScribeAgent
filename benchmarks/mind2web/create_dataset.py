import argparse
import pandas as pd
import json
import re
import transformers
import functools
import html
import os

from tqdm import tqdm
from copy import deepcopy
from lxml import html as lxml_html
from bs4 import BeautifulSoup, Tag, Comment, NavigableString
from preprocess import Processor
from functools import lru_cache

BASE_CONTEXT_LEN = 32768
HF_TOKEN = os.environ.get("HF_TOKEN")
NFILES = {"test_domain": 10, "test_task": 3, "test_website": 2}

functools.lru_cache(maxsize=1000)
def token_count(chunk):
	return tokenizer(chunk, return_tensors="pt")["input_ids"].shape[1]

def clean_chunk(dom):
		if dom.find("<") < dom.find(">") + 1 and dom.find("<") != -1:
			return dom[dom.find("<"):]
		elif dom.find(">") != -1:
			return dom[dom.find(">") + 1:]
		return dom

def naive_chunks_left(dom, tokenizer, max_context_len):

	chunks = []
	token_num = token_count(dom)
	if max_context_len < 1000:
		print("Chunk size too small", max_context_len)
		return []
	while(token_num > max_context_len):
		cur_dom = deepcopy(dom)
		tokens = tokenizer(cur_dom, return_tensors="pt")["input_ids"][0]
		tokens = tokens[-max_context_len:]
		cur_dom = tokenizer.decode(tokens)
		cur_dom = clean_chunk(cur_dom)
		if (cur_dom.find("<") == -1 and cur_dom.find(">") == -1) or len(cur_dom) < 500:
			if len(cur_dom) > 500:
				dom = dom[:-len(cur_dom)]
			else:
				dom = dom[:-500]
				if dom.rfind("<") > dom.rfind(">") + 1:
					dom = dom[:dom.rfind("<")]
				else:
					dom = dom[:dom.rfind(">") + 1]
		else:
			chunks.append(cur_dom)
			dom = dom[:-len(cur_dom)]
		token_num = token_count(dom)
	chunks.append(dom)
	return chunks
	
def create_dataset(task, max_context_len, processor, tokenizer):
	num_files = NFILES[task]
	filenames = [f"data/{task}/{task}_{i}.json" for i in range(num_files)]

	num_empty_pos_candidates = 0
	df = pd.DataFrame(columns=["wid", "action_id", "annotation_id", "cleaned_html", "url", "op", "obj", "chunks", "description_", "target", "target_in_dom", "target_ids", "full_map", "previous steps", "workflow_desc", "pos_candidates"])
	for file in filenames:
		with open(file, "r") as f:
			data = json.load(f)
			for wid, datapoint in tqdm(enumerate(data)):
				url = processor.clean_url(datapoint["website"])
				obj = datapoint["confirmed_task"]

				workflow_desc = datapoint["action_reprs"]

				previous_actions = ""
				step_id = 1

				for aid, step in enumerate(datapoint["actions"]):
					action_id = step["action_uid"]
					action = step["operation"]["op"]

					step_repr = processor.clean_string(workflow_desc[aid])

					if len(step["pos_candidates"]) == 0:
						num_empty_pos_candidates += 1
						target_ids_backend = []
					else:
						target_ids_backend = [target["backend_node_id"] for target in step["pos_candidates"]]
					
					raw_html = step["raw_html"]
					full_html, nmap, cleaned_html = processor.process_html(raw_html)

					target_ids = processor.get_target_nodes(target_ids_backend, nmap)

					try:
						cleaned_html = cleaned_html[re.search("<body", cleaned_html).start():re.search("</body>", cleaned_html).end()]
					except:
						pass
						
					inv_nmap = {v: k for k, v in nmap.items()}
					cleaned_tree = lxml_html.fromstring(cleaned_html)
			
					target_in_dom = False
					for candidate in target_ids_backend:
						if str(candidate) not in inv_nmap.keys():
							continue

						target_id = inv_nmap[str(candidate)]
						if "node=\"" + str(target_id) not in cleaned_html:
							continue

						try:
							selected_element = cleaned_tree.xpath(f"//*[@node=\"{target_id}\"]")[0]
						except:
							continue
						target_in_dom = True
						break

					task_prefix = "Objective: " + obj + "\nURL: " + url + "\n"
					prefix_tokens = len(tokenizer(task_prefix + previous_actions)["input_ids"])
					obs_tokens = max_context_len - prefix_tokens
					full_obs_len = len(tokenizer(cleaned_html)["input_ids"])

					if full_obs_len > obs_tokens:
						chunks = naive_chunks_left(cleaned_html, tokenizer, obs_tokens)
					else:
						chunks = [cleaned_html]
					
					if action in ["CLICK", "SELECT"]:
						action_new = "mouse_click_action"
						sidx = re.search("] ", step_repr).end()
						eidx = re.search(" ->", step_repr).start()

						description = action[0] + action[1:].lower()
						description_ = step_repr[sidx:eidx].strip()

						if len(description_) > 0:
							description += " \"" + description_ + "\""
					else:
						action_new = "keyboard_sequence_action"
						description_ = step["operation"]["value"].strip()
						description = "Type \"" + description_ + "\""
					
					prompts = []
					for chunk in chunks:
						prompts.append(task_prefix + "Observation: " + chunk + "\nStep-by-step guide:\n" + previous_actions + "\n")
					
					cur_step = str(step_id) + ".\nDescription: " + description + "\nAction: " + action_new + "\nNode: " + str(target_id) + "\nTarget: " + processor.print_without_children(selected_element) + "\n"

					previous_actions += cur_step
					step_id += 1

					
					row = {"wid": wid,"action_id": action_id,  "annotation_id": datapoint["annotation_id"], "cleaned_html": cleaned_html, "url": str(url), "op": action,"obj": obj, "chunks": str(prompts), "description_": str(description_), "target": str(cur_step), "target_in_dom": target_in_dom, "target_ids": str(target_ids), "full_map": str(nmap), "previous steps": str(previous_actions), "workflow_desc":str(workflow_desc), "pos_candidates": str(step["pos_candidates"])}

					prev = df.shape[0]
					df = pd.concat([df, pd.DataFrame(row, index=[prev])], ignore_index=True)
					assert df.shape[0] == prev + 1
	
	print("_"*50)
	print("Number of empty pos candidates: ", num_empty_pos_candidates)
	return df

if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='M2W Evaluation')
	parser.add_argument('--task', type=str, default="test_domain", help='task, test_domain or test_task or test_website')
	parser.add_argument('--model_name_or_path',type=str, default="Qwen/Qwen2.5-7B-Instruct")
	parser.add_argument('--rope_scaling',type=int, default=3)
	
	args = parser.parse_args()
	max_context_len = BASE_CONTEXT_LEN * args.rope_scaling - 200

	tokenizer = transformers.AutoTokenizer.from_pretrained(
		"Qwen/Qwen2.5-7B-Instruct",
		model_max_length=max_context_len,
		padding_side="left"
	)
	processor = Processor(args.model_name_or_path, max_context_len=max_context_len)
	df = create_dataset(args.task, max_context_len=max_context_len, processor=processor, tokenizer=tokenizer)
	df.to_csv(f"{args.task}_rope_{args.rope_scaling}.csv", index=False)