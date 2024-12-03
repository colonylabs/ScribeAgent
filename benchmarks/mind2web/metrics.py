import string, re
import pandas as pd
from difflib import SequenceMatcher
import numpy as np
from lxml import html as lxml_html
import ast
from bs4 import BeautifulSoup, Tag
import pickle
import copy

file_path = 'scores_all_data.pkl'

with open(file_path, 'rb') as file:
	scores = pickle.load(file)

def calculate_f1(pred, label):
	# Taken from the Mind2Web repo: https://github.com/OSU-NLP-Group/Mind2Web/tree/main

	pred = set(pred.strip().split())
	try:
		label = set(label.strip().split())
	except:
		print(label)

	pred = set([x for x in pred if x not in string.punctuation])
	label = set([x for x in label if x not in string.punctuation])
	if len(pred) == 0 and len(label) == 0:
		return 1
	if len(pred) == 0 or len(label) == 0:
		return 0

	tp = len(pred & label)
	fp = len(pred - label)
	fn = len(label - pred)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	if precision == 0 or recall == 0:
		return 0
	f1 = 2 * precision * recall / (precision + recall)
	return f1

def get_selected_element_pred(cleaned_html, majority_vote, majority_response):
	cleaned_tree = lxml_html.fromstring(cleaned_html)
	selected_element_pred = cleaned_tree.xpath(f"//*[@node=\"{majority_vote}\"]")
	
	if len(selected_element_pred) == 0:
		selected_element_pred = majority_response[re.search("\nTarget: ", majority_response).end():]
	else:
		selected_element_pred = lxml_html.tostring(selected_element_pred[0], pretty_print=True, encoding=str)     
	
	return selected_element_pred

def get_target_id_pred_backend(majority_vote, nmap):
	full_nmap = ast.literal_eval(nmap)
	if str(majority_vote) in full_nmap:
		return full_nmap[str(majority_vote)]
	else:
		return None
	
def get_target_ids_backend(nmap, target_ids):
	full_nmap = ast.literal_eval(nmap)
	target_ids_backend = []
	for target_id in target_ids:
		if str(target_id) in full_nmap:
			target_ids_backend.append(full_nmap[str(target_id)])
	return target_ids_backend

def element_match(final_pred_id, positive_ids, all_candidates=[]):
	if final_pred_id is None:
		return False
	
	if len(positive_ids) == 0:
		return False
	
	return final_pred_id in [int(x) for x in positive_ids]

def intent_match(pred_description, target_description, cleaned_description, cleaned_description_pred):	
	cleaned_description_pred_set = set(cleaned_description_pred.split())
	cleaned_description_set = set(cleaned_description.split())

	return pred_description.lower() in target_description.lower() or target_description.lower() in pred_description.lower() or cleaned_description in cleaned_description_pred or cleaned_description_pred in cleaned_description or cleaned_description_pred_set.issubset(cleaned_description_set) or cleaned_description_set.issubset(cleaned_description_pred_set) or ''.join(cleaned_description_pred.split()) in ''.join(cleaned_description.split()) or ''.join(cleaned_description.split()) in ''.join(cleaned_description_pred.split())

def format_description(description, operation):
	return f"{operation} \"{description}\""


def reformat_click_to_type(final_response):
	pred_desc = final_response.split("Description: ")[1].split("\nAction:")[0]
	pred_action_type = final_response.split("\nAction: ")[1].split("\nNode:")[0]
	pred_target_tag = final_response.split("Target: <")[1].split(" ")[0].strip()

	toggle = False
	if pred_target_tag in ["input", "textarea"]:
		if pred_action_type == "mouse_click_action":
			pred_action_type = "keyboard_sequence_action"
			toggle = True
	
	return final_response.split("Description: ")[0] + "Description: " + pred_desc + "\nAction: " + pred_action_type + "\nNode: " + final_response.split("\nNode: ")[1], toggle


def collect_ids(tag, tags):
	# Collect all tags in tags
	if isinstance(tag, Tag):
		tags.append(tag.get("node"))
		for child in tag.children:
			collect_ids(child, tags) 
			
def mcq_element_match(x):
	final_pred_id, final_pred_response, positive_ids, cleaned_html, nmap, action_id, annotation_id, operation = x[1]["majority_vote"], x[1]["majority_response"], ast.literal_eval(x[1]["target_ids"]), x[1]["cleaned_html"], x[1]["full_map"], x[1]["action_id"], x[1]["annotation_id"], x[1]["op"]
	sample_id = f"{annotation_id}_{action_id}"
	ranks = scores["ranks"][sample_id]

	final_pred_response, _ = reformat_click_to_type(final_pred_response)
	full_nmap = ast.literal_eval(nmap)

	inverted_nmap = {}

	for k, v in full_nmap.items():
		inverted_nmap[v] = k

	soup = BeautifulSoup(cleaned_html, 'html.parser')

	chosen_id = -1

	chosen = False
	for k, v in ranks.items():
		parent = soup.find(attrs={"node": f'{int(inverted_nmap[k])}'})
		if parent is None:
			continue
		children = []
		collect_ids(parent, children)

		if str(final_pred_id) in children and not chosen:
			chosen_id = int(inverted_nmap[k])
			chosen = True

	pseudo_positive_ids = copy.deepcopy(positive_ids)
	for pid in positive_ids:
		parent = soup.find(attrs={"node": f'{pid}'})
		if parent is None:
			continue
		children_ids = []
		collect_ids(parent, children_ids)
		pseudo_positive_ids.extend(children_ids)

	action_pred = final_pred_response.split("Action: ")[1].split("\nNode:")[0]

	element_acc = chosen_id in [int(x) for x in positive_ids]

	step_sc  = 0 
	if (operation in ["CLICK", "SELECT"] and action_pred.split("_")[0] == "mouse") or (operation == "TYPE" and action_pred.split("_")[0] == "keyboard"):
		if element_acc:
			step_sc = 1

	return element_acc, step_sc

def calculate_metrics(final_pred_id, final_pred_response, operation, target_description, positive_ids, target_in_dom, cleaned_html, nmap, action_id, annotation_id, use_intent=True):
	if final_pred_id is None:
		return 0, 0, 0, 0, 0

	if not target_in_dom:
		return 0, 0, 0, 0, 0

	final_pred_response, _ = reformat_click_to_type(final_pred_response)
	
	selected_element_pred = get_selected_element_pred(cleaned_html, final_pred_id, final_pred_response)
	target_id_pred_backend = get_target_id_pred_backend(final_pred_id, nmap)
	target_ids_backend = get_target_ids_backend(nmap, positive_ids)
	cleaned_tree = lxml_html.fromstring(cleaned_html)
	target_id = ast.literal_eval(nmap)[str(positive_ids[-1])]
	selected_element = cleaned_tree.xpath("//*[@node=\"" + str(target_id) + "\"]")
	element_acc = element_match(final_pred_id, positive_ids)
	
	soup = BeautifulSoup(cleaned_html, 'html.parser')
	pseudo_positive_ids = copy.deepcopy(positive_ids)
	for pid in positive_ids:
		parent = soup.find(attrs={"node": f'{pid}'})
		if parent is None:
			continue
		children_ids = []
		collect_ids(parent, children_ids)
		pseudo_positive_ids.extend(children_ids)

	subchild_element_acc = element_match(final_pred_id, pseudo_positive_ids)

	if use_intent:
		try:
			target_el = lxml_html.tostring(selected_element, pretty_print=True, encoding=str)
		except:
			target_el = ""
		pseudo_match = target_id_pred_backend in target_ids_backend or ("node=\"" + str(final_pred_id)) in target_el or ("node=\"" + str(target_id)) in selected_element_pred
		element_acc = int(pseudo_match or element_acc)
	
	pred_description = final_pred_response.split("Description: ")[1].split("\nAction:")[0].strip()

	if pd.isna(target_description) or target_description is None:
		target_description = ""

	target_desc = format_description(target_description, operation)

	cleaned_description_pred = re.sub(r"\s+", " ", re.sub(r'[^\w\d\s]+', ' ', pred_description).lower())
	cleaned_description = re.sub(r"\s+", " ", re.sub(r'[^\w\d\s]+', ' ', target_desc).lower())

	f1 = calculate_f1(cleaned_description, cleaned_description_pred)
	step_sc, subchild_step_sc, i_step_sc = 0, 0, 0
	action_pred = final_pred_response.split("Action: ")[1].split("\nNode:")[0]

	# Do not use formatted description here
	m2w_action_desc = " " + target_description + " -> " + operation
	pseudo_f1 = calculate_f1(m2w_action_desc, operation + pred_description)

	if (operation in ["CLICK", "SELECT"] and action_pred.split("_")[0] == "mouse") or (operation == "TYPE" and action_pred.split("_")[0] == "keyboard"):
		if element_acc:
			step_sc = 1
		
		if subchild_element_acc:
			subchild_step_sc = 1

		f1 = max(f1, pseudo_f1)

	if f1 != 1:
		f1 = max(f1, calculate_f1(m2w_action_desc, pred_description))

	return element_acc, f1, step_sc, subchild_element_acc, subchild_step_sc