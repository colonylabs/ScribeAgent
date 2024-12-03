import pandas as pd
import numpy as np
import csv, sys, re, argparse, os
from functools import lru_cache
from bs4 import BeautifulSoup, Tag, Comment, XMLParsedAsHTMLWarning, NavigableString
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool
import base64, warnings, copy

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
sys.setrecursionlimit(16000)
csv.field_size_limit(sys.maxsize)
tqdm.pandas()
HF_TOKEN = os.environ.get("HF_TOKEN")

TOKEN_RATIO = 2

tokenizer = AutoTokenizer.from_pretrained(
	"Qwen/Qwen2.5-7B-Instruct",
	cache_dir="./",
	token=HF_TOKEN
)

def create_soup(dom):
	# Create a BeautifulSoup object from the HTML DOM
	return BeautifulSoup(dom, "html.parser")


# TODO: Replace this with soup.find_all(True)
def find_tags(soup):
	# Find all tags in the DOM since findall() does not return all tags
	def collect_tags(tag, tags):
		if isinstance(tag, Tag):
			tags.append(tag)
			for child in tag.children:
				collect_tags(child, tags)

	all_tags = []
	collect_tags(soup, all_tags)
	return all_tags

		
def assign_element_id(all_tags, soup):
	# Assign an element-id to each tag in the DOM
	def assign_decreasing(all_tags):
		for i, tag in enumerate(all_tags[::-1]):
			tag["node"] = int(i)

	assign_decreasing(all_tags)


salient_attributes = {
	"alt",
	"aria-role",
	"aria-label",
	"aria-selected",
	"data-content",
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


def remove_comments(soup):
	# Remove all comments from the DOM
	comments = soup.find_all(string=lambda text: isinstance(text, Comment))
	for comment in comments:
		comment.extract()

@lru_cache(maxsize=2**12)
def token_ratio(window):
	return float(len(window)) / (len(tokenizer(window, add_special_tokens=False)["input_ids"]) + 1e-5)


def removing_extra_attributes(all_tags, max_len=128):
	# Remove all attributes from the tags that are not in the salient_attributes and sub_salient_attributes list and remove empty attributes
	for tag in all_tags:
		if tag.attrs is None:
			continue

		for attr in list(tag.attrs):
			if attr in ["title", "alt", "aria-label", "node"]:
				continue

			if type(tag[attr]) == list:
				tag[attr] = [v for v in tag[attr] if token_ratio(v) >= TOKEN_RATIO]

				if len(tag[attr]) == 0:
					del tag[attr]
					continue
			else:
				if len(tag[attr]) > max_len and token_ratio(tag[attr]) < TOKEN_RATIO:
					del tag[attr]
					continue

			if attr in tag:
				if type(tag[attr]) == list:
					tag[attr] = " ".join(tag[attr])
				tag[attr] = str(tag[attr])[:max_len]

			if "script" in attr.lower():
				del tag[attr]
			elif "style" in attr.lower():
				if "user-select: none" in tag[attr]:
					tag[attr] = "user-select: none"
				else:
					del tag[attr]
			elif "jsaction" in attr.lower():
				tag[attr] = ""
			elif attr.lower().startswith("on"):
				tag[attr] = ""
			elif attr not in salient_attributes:
				del tag[attr]
			elif (tag[attr] == "" or tag[attr] == "none"):
				del tag[attr]
			elif attr in tag:
				if tag.name == "iframe":
					if attr != "node":
						del tag[attr]

	return all_tags


valid_tags = {
	'div', 'body', 'span', 'svg', 'input', 'img', 'p', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'i', 'u', 'strong', 'em', 'abbr', 'cite', 'q', 'code', 'ins', 'var', 'area', 'ul', 'li', 'ol', 'dl', 'dt', 'dd', 'form', 'button', 'col', 'textarea', 'path', 'lightning-primitive-icon', 'select', 'label', 'td', 'canvas', 'circle', 'i18n-string', 'table', 'tr', 'image', 'footer', 'use', 'option', 'rect', 'mark', 'section', 'th', 'polygon', 'aside', 'main', 'header', 'pre', 'figure'
}
code_elements_to_decompose = {
	'style', 'script'
}


def removing_tags(all_tags):
	count = 0
	for tag in all_tags[1:]:
		if tag.name in code_elements_to_decompose:
			tag.decompose()
		elif tag.name not in valid_tags:
			tag.unwrap()
	return all_tags


def target_html(soup, cssselector):
	# Extract the target tag and its element-id using the cssselector
	try:
		tag = soup.select_one(cssselector)
	except:
		return "<phantom_tag node=1>", -1

	copied_tag = copy.copy(tag)
	# Keep the parent tag with its navigable string
	if copied_tag:
		for child in copied_tag.find_all(recursive=False):
			child.decompose()
	try:
		element_id = tag["node"]
	except:
		return "<phantom_tag node=1>", -1
	tag = copied_tag.prettify()
	tag = tag[:tag.find('>')+1]
	return tag, element_id


def target_html_clean(target_html, max_len=128):
	# Same pre-processing as rest of the DOM for target tag
	soup = BeautifulSoup(target_html, "html.parser")
	tag = list(soup.children)[0]

	for attr in list(tag.attrs):
		if attr in ["title", "alt", "aria-label", "node"]:
			continue

		if type(tag[attr]) == list:
			tag[attr] = [v for v in tag[attr] if token_ratio(v) >= TOKEN_RATIO]
			if len(tag[attr]) == 0:
				del tag[attr]
				continue
		else:
			if len(tag[attr]) > max_len and token_ratio(tag[attr]) < TOKEN_RATIO:
				del tag[attr]
				continue

		if attr in tag:
			if type(tag[attr]) == list:
				tag[attr] = " ".join(tag[attr])
			tag[attr] = str(tag[attr])[:max_len]

		if "script" in attr.lower():
			del tag[attr]
			continue
		
		if "style" in attr.lower():
			if "user-select: none" in tag[attr]:
				tag[attr] = "user-select: none"
			else:
				del tag[attr]
			continue

		if "jsaction" in attr.lower():
			tag[attr] = ""
			continue

		if attr.lower().startswith("on"):
			tag[attr] = ""
			continue

		if attr.lower() not in salient_attributes:
			del tag[attr]
			continue
		elif (tag[attr] == "" or tag[attr] == "none"):
			del tag[attr]
			continue
		
		if attr in tag:
			if tag.name == "iframe":
				if attr != "node":
					del tag[attr]
	tag = soup.prettify()
	# Only include opening <>, without Navigable Strings or closing <>
	return tag[:tag.find(">")+1]


def remove_extra_whitespace(dom):
	return re.sub(r"\s+", " ", dom)


def generate_output(row):
	sc, ad, ak, id= row["step_count"], row["action_description"], row["action_kind"], row["target_element_id"]
	return f"{sc}.\nDescription: {ad}\nAction: {ak}\nNode: {int(id)}\nTarget: {row['target_html']}\n"


no_drop_tags = {
	'span', 'svg', 'input', 'img', 'p', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6','button', 'textarea', 'select', 'i18n-string', 'canvas'
}


def is_bad_tag(tag):
	if any([x.lower() in ["title", "alt", "aria-label", "style"] for x in list(tag.attrs)]):
		return False
	if tag.name not in no_drop_tags:
		return True
	return False


def prune_children(tag):
	if isinstance(tag, NavigableString):
		if tag.text.strip() == "":
			tag.extract()
		return
	else:
		if len(list(tag.children)) > 0:
			for child in list(tag.children):
				prune_children(child)
		
		if len(list(tag.children)) == 0:
			if is_bad_tag(tag):
				tag.decompose()
	return


def process_data(input):
	row = {
		"dom" : input[1]["dom"], 
		"cssselector" : input[1]["cssselector"]
	}
	soup = create_soup(row["dom"])
	all_tags = find_tags(soup)
	assign_element_id(all_tags, soup)
	remove_comments(soup)
	_target_html, target_element_id = target_html(soup, row["cssselector"])
	if target_element_id >= 0:
		all_tags = removing_tags(all_tags)
		_target_html = target_html_clean(_target_html, 32)
		all_tags = removing_extra_attributes(all_tags, 32)
		prune_children(soup)
		try:
			processed_dom = soup.prettify()
		except:
			return (None, None, None)
		processed_dom = remove_extra_whitespace(processed_dom)
		return (_target_html, target_element_id, processed_dom)
	else:
		return (None, None, None)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", action="store_true", help="train or test")
	args = parser.parse_args()

	path = "data/"
	if args.train:
		filename = "train_data.csv"
	else:
		filename = "test_data.csv"
	data = pd.read_csv(path + filename)
	data = data.drop_duplicates(subset=["action_id"]).reset_index(drop=True)

	useful_cols = ['workflow_id', 'Objective', 'action_kind', 'action_description', 'url', 'action_id', 'screenshot_exists', 'cssselector', 'step_count', 'dom']
	data = data[useful_cols]
	data = data.dropna(subset=["dom", "cssselector"]).copy()
	print(filename.split("_")[0] + ":", data.shape)

	with Pool(96) as pool:
		results = list(tqdm(pool.imap(process_data, data.iterrows()), total=len(data), desc="DOM Processing"))

	data["target_html"], data["target_element_id"], data["processed_dom"] = zip(*results)
	nan_wid = data[data["target_element_id"].isna()]["workflow_id"].unique()
	data = data[~data["workflow_id"].isin(nan_wid)].reset_index(drop=True)

	step_counter = {wid: 1 for wid in data["workflow_id"].unique()}
	data["step_count"] = -1
	for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Step count"):
		data.at[idx, "step_count"] = step_counter[row["workflow_id"]]
		step_counter[row["workflow_id"]] += 1
	assert data["step_count"].min() > 0

	data["step_count"] = data["step_count"].astype(int)
	data = data.sort_values(["workflow_id", "step_count"]).copy().reset_index(drop=True)
	data["target"] = data[["step_count", "action_description", "action_kind", "target_html", "target_element_id"]].apply(generate_output, axis=1)

	data["prev_steps"] = ""
	log_prev_step = {wid: "" for wid in data["workflow_id"].unique()}
	for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Previous step"):
		data.at[idx, "prev_steps"] = log_prev_step[row["workflow_id"]]
		log_prev_step[row["workflow_id"]] += row["target"]

	data = data[["workflow_id", "action_id", "Objective", "url", "screenshot_exists", "processed_dom", "prev_steps", "target"]]
	assert data["action_id"].nunique() == len(data)
	if args.train:
		data.to_csv(path + "train.csv", index=False)
	else:
		data.to_csv(path + "test.csv", index=False)
