import argparse
import ast
import transformers
import torch
import re
import pickle
import pandas as pd
from collections import defaultdict
from metrics import calculate_f1, element_match, calculate_metrics, format_description, reformat_click_to_type, mcq_element_match
from lxml import html as lxml_html
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import copy
 
tqdm.pandas()

def gather_predictions(predictions):
	# I have a list of lists of predictions
	# The first axis represents each chunk
	# The second axis represents each sample in the chunk
	# I want to find the most common node_id across the samples in all chunks
	predictions = ast.literal_eval(predictions)

	frequency = defaultdict(int)
	responses = defaultdict(list)
	for dom_preds in predictions:
		for sample in dom_preds:
			try:
				node_id = int(sample.split("Node: ")[1].split(" ")[0].strip())
			except:
				continue
			frequency[node_id] += 1
			responses[node_id].append(sample)

	sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
	max_vote = sorted_frequency[0][-1]

	candidates = []
	for k, v in sorted_frequency:
		if v == max_vote:
			candidates.append(k)
	
	# This could be chosen at random, but we do max
	best_candidate = max(candidates)

	# This could also be chosen at random, but we do max length
	responses = sorted(responses[best_candidate], key=lambda x: len(x), reverse=True)
	assert type(responses[0]) == str
	return best_candidate, responses[0]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='M2W Evaluation')
	parser.add_argument('--task', type=str, default="test_domain", help='task, test_domain or test_task or test_website')
	parser.add_argument("--use_intent", action="store_true", help="use intent matching")
	args = parser.parse_args()
	
	path = f"{args.task}_response"
	df = pd.read_csv(path)

	df[["majority_vote", "majority_response"]] = df["response"].apply(lambda x: pd.Series(gather_predictions(x)))

	df[["element_acc", "f1", "step_success", "subchild_element_acc", "subchild_step_success"]] = df.progress_apply(lambda x: pd.Series(calculate_metrics(x["majority_vote"], x["majority_response"], x["op"], x["description_"], ast.literal_eval(x["target_ids"]), x["target_in_dom"], x["cleaned_html"], x["full_map"], x["action_id"], x["annotation_id"], use_intent=args.use_intent)), axis=1)

	with Pool(128) as pool:
		df[["mcq", "mcq_step"]] = list(tqdm(pool.imap(mcq_element_match, df.iterrows()), total=len(df)))

	print(f"ELEMENT ACCURACY: {df['element_acc'].mean()}")
	print(f"F1: {df['f1'].mean()}")
	print(f"STEP SUCCESS: {df['step_success'].mean()}")

	print(f"MCQ: {df['mcq'].mean()}")
	print(f"MCQ STEP SC: {df['mcq_step'].mean()}")

	print(f"Subchild Direct: {df['subchild_element_acc'].mean()}")
	print(f"Subchild Direct STEP SC: {df['subchild_step_success'].mean()}")

	df["valid"] = df["target_ids"].apply(lambda x: len(ast.literal_eval(x)) != 0)
	groups = df.groupby("wid")

	print("Direct Task SR")
	print(sum((group['element_acc'] == 1.0).all() for _, group in groups))

	print("Direct Task Subchild SR")
	print(sum((group['subchild_element_acc'] == 1.0).all() for _, group in groups))

	print("MCQA Task Success Count")
	print(sum((group['mcq'] == 1.0).all() for _, group in groups))

	count = 0
	for _, group in groups:
		if all([(x["mcq"] == 1) or (x["valid"] == False) for _,x in group.iterrows()]):
			count += 1
	
	print("MCQA Task Success Count W/O INVALID")
	print(count)
