import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool
import argparse
import sys, json, os
from ast import literal_eval
tqdm.pandas()
sys.setrecursionlimit(4000)
import functools

HF_TOKEN = os.environ.get("HF_TOKEN")
CHUNK_SIZE = 4*32768 - 50

functools.lru_cache(maxsize=1000)
def token_count(chunk):
	return tokenizer(chunk, return_tensors="pt")["input_ids"].shape[1]


def naive_chunks_left(dom, chunk_size=CHUNK_SIZE):
	def clean_chunk(dom):
		if dom.find("<") < dom.find(">") + 1 and dom.find("<") != -1:
			return dom[dom.find("<"):]
		elif dom.find(">") != -1:
			return dom[dom.find(">") + 1:]
		return dom

	chunks = []
	token_num = token_count(dom)
	if chunk_size < 1000:
		print("Chunk size too small", chunk_size)
		return []
	while(token_num > chunk_size):
		cur_dom = deepcopy(dom)
		tokens = tokenizer(cur_dom, return_tensors="pt")["input_ids"][0]
		tokens = tokens[-chunk_size:]
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


def chunk_wrapper(input):
	prompt, target = input
	extra_tokens = token_count(target)
	dom = prompt.split("\nObservation: ")[1].split("\nStep-by-step guide:")[0]
	prepend = prompt.split("Observation: ")[0]
	postpend = "\nStep-by-step guide:" + prompt.split("\nStep-by-step guide:")[1]
	if token_count(prepend + "\nObservation: " + postpend) + extra_tokens > 16000:
		print("Unexpected behavior", token_count(prepend + "\nObservation: " + postpend), extra_tokens)
		return []
	chunks = naive_chunks_left(dom, chunk_size=CHUNK_SIZE - token_count(prepend + "\nObservation: " + postpend) - extra_tokens)
	final_chunks = []
	for x in chunks:
		final_chunks.append(prepend + "Observation: " + x + postpend)
	return final_chunks


def init_worker():
	import signal
	signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, default="./train.csv", help="Path")
	args = parser.parse_args()

	filtered_path = args.path
	filtered_df = pd.read_csv(filtered_path)

	filtered_df = filtered_df[["workflow_id", "action_id", "prompt", "target"]].copy()
	print(filtered_df.shape)

	tokenizer = AutoTokenizer.from_pretrained(
		"Qwen/Qwen2.5-7B-Instruct",
		token=HF_TOKEN,
		model_max_length=2**22, # arbitrarily large to avoid warnings
	)

	if "train" in filtered_path:
		with Pool(processes=192, initializer=init_worker) as pool:
			filtered_df["num_tokens"] = list(tqdm(pool.imap(token_count, filtered_df["prompt"]), total=len(filtered_df)))
		filtered_df = filtered_df[filtered_df["num_tokens"] < 32*CHUNK_SIZE].copy()

	temp = filtered_df[["prompt", "target"]].apply(lambda x: (x["prompt"], x["target"]), axis=1)

	with Pool(processes=128, initializer=init_worker) as pool:
		filtered_df["naive_chunks"] = list(tqdm(pool.imap(chunk_wrapper, temp), total=len(filtered_df), desc="Naive chunking"))

	df = filtered_df.copy()
	if "train" in filtered_path:
		def extract_id(target):
			return 'node="' + str(int(target.split("\nNode: ")[1].split("\n")[0].split(" ")[0])) + '"'

		def extract_dom(chunk):
			return chunk.split("\nObservation: ")[1].split("\nStep-by-step guide:")

		df["_idx"] = df.apply(lambda x: [(extract_id(x["target"]) in chunk) for chunk in x["naive_chunks"]], axis=1)
		df = df[df["_idx"].apply(lambda x: any(x))]
		df["chunk"] = df.apply(lambda x: x["naive_chunks"][x["_idx"].index(True)], axis=1)
		print("Final train set", df.shape)
		print("# workflows:", df["workflow_id"].nunique())
		df[["workflow_id", "action_id", "chunk", "target"]].to_csv("data/train_final.csv", index=False)
	else:
		df["chunk"] = df.apply(lambda x: x["naive_chunks"][0], axis=1)
		print("Final test set", df.shape)
		print("# workflows:", df["workflow_id"].nunique())
		df[["workflow_id", "action_id", "chunk", "target"]].to_csv("data/test_final_131k.csv", index=False)
