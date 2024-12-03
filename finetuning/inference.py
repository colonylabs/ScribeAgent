import csv, os
import sys
csv.field_size_limit(sys.maxsize)
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import Optional
import random
import torch
import transformers
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
from transformers.trainer_utils import get_last_checkpoint
from torch.multiprocessing import Process, set_start_method
import numpy as np
from vllm import LLM, SamplingParams
from peft import PeftModel
HF_TOKEN = os.environ.get("HF_TOKEN")

def print_trainable_parameters(model):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	print(
		f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
	)

@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
	model_type: Optional[str] = field(default="llama")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
	cache_dir: Optional[str] = field(default=None)
	optim: str = field(default="adamw_torch")
	file_name: str = field(default="response_test.csv")
	rank: int = field(default=0)
	model_max_length: int = field(
		default=8192 * 4,
		metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
	)
	use_flash_attn: bool = field(
		default=True,
		metadata={"help": "Whether use flash attention for training."},
	)

PREPEND = "Help achieve the objective by generating the next step."


def eval(rank, idx, model_args, training_args):
	# Load model and tokenizer
	rope_scaling = {
		"factor": 1.0,
		"original_max_position_embeddings": 32768,
		"type": "yarn"
	}
	model = LLM(model="7b_model", 
				rope_scaling=rope_scaling,
				seed = training_args.rank)
	sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=500)

	tokenizer = transformers.AutoTokenizer.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=training_args.cache_dir,
		model_max_length=training_args.model_max_length,
		padding_side="left",
		token=HF_TOKEN
	)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.padding_side = "left"

	workflow_df = pd.read_csv(f"./data/test_final.csv")
	workflow_df = workflow_df.iloc[idx].reset_index(drop=True)
	workflow_df.rename(columns={"chunk": "label_ids", "target": "label"}, inplace=True)

	response = []
	for prompt in tqdm(workflow_df["label_ids"].to_list()):
		messages = [
			{"role": "system", "content": PREPEND},
			{"role": "user", "content": prompt}
		]
		input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

		generated_ids = model.generate(
			[input_text],
			sampling_params=sampling_params,
			use_tqdm=False
		)
		generated_text = generated_ids[0].outputs[0].text
		response.append(generated_text)

	df = workflow_df.copy()
	df["response"] = response
	df.to_csv(f"{training_args.file_name}_{rank}.csv", index=False)


if __name__ == "__main__":
	set_start_method('spawn', force=True)
	parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
	model_args, training_args = parser.parse_args_into_dataclasses()
	df = pd.read_csv(f"./data/test_final.csv")
	idx = [i for i in range(len(df))]
	num_proc = 8
	idx = np.array_split(idx, num_proc)
	del df
	eval(training_args.rank, idx[training_args.rank], model_args, training_args)
