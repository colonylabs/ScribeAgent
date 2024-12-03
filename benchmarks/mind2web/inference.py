import csv, ast, sys, math
csv.field_size_limit(sys.maxsize)
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import pickle as pkl
import torch
import transformers
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
from torch.multiprocessing import Process, set_start_method
import numpy as np
from accelerate import infer_auto_device_map
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tqdm.pandas()


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
class DataArguments:
	test_file_path: Optional[str] = field(default="test_final.csv")

@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
	model_type: Optional[str] = field(default="llama")
	rope_scaling: Optional[int] = field(default=3)
	vllm_model: Optional[str] = field(default="7b_vllm")
	n_sample: Optional[int] = field(default=64)
	temperature: Optional[float] = field(default=0.7)

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

tokenizer = AutoTokenizer.from_pretrained(
	"Qwen/Qwen2-7B-Instruct",
	cache_dir="./",
)

def token_length(value):
	return len(tokenizer(value)["input_ids"])

def eval(model_args, training_args, data_args, idx):
	rope_scaling = {
		"factor": 4.0,
		"original_max_position_embeddings": 32768,
		"type": "yarn"
	}
	model = LLM(model=model_args.vllm_model, 
			 rope_scaling=rope_scaling,
			 enable_prefix_caching=True,
			 quantization="fp8")
	sampling_params = SamplingParams(n=model_args.n_sample, temperature=model_args.temperature, top_p=0.95, max_tokens=500)

	tokenizer = transformers.AutoTokenizer.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=training_args.cache_dir,
		padding_side="left",
	)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.padding_side = "left"

	workflow_df = pd.read_csv(data_args.test_file_path)
	workflow_df = workflow_df.iloc[idx].reset_index(drop=True)
	workflow_df.rename(columns={"chunks": "label_ids", "target": "label"}, inplace=True)

	response = []
	for chunks in tqdm(workflow_df["label_ids"].to_list()):
		chunk_response = []
		for prompt in ast.literal_eval(chunks):
			samples = []
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
			for i in range(model_args.n_sample):
				generated_text = generated_ids[0].outputs[i].text
				samples.append(generated_text)
			chunk_response.append(samples)
		response.append(chunk_response)

	df = workflow_df.copy()
	df["response"] = response
	df.to_csv(f"{training_args.file_name}_{training_args.rank}.csv", index=False)


if __name__ == "__main__":
	set_start_method('spawn', force=True)
	parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
	model_args, training_args, data_args = parser.parse_args_into_dataclasses()
	df = pd.read_csv(data_args.test_file_path)
	idx = [i for i in range(len(df))]
	num_proc = 8
	idx = np.array_split(idx, num_proc)
	del df
	eval(model_args, training_args, data_args, idx[training_args.rank])