from dataclasses import dataclass, field
from typing import Optional
import transformers
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from peft import PeftConfig, PeftModel
import os
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

tokenizer = AutoTokenizer.from_pretrained(
	"Qwen/Qwen2.5-7B-Instruct",
	cache_dir="./",
	token=os.environ.get("HF_TOKEN"),
)

def token_length(value):
	return len(tokenizer(value)["input_ids"])

def eval():
	config = transformers.AutoConfig.from_pretrained(
		"Qwen/Qwen2.5-7B-Instruct",
		token=os.environ.get("HF_TOKEN"),
	)

	peft_config = PeftConfig.from_pretrained("finetuned_7b")

	model = transformers.AutoModelForCausalLM.from_pretrained(
		peft_config.base_model_name_or_path,
		torch_dtype=torch.bfloat16,
		cache_dir="/root/.cache"
	)

	tokenizer = AutoTokenizer.from_pretrained("finetuned_7b")
	peft_model = PeftModel.from_pretrained(model, "finetuned_7b")
	model = peft_model.merge_and_unload()
	model.load_state_dict(torch.load("finetuned_7b/model.pt"), strict=False)
	model.save_pretrained("7b_model")
	tokenizer.save_pretrained("7b_model")

if __name__ == "__main__":
	eval()
