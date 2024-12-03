import csv, sys, random, torch
from collator import DataCollatorForWorkflow
csv.field_size_limit(sys.maxsize)
import pandas, logging, datasets, math
from dataclasses import dataclass, field
from typing import Optional
import transformers, os
from torch.utils.data import Dataset
from transformers import Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
from datasets import Dataset

HF_TOKEN = os.environ.get("HF_TOKEN")
DEFAULT_EOS_TOKEN = "<|im_end|><|endoftext|>"

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
	model_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2")
	model_type: Optional[str] = field(default="mistral")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
	cache_dir: Optional[str] = field(default=None)
	optim: str = field(default="adamw_torch")
	model_max_length: int = field(
		default=32768,
		metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
	)
	use_flash_attn: bool = field(
		default=True,
		metadata={"help": "Whether use flash attention for training."},
	)
	low_rank_training: bool = field(
		default=True,
		metadata={"help": "Whether use low rank adaptation for training."},
	)
	trainable_params: str = field(
		default="embed,norm",
		metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
	)

@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	Using `HfArgumentParser` we can turn this class
	into argparse arguments to be able to specify them on
	the command line.
	"""
	dataset_path: Optional[str] = field(
		default=None,
		metadata={"help": "The path to the dataset."},
	)

class CustomTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		outputs = model(**inputs)

		loss = None
		if "labels" in inputs:
			labels = inputs["labels"]
			# Upcast to float if we need to compute the loss to avoid potential precision issues
			logits = outputs.logits
			logits = logits.float()

			# Shift so that tokens < n predict n
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()

			# Flatten the tokens
			loss_fct = torch.nn.CrossEntropyLoss()
			shift_logits = shift_logits.view(-1, logits.shape[-1])
			shift_labels = shift_labels.view(-1)

			# # Better memory management
			pad_idx = shift_labels != -100
			pad_idx = pad_idx.tolist().index(True)
			shift_labels = shift_labels[pad_idx:]
			shift_logits = shift_logits[pad_idx:]

			# Enable model parallelism
			shift_labels = shift_labels.to(shift_logits.device)
			loss = loss_fct(shift_logits, shift_labels)
			assert (outputs.loss - loss).abs() < 1e-5
		
		return (loss, outputs) if return_outputs else loss

logger = logging.getLogger(__name__)

PREPEND = "Help achieve the objective by generating the next step."

def train():
	seed = 0
	torch.cuda.empty_cache()
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed) 
	torch.cuda.manual_seed_all(seed)

	parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, DataTrainingArguments))
	model_args, training_args, data_args = parser.parse_args_into_dataclasses()
	
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Set RoPE scaling factor
	config = transformers.AutoConfig.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=training_args.cache_dir,
		token=HF_TOKEN,
	)

	orig_rope_scaling = getattr(config, "rope_scaling", None)
	if orig_rope_scaling is None:
		orig_rope_scaling = {"factor": 1}

	orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
	orig_ctx_len = getattr(config, "max_position_embeddings", None)
	if orig_ctx_len:
		orig_ctx_len *= orig_rope_scaling_factor
		if training_args.model_max_length > orig_ctx_len:
			scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
			config.rope_scaling = {"type": "linear", "factor": scaling_factor}

	model = transformers.AutoModelForCausalLM.from_pretrained(
		model_args.model_name_or_path,
		config=config,
		cache_dir=training_args.cache_dir,
	)
	tokenizer = transformers.AutoTokenizer.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=training_args.cache_dir,
		model_max_length=training_args.model_max_length,
		padding_side="left",
	)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.padding_side = "left"

	workflow_df = pandas.read_csv(data_args.dataset_path).sample(frac=1, random_state=seed)
	workflow_df["target"] = workflow_df["target"].apply(lambda x: x + " " + DEFAULT_EOS_TOKEN)
	workflow_df = workflow_df[['chunk','target']]
	workflow_df.rename(columns={"chunk": "label_ids", "target": "label"}, inplace=True)
	train_dataset = Dataset.from_pandas(workflow_df)

	data_collator = DataCollatorForWorkflow(tokenizer=tokenizer, max_length=training_args.model_max_length, prepend=PREPEND)
	if training_args.low_rank_training:
		targets =[
			"q_proj",
			"k_proj",
			"v_proj",
			"o_proj",
			"gate_proj",
			"up_proj",
			"down_proj",
			"lm_head",
			"embed_tokens",
		]
		config = LoraConfig(
			r=64,
			lora_alpha=128,
			target_modules=targets,
			lora_dropout=0.05,
			bias="none",
			task_type="CAUSAL_LM",
		)
		model = get_peft_model(model, config)
		
		# enable trainable params
		flag = True
		param_list = []
		for n, p in model.named_parameters():
			if any([k in n for k in training_args.trainable_params.split(",")]):
				if flag:
					# Just for Qwen - don't want to finetune the whole embedding layer
					flag = False
				else:
					p.requires_grad_()
					param_list.append(n)
					
	model.config.use_cache = False         # required for gradient checkpointing
	model.enable_input_require_grads()     # required for gradient checkpointing
	model.gradient_checkpointing_enable()  # enable gradient checkpointing

	print_trainable_parameters(model)
	param_list = [".".join(x.split(".")[2:]) for x in param_list]

	trainer = CustomTrainer(
		model=model,
		tokenizer=tokenizer, 
		args=training_args,
		train_dataset=train_dataset,
		data_collator=data_collator)

	torch.cuda.empty_cache()

	trainer.train(resume_from_checkpoint=None)

	trainer.save_state()
	trainer.save_model(output_dir=training_args.output_dir)

	model.to("cpu")
	save_dict = {x : model.model.state_dict()[x] for x in param_list[2:]}
	torch.save(save_dict, training_args.output_dir + "/model" + os.environ.get("RANK", -1) + ".pt")


if __name__ == "__main__":
	train()