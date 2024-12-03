import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List
import math
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForWorkflow:
	"""Data collator for decoder-only models. Does left padding."""
	
	tokenizer: PreTrainedTokenizerBase
	max_length: Optional[int] = None
	label_pad_token_id: int = -100
	return_tensors: str = "pt"
	eval_mode: bool = False
	prepend: str = None

	def __call__(self, batch, return_tensors=None):

		if return_tensors is None:
			return_tensors = self.return_tensors

		model_inputs = defaultdict(list)
		
		num_passed = 0

		for idx, instance in enumerate(batch):
			message = [
				{"role": "system", "content": self.prepend},
				{"role": "user", "content": instance["label_ids"]}
			]
			input_text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
			if self.eval_mode:
				tokenized_input = self.tokenizer(input_text, padding=True, pad_to_multiple_of=4)["input_ids"]
				tokenized_output = []
			else:
				tokenized_input = self.tokenizer(input_text, padding=True, pad_to_multiple_of=4)["input_ids"]
				tokenized_output = self.tokenizer(instance["label"], add_special_tokens=False)["input_ids"] #+ [self.tokenizer.eos_token_id]
				if len(tokenized_output) % 4 != 0:
					tokenized_output += [self.tokenizer.eos_token_id] * (4 - len(tokenized_output) % 4)

			input_len = len(tokenized_input)
			# print(len(tokenized_input)+len(tokenized_output))
			if len(tokenized_input)+len(tokenized_output) > self.max_length:
				if num_passed == 0 and idx == len(batch) - 1:
					to_trim = len(tokenized_input) + len(tokenized_output) - self.max_length
					to_trim = math.ceil(to_trim//4)*4
					tokenized_input = tokenized_input[:int(0.5 * input_len) - int(0.5 * to_trim)] + tokenized_input[int(0.5 * input_len) + int(0.5 * to_trim):]
					input_len = len(tokenized_input)
				else:
					continue

			if self.eval_mode:
				model_inputs["input_ids"].append(tokenized_input)
				model_inputs["attention_mask"].append([1 for _ in tokenized_input])

			else:
				tokenized_input += tokenized_output
				tokenized_output = [self.label_pad_token_id] * input_len + tokenized_output

				model_inputs["input_ids"].append(tokenized_input)
				model_inputs["labels"].append(tokenized_output)
				model_inputs["attention_mask"].append([1 for _ in tokenized_input])
			num_passed += 1

		# Left-pad inputs, convert to tensor
		for key, value in model_inputs.items():
			if key == "labels":
				pad_token_id = self.label_pad_token_id
			elif key == "attention_mask":
				pad_token_id = 0
			else:
				pad_token_id = self.tokenizer.pad_token_id

			# To left-pad inputs, reverse, then right-pad, then reverse
			value_tensors = [torch.tensor(v[::-1]) for v in value]
			model_inputs[key] = torch.fliplr(
					pad_sequence(
						value_tensors,
						batch_first=True,
						padding_value=pad_token_id,
					)
				)
	
		return dict(model_inputs)
