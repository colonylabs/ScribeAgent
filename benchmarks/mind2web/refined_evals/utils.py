import json
import re
import string
from collections import defaultdict
from beartype.typing import Any, TypedDict, Union
import html
from bs4 import BeautifulSoup, Tag, Comment
from lxml import html as lxml_html
import numpy as np
from functools import lru_cache
import transformers
import torch
import sys
import random
from pathlib import Path
import os
import argparse
import math
from bs4.element import NavigableString
from vllm import LLM, SamplingParams

MAX_RECURSION = 16000
sys.setrecursionlimit(MAX_RECURSION)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def load_model(model_name_or_path, scale_context):
    if "qwen" in model_name_or_path.lower():
        max_context_len = 32768 * scale_context
        prompt_prefix = "Help achieve the objective by generating the next step."
        rope_scaling = {
        		"factor": 4.0,
        		"original_max_position_embeddings": 32768,
        		"rope_type": "yarn"
        	}
        model = LLM(
        		model=model_name_or_path, 
        		rope_scaling=rope_scaling,
        		quantization="fp8",
        		enable_prefix_caching=True,
        	)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-32B-Instruct",
                    model_max_length=max_context_len,
                    padding_side="left",
                    token=HF_TOKEN
                )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        return model, tokenizer, max_context_len, prompt_prefix
    else:
        raise NotImplementedError("Model not yet implemented.")
        
def print_attr(element):
    element_string = f'<{element.tag}'
    for name, value in element.attrib.items():
        element_string += f' {name}="{value}"'
    element_string += '>'
    return element_string

def calculate_f1(pred, label):
    # Taken from the Mind2Web repo: https://github.com/OSU-NLP-Group/Mind2Web/tree/main

    pred = set(pred.strip().split())
    label = set(label.strip().split())

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