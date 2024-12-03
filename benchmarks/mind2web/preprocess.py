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

MAX_RECURSION = 16000
sys.setrecursionlimit(MAX_RECURSION)
HF_TOKEN = os.environ.get("HF_TOKEN")
TOKEN_RATIO = 2

class Processor():
	valid_tags = {
		'div', 'body', 'span', 'svg', 'input', 'img', 'p', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'i', 'u', 'strong', 'em', 'abbr', 'cite', 'q', 'code', 'ins', 'var', 'area', 'ul', 'li', 'ol', 'dl', 'dt', 'dd', 'form', 'button', 'col', 'textarea', 'path', 'lightning-primitive-icon', 'select', 'label', 'td', 'canvas', 'circle', 'i18n-string', 'table', 'tr', 'image', 'footer', 'use', 'option', 'rect', 'mark', 'section', 'th', 'polygon', 'aside', 'main', 'header', 'pre', 'figure'
	}

	code_elements_to_decompose = {
		'style', 'script'
	}

	salient_attributes = {
		"alt",
		"aria-role",
		"aria-label",
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

	def __init__(self, model_name_or_path, max_context_len = 32768, max_attr_len=32, max_text_len=500, clean_dropdown_limit=20):
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(
			model_name_or_path,
			model_max_length=max_context_len,
			padding_side="left",
			token=HF_TOKEN
		)
		self.clean_dropdown_limit = clean_dropdown_limit
		self.max_attr_len = max_attr_len
		self.max_text_len = max_text_len

	@lru_cache(maxsize=2**12)
	def token_ratio(self, text):
		return float(len(text)/ len(self.tokenizer(text, add_special_tokens=False)['input_ids']) + 1e-5)
	
	def collect_tags(self, tag, tags):
		# Collect all tags in tags
		if isinstance(tag, Tag):
			tags.append(tag)
			for child in tag.children:
				self.collect_tags(child, tags)

	def assign_ids(self, all_tags, nmap = {}):
		for i, tag in enumerate(all_tags[::-1]):
			tag["node"] = int(i)
			# Instead of try catch, we can check if "backend_node_id" is in tag
			try:
				nmap[str(i)] = tag["backend_node_id"]
				del tag["backend_node_id"]
			except:
				pass
		
		return nmap
	
	def clean_url(self, url):
		if "." not in url:
			url = url + ".com"
		if "https://www." not in url:
			url = "https://www." + url
		return url

	def truncate_string(self, all_tags):
		for tag in all_tags:
			if isinstance(tag, NavigableString):
				tag.string = tag.string[:self.max_text_len]

		return all_tags
		
	def clean_string(self, text):
		text = html.unescape(text)
		try:
			text = bytes(text, "utf-8").decode("unicode_escape")
		except:
			pass
		text = text.replace("–", '-').replace("•", '-').replace("’", '\'').replace("‹", '<').replace("×", '*').replace("·", '.').replace("”","\"").replace("＋", '+')
		text = text.replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")
		text = re.sub(r'[^\x00-\x7F]+',' ', text)
		text = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', ' ', text)    
		pattern = re.compile(r'[\ue000-\uf8ff]')
		text = pattern.sub('', text)
		text = re.sub(r"\s+", " ", text)

		return text

	def print_without_children(self, element):
		element_string = f'<{element.tag}'
		for name, value in element.attrib.items():
			element_string += f' {name}="{value}"'
		element_string += '>'
		return element_string

	def get_target_nodes(self, target_ids_backend, nmap):
		target_ids = []
		for backend_id in target_ids_backend:
			for node_id, backend_node_id in nmap.items():
				if backend_id == backend_node_id:
					target_ids.append(node_id)
					break
		return target_ids

	def delete_comments(self, soup):
		comments = soup.find_all(string=lambda text: isinstance(text, Comment))
		for comment in comments:
			comment.extract()

	def process_html(self, html_content, apply_option_cleaning=False):
		clean_html = self.clean_string(html_content)
		soup = BeautifulSoup(clean_html, 'html.parser')

		all_tags = []
		self.collect_tags(soup, all_tags)

		nmap = self.assign_ids(all_tags, nmap={})

		# Remove comments and truncate strings
		self.delete_comments(soup)
		self.truncate_string(all_tags)

		full_html = soup.prettify()
		full_html = re.sub(r"\s+", " ", full_html)

		num_op_tag = 0
		for tag in all_tags[1:]:
			if tag.name in self.code_elements_to_decompose:
				tag.decompose()                
			elif tag.name not in self.valid_tags:
				tag.unwrap()
			else:
				if tag.name == "option" and tag.text.isdigit():
					num_op_tag += 1
		
		# TODO: check how many times option cleaning removes target
		if apply_option_cleaning and num_op_tag > self.clean_dropdown_limit:
			for tag in all_tags[1:]:
				if tag.name == "option" and tag.text.isdigit():
					tag.decompose()

		for tag in all_tags:
			if tag.attrs is None:
				continue

			for attr in list(tag.attrs):
				if attr not in self.salient_attributes:
					del tag[attr]
					continue
				
				if len(str(tag[attr])) > self.max_attr_len and self.token_ratio(str(tag[attr])) < TOKEN_RATIO:
					del tag[attr]
					continue
					
				if tag[attr] in ["", "none"]:
					del tag[attr]
					continue
			
				if tag.name == "iframe" and attr != "node":
					del tag[attr]
				
		
		final_html = soup.prettify()
		final_html = re.sub(r"\s+", " ", final_html)

		return full_html, nmap, final_html
	
