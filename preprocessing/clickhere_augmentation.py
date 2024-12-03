from argparse import ArgumentParser
from openai import OpenAI
import base64, json, os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

OPENAI_APIKEY = os.environ.get("OPENAI_APIKEY")
client = OpenAI(api_key=OPENAI_APIKEY)

system = "You are navigating a webpage to achieve an objective. Given the objective, a list of the previous actions, the current action, and a screenshot of the current action on the webpage. The objective and previous steps are only here to ground the current step, the current action and its screenshot are the most useful to your task. Give me a concise description of the current action being done on the webpage. You should look at the part of the webpage with the red circle, this is where the user clicked for the current action. Describe this action and ensure your response is in the same format, concise, coherent. Use any relevant information in the image to ground the action description. Your response should NOT use any json or markdown formatting. The response should be a single sentence that starts with an action verb. For example, 'Click on the 'SUBMIT' button.'"

image_prompt = "Image of the current action being done on the webpage with a red circle around the action:"

def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')

def extract_step_desc(row, image_path="/data/new_data/circled_ss"):
	objective = row["Objective"]
	prev_steps = "\nPrevious Steps:\n" + row["prev_steps"]
	current_target = row["target"]
	if not os.path.exists(f"{image_path}/{row['action_id']}_circled.jpeg"):
		return "Error: Whoops! Failed to generate step description"
	image = encode_image(f'{image_path}/{row["action_id"]}_circled.jpeg')
	prompt = objective + prev_steps + "\nCurrent Step:\n" + current_target + image_prompt
	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": [
			{"type": "text", "text": prompt},
			{"type": "image_url", "image_url": {
				"url": f"data:image/jpeg;base64,{image}",
				"detail": "high"
			}}
		]},
	]
	return messages

def augmentation_exists(aids):
	if os.path.exists(f"GPT_augmentation_files/{task}_step_desc.json"):
		with open(f"GPT_augmentation_files/{task}_step_desc.json", "r") as f:
			descriptions = json.load(f)
		for aid in list(descriptions.keys()):
			if descriptions[aid] == "Error: Whoops! Failed to generate step description":
				descriptions.pop(aid)
		aids = [aid for aid in aids if str(aid) not in descriptions]
	return aids


def gpt_call(message):
	try:
		response = client.chat.completions.create(
			model="gpt-4o",
			messages=message
		)
	except:
		return "Error: Whoops! Failed to generate step description"
	return response.choices[0].message.content


def gpt_wrapper(input):
	action_id, message = input
	output = gpt_call(message)
	with open(f"GPT_augmentation_files/{action_id}_step_desc.json", "w") as f:
		json.dump({action_id: output}, f)


def is_click_here(target):
	desc = target.split("Description: ")[1].split("\n")[0]
	try:
		len_string = len(desc.lower().split("click here")[1])
	except:
		len_string = 10

	if "click here" in desc.lower() and len_string < 5:
		return True
	return False


def init_worker():
	import signal
	signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == "__main__":
	args = ArgumentParser()
	args.add_argument("--task", type=str, required=True)
	task = args.parse_args().task

	df = pd.read_csv(f"data/{task}.csv")

	df = df[df.apply(lambda x : is_click_here(x["target"]), axis=1)]
	df["prev_steps"] = df["prev_steps"].fillna("")
	click_here_desc = {row["action_id"]: extract_step_desc(row) for _, row in df.iterrows()}
	click_here_desc = {k: v for k, v in click_here_desc.items() if type(v) != str}

	novel_descriptions = augmentation_exists(list(click_here_desc.keys()))
	click_here_desc = {aid : click_here_desc[aid] for aid in click_here_desc if aid in novel_descriptions}

	with Pool(processes=16, initializer=init_worker) as pool:
		_ = list(tqdm(pool.imap(gpt_wrapper, click_here_desc.items()), total=len(click_here_desc), desc="GPT call"))

	for aid in click_here_desc:
		if os.path.exists(f"GPT_augmentation_files/{aid}_step_desc.json"):
			with open(f"GPT_augmentation_files/{aid}_step_desc.json", "r") as f:
				click_here_desc[aid] = json.load(f)[str(aid)]
			os.remove(f"GPT_augmentation_files/{aid}_step_desc.json")
		else:
			raise Exception("This should never happen! We ran GPT query but it didn't create a corresponding file")

	if os.path.exists(f"GPT_augmentation_files/{task}_step_desc.json"):
		with open(f"GPT_augmentation_files/{task}_step_desc.json", "r") as f:
			existing_desc = json.load(f)
		click_here_desc.update(existing_desc)

	with open(f"GPT_augmentation_files/{task}_step_desc.json", "w") as f:
		json.dump(click_here_desc, f)

