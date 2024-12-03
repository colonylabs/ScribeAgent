from multiprocessing import Pool
from openai import OpenAI
import pandas as pd
import base64, json, os
from tqdm import tqdm
from argparse import ArgumentParser

prepend = """\
You will be provided with a step-by-step process guide, which may include both textual and visual steps, and a series of screenshots. \
Your task is to generate a detailed objective (~20 words) given the step-by-step process and a suboptimal objective and URL of the website. Don't return the URL or steps from the guide."""
OPENAI_APIKEY = os.environ.get("OPENAI_APIKEY")

def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


def extract_step_desc(steps):
	steps = steps.split("\nDescription: ")[1:]
	return [step.split("\n")[0] for step in steps]


def augmentation_exists(wids):
	if os.path.exists(f"GPT_augmentation_files/{task}_objectives_clean.json"):
		with open(f"GPT_augmentation_files/{task}_objectives_clean.json", "r") as f:
			objectives = json.load(f)
		for wid in list(objectives.keys()):
			if objectives[wid] == "Error: Whoops! Failed to generate objective":
				objectives.pop(wid)
		wids = [wid for wid in wids if str(wid) not in objectives]
	return wids


def init_worker():
	import signal
	signal.signal(signal.SIGINT, signal.SIG_IGN)


def gpt_call(message):
	try:
		response = client.chat.completions.create(
			model="gpt-4o",
			messages=message
		)
	except:
		return "Error: Whoops! Failed to generate objective"
	return response.choices[0].message.content


def gpt_wrapper(input):
	workflow_id, message = input
	try:
		output = gpt_call(message)
	except Exception as e:
		print(e)
		output = "Error: Whoops! Failed to generate objective"
	with open(f"GPT_augmentation_files/{workflow_id}_objectives.json", "w") as f:
		json.dump({workflow_id: output}, f)


client = OpenAI(api_key=OPENAI_APIKEY)
args = ArgumentParser()
args.add_argument("--task", type=str, required=True)
task = args.parse_args().task
df = pd.read_csv(f"data/{task}.csv")
print(task, df.shape)

new_objectives = {}
for wid, group in tqdm(df.groupby("workflow_id"), desc="Prompt creation"):
	action = group.values[:, 1]
	exists = group.values[:, 4]
	if group.shape[0] > 1:
		steps = extract_step_desc(group.values[-1][6] + group.values[-1][7])
	else:
		steps = extract_step_desc(group.values[-1][7])
	objective = group.values[0][2]
	url = "/".join(group.values[0][3].split("/")[:5])
	prompt = "Objective: " + objective + "\nURL: " + url 
	step_log = []
	for i in range(len(action)):
		if exists[i]:
			step_log.append(
				{"role": "user", "content": [
					{"type": "text", "text": steps[i]},
					{"type": "image_url", "image_url": {
						"url": f"data:image/jpeg;base64,{encode_image(f'/data/screenshot/{action[i]}.jpeg')}", # TODO: Change this to the correct path
						"detail": "low"
					}}
				]}
			)
		else:
			step_log.append({"role": "user", "content": steps[i]})
	messages = [
		{"role": "system", "content": prepend},
		*step_log,
		{"role": "user", "content": prompt},
	]
	new_objectives[wid] = messages

novel_objective = augmentation_exists(list(new_objectives.keys()))
new_objectives = {wid : new_objectives[wid] for wid in new_objectives if wid in novel_objective}
with Pool(processes=16, initializer=init_worker) as pool:
	hello = list(tqdm(pool.imap(gpt_wrapper, new_objectives.items()), total=len(new_objectives), desc="GPT call"))

objectives = {}
for wid in new_objectives:
	if os.path.exists(f"GPT_augmentation_files/{wid}_objectives.json"):
		with open(f"GPT_augmentation_files/{wid}_objectives.json", "r") as f:
			objectives[wid] = json.load(f)[str(wid)]
		os.remove(f"GPT_augmentation_files/{wid}_objectives.json")
	else:
		print("Could not find", wid)

for key, value in objectives.items():
	if "Objective:" in value[:45]:
		objectives[key] = value.split("Objective:")[1].strip()
	elif "objective:" in value[:45]:
		objectives[key] = value.split("objective:")[1].strip()
	elif "**Objective**:" in value[:45]:
		objectives[key] = value.split("**Objective**:")[1].strip()
	elif "**Objective:**" in value[:45]:
		objectives[key] = value.split("**Objective:**")[1].strip()
	elif "**Objective:" in value[:45]:
		objectives[key] = value.split("**Objective:")[1].strip().replace("**", "")

if os.path.exists(f"GPT_augmentation_files/{task}_objectives_clean.json"):
	with open(f"GPT_augmentation_files/{task}_objectives_clean.json", "r") as f:
		objectives_clean = json.load(f)
	objectives.update(objectives_clean)

with open(f"GPT_augmentation_files/{task}_objectives_clean.json", "w") as f:
	json.dump(objectives, f)