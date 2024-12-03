import pandas as pd
import json
from argparse import ArgumentParser
from tqdm import tqdm
tqdm.pandas()

def regenerate_target(action_id, target):
	if str(action_id) not in descriptions:
		return target
	return target.split("\nDescription:")[0] + "\nDescription: " + descriptions[str(action_id)] + "\nAction: " + target.split("\nAction:")[1]


def generate_prompt(row):
	return f"Objective: {row['Objective']}\nURL: {row['url']}\nObservation: {row['processed_dom']}\nStep-by-step guide:\n{row['prev_steps']}"


def multiple_node_id(target):
	new_target = target.split("\nTarget:")[0]
	node = new_target.split("\nNode:")[1].strip()
	return new_target + f" {node}"*4 + "\nTarget:" + target.split("\nTarget:")[1]


if __name__ == "__main__":
	arg = ArgumentParser()
	arg.add_argument("--task", type=str, required=True)
	task = arg.parse_args().task
	df = pd.read_csv(f"data/{task}.csv")

	with open(f"GPT_augmentation_files/{task}_objectives_clean.json", "r") as f:
		objectives = json.load(f)
	for wid in tqdm(objectives, desc="Objective augmentation"):
		df.loc[df['workflow_id'] == int(wid), "Objective"] = objectives[wid]

	with open(f"GPT_augmentation_files/{task}_step_desc.json", "r") as f:
		descriptions = json.load(f)
	df["target"] = df.progress_apply(lambda row: regenerate_target(row["action_id"], row["target"]), axis=1)

	df["prev_steps"] = ""
	log_prev_step = {wid: "" for wid in df["workflow_id"].unique()}
	for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Previous step"):
		df.at[idx, "prev_steps"] = log_prev_step[row["workflow_id"]]
		log_prev_step[row["workflow_id"]] += row["target"]
	
	df["target"] = df["target"].apply(multiple_node_id)
	df["prompt"] = df[["Objective", "url", "processed_dom", "prev_steps"]].progress_apply(generate_prompt, axis=1)
	df[["workflow_id", "action_id", "prompt", "target"]].to_csv(f"data/{task}_augmented.csv", index=False)