from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import csv, sys, re, torch, os
import pickle as pkl
csv.field_size_limit(sys.maxsize)
tqdm.pandas()

DEVICE="cuda"
model_ckpt = "papluca/xlm-roberta-base-language-detection"
input_file = "data/top_apps.csv"
path = ""
out_path = ""

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(DEVICE)
id2lang = model.config.id2label

# Attempt to read the CSV file directly
data = []
progress_bar = tqdm()
with open(path + input_file) as file:
	reader = csv.reader(file)
	first_row = True 
	while True:
		try:
			row = next(reader)
		except StopIteration:
			break
		progress_bar.update(1)
		if first_row:
			first_row = False
			continue
		data.append(row)
progress_bar.close()

with open(path + input_file) as file:
	for row in csv.reader(file):
		columns = row
		break

data = pd.DataFrame(data, columns=columns)
ignore_wid = set()
data[(data["action_kind"] == "section") | (data["action_kind"] == "stack")]["document_metadata_id"].apply(lambda x: ignore_wid.add(x))
data = data[~data["document_metadata_id"].isin(ignore_wid)]
data = data[data["action_kind"] != "instruction"]

cleaned_data = pd.DataFrame()
cleaned_data["workflow_id"] = data["document_metadata_id"].astype(int)
cleaned_data["Objective"] = data["document_metadata_name"]
cleaned_data["action_kind"] = data["action_kind"]
cleaned_data["action_description"] = data["action_description"]
cleaned_data["url"] = data["action_url"]
cleaned_data["cssselector"] = data["action_target_selector"]
cleaned_data["step_count"] = data["action_step_number"]
cleaned_data["dom"] = data["action_dom"]
cleaned_data["action_id"] = data["action_id"]
cleaned_data["screenshot_exists"] = cleaned_data["action_id"].apply(lambda x: os.path.exists(f"/data/new_data/screenshot/{x}.jpeg"))
cleaned_data = cleaned_data.drop_duplicates().copy()
del data

def detect_lang(desc):
	if type(desc) != str:
		return 'nope'
	desc = re.sub(r'[^\x00-\x7F]+', ' ', desc)
	desc = re.sub(r'http\S+', '', desc)
	desc = re.sub(r'[\[\]+"]', '', desc)
	inputs = tokenizer(desc, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
	with torch.no_grad():
		outputs = model(**inputs).logits
	preds = torch.softmax(outputs, dim=-1)
	_, idxs = torch.max(preds, dim=1)
	return id2lang[idxs.item()]


def is_in_english(sent):
	c = 0
	cerr = 0
	for s in sent.split(): 
		c += 1.0
		try:
			s.encode(encoding='utf-8').decode('ascii')
		except UnicodeDecodeError:
			cerr += 1.0
	isinEng = c <= 3 or (cerr/c <= 0.25)
	if isinEng:
		for specialc in ["ñ","ó","ú","ä","í","á","é","ç","õ","ã","ç","É","ï","Ü","ë","è","ü","註解"]:
			if specialc in sent:
				isinEng=False
				break

		if isinEng:
			des = re.sub(r'[^\x00-\x7F]+','', sent)
			if len(sent) - len(des) >= 5:
				isinEng=False
	return isinEng

def null_wid(data):
	wid_dict = {wid: False for wid in data["workflow_id"].unique()}
	for _, row in tqdm(data[["dom", "workflow_id"]].iterrows(), total=data.shape[0], desc="Dropping workflows with null DOM"):
		if not pd.isna(row["dom"]):
			wid_dict[row["workflow_id"]] = True
	null_wid = []
	for wid in wid_dict:
		if not wid_dict[wid]:
			null_wid.append(wid)
	return data[~data["workflow_id"].isin(null_wid)]

def init_worker():
	import signal
	signal.signal(signal.SIGINT, signal.SIG_IGN)

non_eng_wid = {}
for _, row in tqdm(cleaned_data.iterrows(), total=cleaned_data.shape[0], desc="Language filtering"):
	if row["workflow_id"] not in non_eng_wid:
		non_eng_wid[row["workflow_id"]] = detect_lang(row["Objective"])
non_eng_wid = [wid for wid, lang in non_eng_wid.items() if lang != "en"]
cleaned_data = cleaned_data[~cleaned_data["workflow_id"].isin(non_eng_wid)]
cleaned_data = cleaned_data[~cleaned_data["action_description"].isna()]
cleaned_data = cleaned_data[cleaned_data["action_description"].progress_apply(is_in_english)]
cleaned_data = cleaned_data[~cleaned_data["Objective"].isna()]
cleaned_data = cleaned_data[cleaned_data["Objective"].progress_apply(is_in_english)]
cleaned_data = null_wid(cleaned_data)
cleaned_data = cleaned_data.dropna(subset=["dom"]).copy()
print("Filtered data size: ", cleaned_data.shape)

########### Create train-test split ###########

with open("data/test_id.pkl", "rb") as file:
	test_id = pkl.load(file)

data = cleaned_data.copy()
print("Test percentage:", data["workflow_id"].isin(test_id).mean())
print("Screenshot percentage:", data["screenshot_exists"].mean())
train_data = data[~data["workflow_id"].isin(test_id)].copy()
test_data = data[data["workflow_id"].isin(test_id)].copy()
test_data.to_csv("data/test_data.csv", index=False)
train_data.to_csv("data/train_data.csv", index=False)
print("Train data size:", train_data.shape)
print("Test data size:", test_data.shape)