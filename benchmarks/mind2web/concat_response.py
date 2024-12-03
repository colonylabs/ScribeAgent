import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Concatenate responses')
parser.add_argument('--task', type=str, default="test_website", help='task, test_domain or test_task or test_website')

args = parser.parse_args()
dfs = [pd.read_csv(f"./{args.task}_response_{i}.csv") for i in range(8)]
df = pd.concat(dfs)

df.to_csv(f"{args.task}_response.csv", index=False)