import os

import pandas as pd
import csv
import json


def csv_to_jsonl(csv_path, jsonl_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile, \
         open(jsonl_path, 'w', encoding='utf-8') as jsonlfile:
        
        reader = csv.DictReader(csvfile)
        for row in reader:
            jsonlfile.write(json.dumps(row, ensure_ascii=False) + '\n')


os.chdir("")

df1 = pd.read_json("memes_data/memes-trainval.json")
df2 = pd.read_json("memes_data/memes-test.json")

data = pd.concat((df1, df2))

df3 = pd.read_json("ocr/ocr_results.jsonl", lines=True)

df3["img_fname"] = df3["file"].str.split("\\", n=8).str[7]

df3.drop(columns=["file"], inplace=True)

merged_data = pd.merge(data, df3, on=["img_fname"])

merged_data.drop(columns=["category", "url", "metaphors"], inplace=True)

merged_data.to_csv("meme_temp/meme_data.csv", index=False)

csv_to_jsonl("meme_temp/meme_data.csv", "meme_temp/meme_data.jsonl")
