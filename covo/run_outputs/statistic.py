import numpy as np
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip out whitespace and check if the line is not empty
            if line.strip():
                data.append(json.loads(line))
    return data

correct_num = 0
total_num = 0
folder_path = 'riv_group.jsonl'

content = read_jsonl(folder_path)

for item in content:
    coe_list = []
    for key, value in item.items():
        coe_list.append(np.mean(value["volatility"]))  # np.mean(value["volatility"])
    
    for key, value in item.items():
        if len(coe_list)>1:
            total_num += 1
            if value["is_correct"] == value["judge"]:
                correct_num += 1

print(f"Accuracy: {correct_num/total_num}")