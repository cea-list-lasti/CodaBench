import json
import glob
import os

"""
Utility file to combine multiple jsonl files from folder FOLDER into one file.
Useful when doing using multiple gpus to //
"""

FOLDER = "T0.2_preprompt_python"

# Function to combine dictionaries
def combine_dicts(*dicts):
    combined_dict = {}
    for data_dict in dicts:
        for key, values in data_dict.items():
            combined_dict.setdefault(key, []).extend(values)
    return combined_dict

# Load the dictionaries from JSONL files
def load_dict_from_jsonl(file_path):
    result_dict = {}
    with open(file_path, "r") as infile:
        for line in infile:
            entry = json.loads(line)
            key = entry["task_id"]
            values = entry["completion"]
            result_dict.setdefault(key, []).extend(values)
    return result_dict

# Use glob to get the list of all JSONL files in the current directory
jsonl_files = glob.glob(os.path.join(FOLDER,"*.jsonl"))

# Load dictionaries from all the JSONL files
all_dicts = [load_dict_from_jsonl(file_path) for file_path in jsonl_files]

# Combine all the dictionaries into one
combined_dict = combine_dicts(*all_dicts)

# Write the combined dictionary to a new JSONL file with one value per line
with open(os.path.join(FOLDER,"combined_data.jsonl"), "w") as outfile:
    for key, values in combined_dict.items():
        json.dump({"task_id": key, "completion": values}, outfile)
        outfile.write("\n")
