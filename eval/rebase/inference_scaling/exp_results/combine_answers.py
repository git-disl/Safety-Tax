import json
import os

# Initialize a list to store all JSON data
combined_data = []

# Read all JSON files
PATH="/root/swj0419/o1/ref/inference_scaling_1/exp_results/rebase_1"
for file in os.listdir(PATH):
    if file.endswith(".json"):
        with open(PATH + "/" + file, "r") as f:
            data = json.load(f)
            combined_data.append(data)

# Write combined data to a new JSON file
with open(PATH + "/answers.json", "w") as f:
    json.dump(combined_data, f, indent=4)