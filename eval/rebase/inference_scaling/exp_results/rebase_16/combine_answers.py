import json
import os

# Initialize a list to store all JSON data
combined_data = []

# Read all JSON files
for file in os.listdir("."):
    if file.endswith(".json"):
        with open(file, "r") as f:
            data = json.load(f)
            combined_data.append(data)

# Write combined data to a new JSON file
with open("answers.json", "w") as f:
    json.dump(combined_data, f, indent=4)