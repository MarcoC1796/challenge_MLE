import os
import pandas as pd
import json

script_dir = os.path.dirname(os.path.abspath(__file__))

data_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
data = pd.read_csv(filepath_or_buffer=data_file_path)

categorical_columns = ["OPERA", "TIPOVUELO", "MES"]
category_mapping = {}

for column in categorical_columns:
    unique_categories = data[column].unique().tolist()
    category_mapping[column] = unique_categories

output_file = os.path.join(script_dir, "category_mapping.json")

with open(output_file, "w") as json_file:
    json.dump(category_mapping, json_file, indent=4)

print(f"Category mapping saved to {output_file}")
