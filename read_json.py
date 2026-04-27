import json

input_path = r"training_dir/mepu-sowod/fs-t1-self-train/inference/inference_results.json"
output_path = r"training_dir/mepu-sowod/fs-t1-self-train/inference/inference_results_pretty.json"

with open(input_path, "r") as f:
    data = json.load(f)

with open(output_path, "w") as f:
    json.dump(data, f, indent=4)  # <-- indent makes it vertical and readable

print(f"✅ Saved vertical (pretty) JSON at: {output_path}")

