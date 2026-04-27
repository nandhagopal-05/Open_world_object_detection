import json
with open('training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json') as f:
    data = json.load(f)
key = list(data.keys())[0]
print(f"Keys in image dict: {list(data[key].keys())}")
if "scores" in data[key]:
    print(f"Scores exist: {data[key]['scores'][:5]}")
else:
    print("Scores KEY IS MISSING!")
