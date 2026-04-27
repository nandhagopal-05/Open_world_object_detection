import json
import sys

filename = "training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json"
try:
    with open(filename, 'r') as f:
         data = json.load(f)
    print(f"Total inference results: {len(data)}")
    if len(data) > 0:
         print(f"First element keys: {data[0].keys()}")
         valid_boxes = [d for d in data if 'bbox' in d]
         print(f"Total valid bounding boxes: {len(valid_boxes)}")
         if len(valid_boxes) > 0:
             print(f"Sample bbox: {valid_boxes[0]}")
         else:
             print("No bounding boxes found in inference output.")
except Exception as e:
    print(f"Error reading {filename}: {e}")
