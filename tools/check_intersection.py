import json

# fast read sowod image ids
with open("datasets/sowod/ImageSets/Main/t1_train.txt") as f:
    sowod_ids = [line.strip() for line in f.readlines()]

with open("training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json") as f:
    proposals = json.load(f)
inf_ids = list(proposals.keys())

print(f"Sample SOWOD ID: {sowod_ids[0]}, Type: {type(sowod_ids[0])}")
print(f"Sample Inf ID: {inf_ids[0]}, Type: {type(inf_ids[0])}")

matches = set(sowod_ids).intersection(set(inf_ids))
print(f"Total matching IDs: {len(matches)}")
