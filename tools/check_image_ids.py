import sys
sys.path.append(".")
from detectron2.data.datasets.pascal_voc import load_voc_instances
try:
    data_dict = load_voc_instances("datasets/sowod", "t1_train", ["unknown"])
    print(f"Loaded {len(data_dict)} instances.")
    if len(data_dict) > 0:
        print(f"Sample image_id type: {type(data_dict[0]['image_id'])}")
        print(f"Sample image_id value: '{data_dict[0]['image_id']}'")
        
    import json
    with open("training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json") as f:
        proposals = json.load(f)
    print(f"Sample inference image_id type: {type(list(proposals.keys())[0])}")
    print(f"Sample inference image_id value: '{list(proposals.keys())[0]}'")
except Exception as e:
    print(e)
