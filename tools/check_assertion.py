import json
import torch
import sys
sys.path.append(".")
from detectron2.data.datasets.pascal_voc import load_voc_instances

print("Loading data...")
ALL_CLS_NAMES_OWDETR = ["airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep",
    "train","elephant","bear","zebra","giraffe","truck","person",
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","dining table",
    "potted plant","backpack","umbrella","handbag","tie",
    "suitcase","microwave","oven","toaster","sink","refrigerator","bed","toilet","couch",
    "frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake",
    "laptop","mouse","remote","keyboard","cell phone",
    "book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush","wine glass","cup","fork","knife","spoon","bowl","tv","bottle"]
data_dict = load_voc_instances("datasets/sowod", "t1_train", ALL_CLS_NAMES_OWDETR)
data_by_id = {str(d['image_id']): d for d in data_dict}

with open("training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json") as f:
    proposals = json.load(f)

for image_id, img_data in data_by_id.items():
    det = proposals.get(image_id)
    if det and len(det.get("bboxes", [])) > 0:
        bboxes = torch.tensor(det["bboxes"]).reshape([-1, 4])
        w = bboxes[:,2] - bboxes[:,0]
        h = bboxes[:,3] - bboxes[:,1]
        size = w * h
        size_img = img_data['height'] * img_data['width']
        mask = size > size_img
        if True in mask:
            print(f"BINGO! Image {image_id} has a box larger than the image! This triggered the AssertionError and crashed Step 8.")
            exit(0)
            
print("None found. I must be wrong.")
