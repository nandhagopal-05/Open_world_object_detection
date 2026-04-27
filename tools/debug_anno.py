import json
import torch
import sys
sys.path.append(".")
from detectron2.data.datasets.pascal_voc import load_voc_instances
from mepu.utils.utils import get_centerness

# We need the IoU function from gen_pseudo_label_new.py
def box_iou(box_as, box_bs):
    box_asExtend=box_as.unsqueeze(1).expand(box_as.shape[0],box_bs.shape[0],4)
    box_bsExtend = box_bs.unsqueeze(0).expand(box_as.shape[0], box_bs.shape[0], 4)
    box1 = box_asExtend
    box2 = box_bsExtend
    leftTop = torch.max(box1[..., 0:2], box2[..., 0:2])
    bottomRight = torch.min(box1[..., 2:4], box2[..., 2:4])
    b1AndB2 = torch.clamp(bottomRight - leftTop, min=0)
    b1AndB2Area = b1AndB2[..., 0:1] * b1AndB2[..., 1:2]
    b1Area = (box1[...,2:3]-box1[...,0:1])*(box1[...,3:4]-box1[...,1:2])
    b2Area = (box2[...,2:3]-box2[...,0:1])*(box2[...,3:4]-box2[...,1:2])
    return b1AndB2Area / (b1Area + b2Area - b1AndB2Area)

print("Loading VOC instances...")
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

print("Loading inference results...")
with open("training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json") as f:
    proposals = json.load(f)

for image_id in data_by_id.keys():
    bboxes_proposal = proposals.get(image_id, {}).get("bboxes", [])
    if len(bboxes_proposal) > 0:
        break

print(f"Testing with image_id: {image_id}")
img_data = data_by_id[image_id]
annos = img_data.get('annotations', [])

bboxes_proposal = torch.tensor(bboxes_proposal).reshape([-1, 4])
scores_res = torch.ones(len(bboxes_proposal))
print(f"Total proposals: {len(bboxes_proposal)}")

bboxes_anno = []
for a in annos:
    category_id = a['category_id']
    if category_id < 19:
        bbox = torch.tensor(a["bbox"]).reshape([-1,4])
        bboxes_anno.append(bbox)
        
if len(bboxes_anno) == 0:
    print("No annotations < 19!")
else:
    bboxes_anno = torch.cat(bboxes_anno, dim=0)
    print(f"Total matched annos: {len(bboxes_anno)}")
    
    iou = box_iou(bboxes_anno, bboxes_proposal).squeeze(dim = -1)
    iou_max, target_idx = iou.max(dim = 0)
    
    bboxes_target = bboxes_anno[target_idx]
    keep = iou_max <= 0.3
    
    print(f"IoU < 0.3 filter passing: {keep.sum().item()}")
    
    bboxes_proposal = bboxes_proposal[keep]
    bboxes_target = bboxes_target[keep]
    
    if len(bboxes_proposal) > 0:
        bboxes_proposal_x = ((bboxes_proposal[:, 0] + bboxes_proposal[:, 2])/2).reshape([-1,1])
        bboxes_proposal_y = ((bboxes_proposal[:, 1] + bboxes_proposal[:, 3])/2).reshape([-1,1])
        bboxes_proposal_center = torch.cat([bboxes_proposal_x, bboxes_proposal_y], dim = 1)
        
        centerness = get_centerness(bboxes_proposal_center, bboxes_target)
        keep = centerness <= 1.0
        
        print(f"Centerness <= 1.0 filter passing: {keep.sum().item()}")
        bboxes_proposal = bboxes_proposal[keep]
        print(f"Final surviving boxes: {len(bboxes_proposal)}")
    else:
        print("0 boxes survived IoU!")
