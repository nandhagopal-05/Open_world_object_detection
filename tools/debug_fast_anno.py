import json
import torch
import torchvision
import sys
sys.path.append(".")
import xml.etree.ElementTree as ET
from mepu.utils.utils import get_centerness

print("Loading one XML annotation...")
xml_path = "datasets/sowod/Annotations/000000558840.xml"
tree = ET.parse(xml_path)
root = tree.getroot()

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

bboxes_anno = []
for obj in root.findall("object"):
    name = obj.find("name").text
    if name in ALL_CLS_NAMES_OWDETR:
        category_id = ALL_CLS_NAMES_OWDETR.index(name)
        if category_id < 19:
            bndbox = obj.find("bndbox")
            bbox = [
                float(bndbox.find("xmin").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymax").text),
            ]
            bboxes_anno.append(torch.tensor(bbox).reshape([-1, 4]))

if len(bboxes_anno) == 0:
    print("No annotations < 19 in this image!")
    exit(0)

bboxes_anno = torch.cat(bboxes_anno, dim=0)
print(f"Total matched annos: {len(bboxes_anno)}")

print("Loading inference results...")
with open("training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json") as f:
    proposals = json.load(f)

bboxes_proposal = proposals.get("000000558840", {}).get("bboxes", [])
bboxes_proposal = torch.tensor(bboxes_proposal).reshape([-1, 4])
print(f"Total proposals before NMS: {len(bboxes_proposal)}")

# Apply NMS mask from earlier
w = bboxes_proposal[:,2] - bboxes_proposal[:,0]
h = bboxes_proposal[:,3] - bboxes_proposal[:,1]
size = w * h
size_img = 800 * 600
mask1 = size <= 0.98 * size_img
mask2 = w / h <= 4
mask3 = h / w <= 4
mask4 = size >= 2000
mask = mask1 & mask2 & mask3 & mask4
bboxes_proposal = bboxes_proposal[mask]
output = torchvision.ops.nms(boxes=bboxes_proposal.float(), scores=torch.ones(len(bboxes_proposal)), iou_threshold=0.3)
bboxes_proposal = bboxes_proposal[output]
print(f"Total proposals after NMS: {len(bboxes_proposal)}")

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

iou = box_iou(bboxes_anno, bboxes_proposal).squeeze(dim = -1)
print(f"IoU matrix shape: {iou.shape}")
iou_max, target_idx = iou.max(dim = 0)

bboxes_target = bboxes_anno[target_idx]
keep = iou_max <= 0.3

print(f"IoU <= 0.3 filter passing: {keep.sum().item()}")

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
