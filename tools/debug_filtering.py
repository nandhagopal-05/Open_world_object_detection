import json
import torch
import torchvision

# Load inference results
with open("training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json") as f:
    proposals = json.load(f)

# Pick an image that has boxes
image_id = None
for k, v in proposals.items():
    if len(v.get("bboxes", [])) > 0:
        image_id = k
        break

if not image_id:
    print("No boxes in any image!")
    exit(0)

print(f"Testing with image_id: {image_id}")
bboxes = torch.tensor(proposals[image_id]["bboxes"]).reshape([-1, 4])
if "scores" in proposals[image_id]:
    scores = torch.tensor(proposals[image_id]["scores"])
else:
    print("No scores found. Using dummy scores.")
    scores = torch.ones(len(bboxes))

print(f"Initial boxes: {len(bboxes)}")

w = bboxes[:,2] - bboxes[:,0]
h = bboxes[:,3] - bboxes[:,1]
size = w * h

# Let's assume a typical image size like 800x600 for tests
size_img = 800 * 600

mask1 = size <= 0.98 * size_img
mask2 = w / h <= 4
mask3 = h / w <= 4
mask4 = size >= 2000

print(f"Passed mask1 (size <= 0.98*size_img): {mask1.sum().item()}")
print(f"Passed mask2 (w/h <= 4): {mask2.sum().item()}")
print(f"Passed mask3 (h/w <= 4): {mask3.sum().item()}")
print(f"Passed mask4 (size >= 2000): {mask4.sum().item()}")

mask = mask1 & mask2 & mask3 & mask4
print(f"Passed all size masks: {mask.sum().item()}")

bboxes = bboxes[mask]
scores = scores[mask]

output = torchvision.ops.nms(boxes=bboxes.float(), scores=scores, iou_threshold=0.3)
print(f"Passed NMS (iou_thr=0.3): {len(output)}")
