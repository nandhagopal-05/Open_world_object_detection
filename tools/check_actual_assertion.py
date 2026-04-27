import json
import torch
import sys
import xml.etree.ElementTree as ET
import glob
import os

with open("training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json") as f:
    proposals = json.load(f)

print("Checking bounding box areas against image areas...")
bad_count = 0
for xml_file in glob.glob("datasets/sowod/Annotations/*.xml")[:5000]:
    image_id = os.path.basename(xml_file).replace(".xml", "")
    det = proposals.get(image_id)
    if det and len(det.get("bboxes", [])) > 0:
        tree = ET.parse(xml_file)
        size_node = tree.getroot().find("size")
        width = int(size_node.find("width").text)
        height = int(size_node.find("height").text)
        size_img = width * height
        
        bboxes = torch.tensor(det["bboxes"]).reshape([-1, 4])
        w = bboxes[:,2] - bboxes[:,0]
        h = bboxes[:,3] - bboxes[:,1]
        size = w * h
        
        if (size > size_img).any():
            bad_count += 1
            print(f"BINGO! Image {image_id} has size {width}x{height} ({size_img}). Box area is {size.max().item()}!")
            break

if bad_count == 0:
    print("Checked 5000 random images. None had out of bounds boxes.")
