import json
import torch

try:
    with open("training_dir/multimodal-mepu/sowod-t1-self-train/inference/inference_results.json") as f:
        proposals = json.load(f)
        
    print(f"Loaded {len(proposals)} images with inference results.")
    
    bad_boxes_count = 0
    for image_id, det in proposals.items():
        if len(det.get("bboxes", [])) > 0:
            bboxes = torch.tensor(det["bboxes"]).reshape([-1, 4])
            w = bboxes[:,2] - bboxes[:,0]
            h = bboxes[:,3] - bboxes[:,1]
            size = w * h
            # Assuming max possible image size is roughly 1200x1200=1440000.
            # If any box size is larger than say, 2,000,000 it is definitely a bad box
            # that would trigger the AssertionError in gen_pseudo_label_new.py!
            if (size > 2000000).any():
                bad_boxes_count += 1
                
    print(f"Found {bad_boxes_count} images containing bounding boxes with size > 2,000,000 pixels!")
    if bad_boxes_count > 0:
        print("This definitely triggered the `assert True not in mask` crash!")
    else:
        print("No outrageously large boxes found... Maybe another bug?")
except Exception as e:
    print(f"Error: {e}")
