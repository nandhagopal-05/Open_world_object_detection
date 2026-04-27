import json

print("Reading pseudo_label_st_filtered.json...")
try:
    with open('datasets/sowod/Annotations/pseudo_label_st_filtered.json') as f:
        data = json.load(f)
    
    counts = {}
    total_boxes = 0
    for file_id, info in data.items():
        if 'classes' in info:
            for cls in info['classes']:
                counts[cls] = counts.get(cls, 0) + 1
                total_boxes += 1
                
    print(f"Total images with pseudo-labels: {len(data)}")
    print(f"Total pseudo-bounding boxes: {total_boxes}")
    print("Class distribution:")
    for cls in sorted(counts.keys()):
        print(f"  Class {cls}: {counts[cls]} boxes")
        
except FileNotFoundError:
    print("File not found! pseudo_label_st_filtered.json is missing.")
    
print("\nReading pseudo_label_st_initial.json (Before filtering)...")
try:
    with open('datasets/sowod/Annotations/pseudo_label_st_initial.json') as f:
        data_initial = json.load(f)
        
    counts_init = {}
    total_boxes_init = 0
    for file_id, info in data_initial.items():
        if 'classes' in info:
            for cls in info['classes']:
                counts_init[cls] = counts_init.get(cls, 0) + 1
                total_boxes_init += 1
                
    print(f"Total images with initial proposals: {len(data_initial)}")
    print(f"Total initial bounding boxes: {total_boxes_init}")
    print("Class distribution (Initial):")
    for cls in sorted(counts_init.keys()):
        print(f"  Class {cls}: {counts_init[cls]} boxes")
except FileNotFoundError:
    print("File not found! pseudo_label_st_initial.json is missing.")
