import os
import shutil
import subprocess

# Define directories
DATA_DIR = "datasets/sowod"
COCO_DIR = "datasets/coco2017"

def main():
    # --- Step 1: Remove existing DATA_DIR if it exists ---
    if os.path.exists(DATA_DIR):
        print(f"Removing existing directory: {DATA_DIR}")
        shutil.rmtree(DATA_DIR)

    # --- Step 2: Make necessary directories ---
    print("Creating directory structure...")
    os.makedirs(os.path.join(DATA_DIR, "Annotations"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "ImageSets", "Main"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "JPEGImages"), exist_ok=True)

    # --- Step 3: Copy COCO images ---
    print("Copying COCO images...")
    train_dir = os.path.join(COCO_DIR, "train2017")
    val_dir = os.path.join(COCO_DIR, "val2017")
    jpeg_dir = os.path.join(DATA_DIR, "JPEGImages")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"COCO train2017 or val2017 not found in {COCO_DIR}")

    # Copy train2017 directory
    shutil.copytree(train_dir, os.path.join(jpeg_dir, "train2017"), dirs_exist_ok=True)
    # Copy all val2017 images into JPEGImages
    for img_file in os.listdir(val_dir):
        src = os.path.join(val_dir, img_file)
        dst = os.path.join(jpeg_dir, img_file)
        shutil.copy2(src, dst)

    # --- Step 4: Convert COCO annotations to VOC ---
    print("Converting COCO annotations to VOC format...")
    subprocess.run([
        "python", "tools/convert_coco_to_voc.py",
        "--dir", DATA_DIR,
        "--ann_path", os.path.join(COCO_DIR, "annotations", "instances_train2017.json")
    ], check=True)

    subprocess.run([
        "python", "tools/convert_coco_to_voc.py",
        "--dir", DATA_DIR,
        "--ann_path", os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
    ], check=True)

    # --- Step 5: Copy OWOD split files ---
    print("Copying OWOD split files...")
    split_dir = "./dataset_splits/s-owod/ImageSets/Main"
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    dst_split = os.path.join(DATA_DIR, "ImageSets", "Main")
    for file_name in os.listdir(split_dir):
        shutil.copy2(os.path.join(split_dir, file_name), dst_split)

    print("\n Dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
