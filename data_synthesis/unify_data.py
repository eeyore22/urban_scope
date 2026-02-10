import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm

# Input paths
IMAGE_DIR = "./data/all_images"
FEATURE_DIR = "./feature_outputs"
PROPORTION_FILE = "./data/gt_proportions_final.json"
OUTPUT_FILE = "./final_dataset.json"

# Helper to parse image_id
def extract_image_id(filepath):
    filename = os.path.basename(filepath)
    image_id = os.path.splitext(filename)[0]
    return image_id

# Load proportions
with open(PROPORTION_FILE, 'r') as f:
    proportion_data = [json.loads(line) for line in f]

# Build a lookup map for proportions
proportion_map = {}
for item in proportion_data:
    # Extract correct image id from path
    image_path = item["image_file"]
    image_filename = os.path.basename(image_path)
    image_id = os.path.splitext(image_filename)[0]
    proportion_map[image_id] = {
        "greenery": item["proportions"]["greenery"],
        "sky": item["proportions"]["sky"],
        "building": item["proportions"]["building"]
    }

# Build unified dataset
dataset = []

for image_path in tqdm(glob(os.path.join(IMAGE_DIR, "*.jpg"))):
    image_id = extract_image_id(image_path)

    # Build file paths
    depth_path = os.path.join(FEATURE_DIR, f"{image_id}_depth.npy")
    meta_path = os.path.join(FEATURE_DIR, f"{image_id}_meta.json")

    # Load object detection meta
    if not os.path.exists(meta_path):
        print(f"Skipping {image_id} (missing meta file)")
        continue

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Count objects (cars & persons only)
    car_count = sum(1 for obj in meta["objects"] if obj["label"] == "car")
    person_count = sum(1 for obj in meta["objects"] if obj["label"] == "person")

    # Get view factors
    if image_id not in proportion_map:
        print(f"Missing proportions for {image_id}")
        continue

    view_factors = proportion_map[image_id]

    dataset.append({
        "image_id": image_id,
        "image_path": image_path,
        "depth_path": depth_path,
        "objects": {
            "car": car_count,
            "person": person_count
        },
        "view_factors": view_factors
    })

# Save final dataset
with open(OUTPUT_FILE, 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"\n✅ Consolidated dataset saved to {OUTPUT_FILE}")
print(f"Total samples: {len(dataset)}")
