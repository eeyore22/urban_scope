import os
import json
from tqdm import tqdm

# Paths
FINAL_PATH = "./data/final_dataset.json"
UPDATED_PATH = "./data/final_dataset_merged.json"
NEW_METADATA_DIR = "./feature_outputs"

# Load original dataset
with open(FINAL_PATH, 'r') as f:
    dataset = json.load(f)

updated_dataset = []
missing = 0

for entry in tqdm(dataset):
    image_id = entry["image_id"]
    meta_path = os.path.join(NEW_METADATA_DIR, f"{image_id}_meta.json")  # or adjust suffix if different

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            new_meta = json.load(f)

        # Merge key fields into the entry
        if "depth_stats" in new_meta:
            entry["depth_stats"] = new_meta["depth_stats"]
        if "spatial_distribution" in new_meta:
            entry["spatial_distribution"] = new_meta["spatial_distribution"]
        if "vertical_layout" in new_meta:
            entry["vertical_layout"] = new_meta["vertical_layout"]

        # Optional: overwrite "objects" if you want to recompute from raw boxes
        if "objects" in new_meta:
            entry["raw_objects"] = new_meta["objects"]  # store detailed boxes if needed
            # Optionally recompute counts:
            from collections import Counter
            labels = [o["label"] for o in new_meta["objects"]]
            obj_counts = dict(Counter(labels))
            entry["objects"] = obj_counts  # replaces existing {"car": 2, "person": 4}
    else:
        missing += 1

    updated_dataset.append(entry)

# Save
with open(UPDATED_PATH, 'w') as f:
    json.dump(updated_dataset, f, indent=2)

print(f"\n✅ Updated dataset written to {UPDATED_PATH}")
print(f"⚠️ Missing metadata files: {missing}")
