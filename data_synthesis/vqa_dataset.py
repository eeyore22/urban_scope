import os, json
import torch
from torch.utils.data import Dataset
from PIL import Image

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class UnifiedUrbanDataset(Dataset):
    def __init__(self, json_path, load_depth=True):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.load_depth = load_depth

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image_path"]).convert("RGB")

        if self.load_depth:
            depth = np.load(item["depth_path"])
            depth = torch.tensor(depth).unsqueeze(0).float()
        else:
            depth = None

        # Return full metadata dict used for QA generation
        metadata = {
            "view_factors": item.get("view_factors", {}),
            "object_counts": item.get("objects", {}),
            "depth_stats": item.get("depth_stats", {}),
            "depth_per_class": item.get("depth_per_class", {}),
            "occlusion_pairs": item.get("occlusion_pairs", {}),
            "spatial_distribution": item.get("spatial_distribution", {}),
            "vertical_layout": item.get("vertical_layout", {}),
        }

        if depth is not None:
            metadata["depth"] = depth

        return image, metadata


import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

class VQADataset(Dataset):
    def __init__(self, image_root, qa_file, feature_root=None, load_depth=True):
        self.image_root = image_root
        self.feature_root = feature_root
        self.load_depth = load_depth
        self.qa_pairs = []

        with open(qa_file, 'r') as f:
            for line in f:
                qa = json.loads(line)
                # Ensure .jpg extension is present just once
                if not qa["image_id"].endswith(".jpg"):
                    qa["image_id"] += ".jpg"
                self.qa_pairs.append(qa)

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]
        image_id = qa["image_id"]
        question = qa["question"]
        answer = qa["answer"]
        qtype = qa["qtype"]
        subtype = qa.get("subtype", "")  # <- NEW

        img_path = os.path.join(self.image_root, image_id)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]

        # Load optional depth features
        if self.load_depth and self.feature_root:
            depth_path = os.path.join(self.feature_root, image_id.replace(".jpg", "_depth.npy"))
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
                depth = np.expand_dims(depth, axis=0).astype(np.float32)
            else:
                depth = np.zeros((1, 256, 256), dtype=np.float32)
        else:
            depth = np.zeros((1, 256, 256), dtype=np.float32)

        return {
            "image": pixel_values,
            "question": question,
            "answer": answer,
            "qtype": qtype,
            "subtype": subtype,            # <- NEW
            "depth": depth
        }
