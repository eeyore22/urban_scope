import os
import gc
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from collections import Counter
import warnings
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TRANSFORMERS_NO_TF_WARNING'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# === Modules ===
class DepthEstimationModule:
    def __init__(self, device):
        self.device = device
        self.model = midas.to(device).eval()
        self.transform = midas_transform

    def predict_batch(self, images):
        depths = []
        for img in images:
            # Ensure NumPy format for MiDaS transform
            if isinstance(img, Image.Image):
                img_np = np.array(img).astype(np.uint8)
            elif isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                img_np = img

            transformed = self.transform(img_np)  # Apply MiDaS transform
            if isinstance(transformed, dict):
                input_tensor = transformed["image"]
            else:
                input_tensor = transformed

            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                pred = self.model(input_tensor)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=img_np.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze().cpu().numpy()
            depths.append(pred)
        return depths


class ObjectDetectionModule:
    def __init__(self, device):
        self.device = device
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
        self.threshold = 0.5

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.softmax(-1)[0, :, :-1]
        scores, labels = logits.max(-1)
        boxes = outputs.pred_boxes[0].cpu().numpy()
        image_w, image_h = image.size

        results = []
        for score, label, box in zip(scores, labels, boxes):
            if score < self.threshold:
                continue
            cx, cy, w, h = box
            x_min = (cx - w/2) * image_w
            y_min = (cy - h/2) * image_h
            x_max = (cx + w/2) * image_w
            y_max = (cy + h/2) * image_h
            results.append({
                "box": [x_min, y_min, x_max, y_max],
                "label": self.model.config.id2label[label.item()],
                "score": score.item()
            })
        return results

class SegFormerSegmentationModule:
    def __init__(self, device):
        self.device = device
        self.processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(device).eval()

    def predict(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=pil_image.size[::-1],
                mode='bilinear',
                align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        return predicted

# === Metadata Computation ===
def compute_metadata(segmentation_mask, depth_map, detections, class_names, object_list=["building", "tree", "car", "person"]):
    metadata = {}

    # Depth stats
    # MiDaS로 예측한 depth map은 각 픽셀마다 상대적인 거리 (relative depth) 값을 가진다
    # 이 depth 값중 0 보다 큰 값만 모아서 가장 멀리있는 픽셀의 값 (max)와 가장 가까운 픽셀의 값 (min)의 차이를 계산한다
    # 즉, range = (가장 먼 거리) - (가장 가까운 거리)
    # 결과적으로 depth_stats.range는 장면 전체가 얼마나 깊이 변화 (depth variation)를 가지고 있는지를 나타내는 지표이다.

    valid_depth = depth_map[depth_map > 0]
    metadata["depth_stats"] = {"range": float(valid_depth.max() - valid_depth.min()) if valid_depth.size > 0 else 0.0}

    avg_depths = {}
    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = map(int, det["box"])
        box_depth = depth_map[y1:y2, x1:x2]
        valid = box_depth[box_depth > 0]
        if valid.size > 0:
            avg_depths[label] = valid.mean()
    if avg_depths:
        closest = min(avg_depths.items(), key=lambda x: x[1])[0]
        metadata["depth_stats"]["closest_object"] = closest

    H, W = segmentation_mask.shape
    spatial_distribution = {}
    for obj in object_list:
        obj_indices = [k for k, v in class_names.items() if v == obj]
        if not obj_indices:
            continue
        obj_mask = np.isin(segmentation_mask, obj_indices)
        total = obj_mask.sum()
        if total == 0:
            continue
        left = obj_mask[:, :W//2].sum()
        ratio_left = left / total
        if ratio_left > 0.6:
            label = "left side"
        elif ratio_left < 0.4:
            label = "right side"
        else:
            label = "even"
        spatial_distribution[obj] = label
    if spatial_distribution:
        metadata["spatial_distribution"] = spatial_distribution

    top_region = segmentation_mask[:int(H * 0.2), :]
    if top_region.size > 0:
        top_labels = top_region.flatten()
        label_counts = Counter(top_labels)
        top_class_idx = label_counts.most_common(1)[0][0]
        top_class_name = class_names.get(top_class_idx, "unknown")
        metadata["vertical_layout"] = {"top_entity": top_class_name}

    return metadata

# === Pipeline ===
def run_pipeline(rank, world_size, input_dir, output_dir, batch_size=1):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    files.sort()
    files = files[rank::world_size]

    depth_estimator = DepthEstimationModule(device)
    detector = ObjectDetectionModule(device)
    segmentor = SegFormerSegmentationModule(device)

    for fname in tqdm(files, desc=f"Rank {rank}"):
        prefix = os.path.splitext(fname)[0]
        json_path = os.path.join(output_dir, f"{prefix}_meta.json")
        depth_path = os.path.join(output_dir, f"{prefix}_depth.npy")

        if os.path.exists(depth_path):
            depth_map = np.load(depth_path)
        else:
            depth_map = depth_estimator.predict_batch([image])[0]
            np.save(depth_path, depth_map)

        # Load or initialize metadata
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                meta = json.load(f)
        else:
            meta = {"objects": objects}


        image_path = os.path.join(input_dir, fname)
        image = Image.open(image_path).convert("RGB")
        depth_map = depth_estimator.predict_batch([image])[0]
        objects = detector.predict(image_path)
        segmentation = segmentor.predict(image)

        class_names = segmentor.model.config.id2label
        meta = {"objects": objects}
        computed = compute_metadata(segmentation, depth_map, objects, class_names)
        meta.update(computed)

        os.makedirs(output_dir, exist_ok=True)
        np.save(depth_path, depth_map)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        torch.cuda.empty_cache()
        gc.collect()

import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    #input_dir = "./data/all_images"
    input_dir = "./sample_images"
    output_dir = "./sample_feature_outputs"
    os.makedirs(output_dir, exist_ok=True)

    run_pipeline(rank, world_size, input_dir, output_dir, batch_size=1)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()

# PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,6,7 torchrun --nproc-per-node=3 dataset/extract_features.py