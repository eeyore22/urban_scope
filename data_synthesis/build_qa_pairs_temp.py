import json
import os
import multiprocessing
from tqdm import tqdm
from dataset.vqa_dataset import UnifiedUrbanDataset

# --- QA Generator ---
class MultimodalQAGenerator:
    def __init__(self, metadata):
        self.meta = metadata
        self.qa_pairs = []

    def generate_all(self):
        self.generate_perceptual_questions()
        return self.qa_pairs

    def generate_perceptual_questions(self):
        obj_counts = self.meta.get("object_counts", {})
        view_factors = self.meta.get("view_factors", {})
        depth_stats = self.meta.get("depth_stats", {})
        depth_per_class = self.meta.get("depth_per_class", {})
        occlusion_pairs = self.meta.get("occlusion_pairs", {})
        spatial_distribution = self.meta.get("spatial_distribution", {})
        vertical_layout = self.meta.get("vertical_layout", {})

        for obj, count in obj_counts.items():
            self.qa_pairs.extend([
                {"question": f"Is there a {obj} in the image?", "answer": "Yes" if count > 0 else "No", "qtype": "perceptual", "subtype": "presence.binary"},
                {"question": f"How many {obj}s are visible in the scene?", "answer": str(count), "qtype": "perceptual", "subtype": "count"}
            ])

        for vf, value in view_factors.items():
            self.qa_pairs.extend([
                {"question": f"Is the scene dominated by {vf}?", "answer": "Yes" if value > 0.5 else "No", "qtype": "perceptual", "subtype": "viewfactor.dominance"},
                {"question": f"Does the scene have sparse {vf}?", "answer": "Yes" if value <= 0.2 else "No", "qtype": "perceptual", "subtype": "viewfactor.sparsity"},
                {"question": f"What is the proportion of {vf} in the scene?", "answer": f"{value:.2f}", "qtype": "perceptual", "subtype": "viewfactor.scalar"}
            ])

        if "range" in depth_stats:
            rng = depth_stats["range"]
            label = "high" if rng > 40 else "moderate" if rng > 20 else "low"
            self.qa_pairs.extend([
                {"question": "Does the scene appear visually complex?", "answer": "Complex" if rng > 20 else "Simple", "qtype": "perceptual", "subtype": "depth.range"},
                {"question": "Is there a large depth variation in the scene?", "answer": "Yes" if rng > 20 else "No", "qtype": "perceptual", "subtype": "depth.variation"},
                {"question": "Does the depth range suggest a complex scene layout?", "answer": "Yes" if rng > 20 else "No", "qtype": "perceptual", "subtype": "depth.variation"},
                {"question": "How would you describe the depth complexity of this scene? (e.g., low, moderate, or high)", "answer": label, "qtype": "perceptual", "subtype": "depth.range.label"}
            ])

        if "building" in depth_per_class:
            avg = depth_per_class["building"]
            self.qa_pairs.extend([
                {"question": "What is the average depth of visible buildings?", "answer": f"{avg:.2f}", "qtype": "perceptual", "subtype": "depth.average_per_object"},
                {"question": "On average, how far are the buildings in the image?", "answer": f"{avg:.2f}", "qtype": "perceptual", "subtype": "depth.average_per_object"}
            ])

        if "closest_object" in depth_stats:
            self.qa_pairs.append({
                "question": "Which object is closest to the camera?",
                "answer": depth_stats["closest_object"],
                "qtype": "perceptual",
                "subtype": "depth.closest_object"
            })

        if obj_counts.get("car", 0) > 0 and obj_counts.get("person", 0) > 0:
            self.qa_pairs.extend([
                {"question": "Are both cars and pedestrians present?", "answer": "Yes", "qtype": "perceptual", "subtype": "object.cooccurrence"},
                {"question": "Do you see people and vehicles in the same scene?", "answer": "Yes", "qtype": "perceptual", "subtype": "object.cooccurrence"}
            ])

        for (a, b), occluded in occlusion_pairs.items():
            if (a == "pedestrian" and b == "vehicle") or (a == "tree" and b == "building"):
                for q in [f"Are the {a}s occluded by a {b}?", f"Is the view of {a}s partially blocked by a {b}?"]:
                    self.qa_pairs.append({
                        "question": q,
                        "answer": "Yes" if occluded else "No",
                        "qtype": "perceptual",
                        "subtype": "occlusion.binary"
                    })

        for obj in ["building", "car"]:
            dist = spatial_distribution.get(obj)
            if dist:
                self.qa_pairs.extend([
                    {"question": f"Are {obj}s mostly on the left side of the image?", "answer": "Yes" if dist == "left side" else "No", "qtype": "perceptual", "subtype": "spatial.distribution"},
                    {"question": f"Are the {obj}s mainly concentrated on the left?", "answer": "Yes" if dist == "left side" else "No", "qtype": "perceptual", "subtype": "spatial.distribution"},
                    {"question": f"Where are the {obj}s mostly located in the image? (e.g., left side, right side, or evenly spread)", "answer": dist, "qtype": "perceptual", "subtype": "spatial.distribution.label"}
                ])

        for obj in ["car", "building"]:
            if f"{obj}_foreground" in depth_per_class:
                self.qa_pairs.extend([
                    {"question": f"Is there a {obj} in the front of the scene?", "answer": "Yes" if depth_per_class[f"{obj}_foreground"] else "No", "qtype": "perceptual", "subtype": "depth.foreground_presence"},
                    {"question": f"Is a {obj} visible close to the viewer?", "answer": "Yes" if depth_per_class[f"{obj}_foreground"] else "No", "qtype": "perceptual", "subtype": "depth.foreground_presence"}
                ])
            if f"{obj}_background" in depth_per_class:
                self.qa_pairs.extend([
                    {"question": f"Are any {obj}s located far in the background?", "answer": "Yes" if depth_per_class[f"{obj}_background"] else "No", "qtype": "perceptual", "subtype": "depth.background_presence"},
                    {"question": f"Do you see any distant {obj}s in the image?", "answer": "Yes" if depth_per_class[f"{obj}_background"] else "No", "qtype": "perceptual", "subtype": "depth.background_presence"}
                ])

        if "skyline_clear" in vertical_layout:
            self.qa_pairs.extend([
                {"question": "Is the skyline clearly visible?", "answer": "Yes" if vertical_layout["skyline_clear"] else "No", "qtype": "perceptual", "subtype": "scene.skyline_visibility"},
                {"question": "Can you clearly see the skyline in the image?", "answer": "Yes" if vertical_layout["skyline_clear"] else "No", "qtype": "perceptual", "subtype": "scene.skyline_visibility"}
            ])
        if "top_sky" in vertical_layout:
            self.qa_pairs.extend([
                {"question": "Is the top of the image mostly sky?", "answer": "Yes" if vertical_layout["top_sky"] else "No", "qtype": "perceptual", "subtype": "segmentation.vertical_layout"},
                {"question": "Does the top portion of the image consist mostly of sky?", "answer": "Yes" if vertical_layout["top_sky"] else "No", "qtype": "perceptual", "subtype": "segmentation.vertical_layout"}
            ])
        if "top_entity" in vertical_layout:
            self.qa_pairs.append({
                "question": "What object occupies the top part of the image?",
                "answer": vertical_layout["top_entity"],
                "qtype": "perceptual",
                "subtype": "segmentation.vertical_layout.entity"
            })


# --- Global Dataset Ref (One Per Worker) ---
_dataset_ref = None

def _init_worker(dataset_path, load_depth):
    global _dataset_ref
    _dataset_ref = UnifiedUrbanDataset(dataset_path, load_depth=load_depth)

def _task_wrapper(idx):
    _, features = _dataset_ref[idx]
    image_id = _dataset_ref.data[idx]["image_id"]
    generator = MultimodalQAGenerator(features)
    qa_pairs = generator.generate_all()
    return [{"image_id": image_id, **qa} for qa in qa_pairs]


# --- Main Parallel QA Builder ---
def build_qa_dataset_parallel(consolidated_path, output_path, load_depth=True, target_image_ids=None):
    temp_dataset = UnifiedUrbanDataset(consolidated_path, load_depth=load_depth)
    args_list = [i for i in range(len(temp_dataset)) if target_image_ids is None or temp_dataset.data[i]["image_id"] in target_image_ids]
    del temp_dataset  # avoid pickling it

    print(f"🔄 Generating QA for {len(args_list)} images using {multiprocessing.cpu_count()} workers...")

    # Load dedup keys
    existing_questions = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as fin:
            for line in fin:
                try:
                    item = json.loads(line)
                    existing_questions.add((item['image_id'], item['question']))
                except:
                    continue
        print(f"✔️ Loaded {len(existing_questions):,} dedup keys")

    # Streamed + buffered writing
    with open(output_path, 'a') as fout:
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=_init_worker,
            initargs=(consolidated_path, load_depth)
        ) as pool:
            buffer = []
            for result in tqdm(pool.imap_unordered(_task_wrapper, args_list), total=len(args_list)):
                for qa in result:
                    key = (qa["image_id"], qa["question"])
                    if key not in existing_questions:
                        buffer.append(json.dumps(qa))
                        existing_questions.add(key)

                if len(buffer) >= 500:
                    fout.write('\n'.join(buffer) + '\n')
                    fout.flush()
                    buffer = []

            if buffer:
                fout.write('\n'.join(buffer) + '\n')
                fout.flush()

    print(f"✅ Done. QA dataset written to {output_path}.")


# --- Entry Point ---
if __name__ == "__main__":
    subset_path = "./qa_dataset_50k.jsonl"
    subset_image_ids = set()
    if os.path.exists(subset_path):
        with open(subset_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    subset_image_ids.add(item["image_id"])
                except:
                    continue

    build_qa_dataset_parallel(
        consolidated_path="./data/final_dataset_2.0.0.json",
        output_path="./qa_dataset_50k.jsonl",
        target_image_ids=subset_image_ids
    )


