import os
import re
import json
import random
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
from dataset.vqa_dataset import UnifiedUrbanDataset

import google.generativeai as genai

class MultimodalQAGenerator:
    def __init__(self, metadata, gemini_api_key="MY_API_KEY", enable_llm=True):
        self.meta = metadata
        self.qa_pairs = []
        self.gemini_api_key = gemini_api_key
        self.enable_llm = enable_llm
        self.llm_usage_count = 0

    def generate_all(self, image_index=0):
        self.generate_perceptual_questions()
        self.generate_reasoning_questions()
        if self.enable_llm:
            self.generate_llm_enhanced_paraphrases()
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
                {"question": f"Is there a {obj} in the image?", "answer": "Yes" if count > 0 else "No", "qtype": "perceptual", "subtype": "presence.binary", "object": obj},
                {"question": f"How many {obj}s are visible in the scene?", "answer": str(count), "qtype": "perceptual", "subtype": "count", "object": obj}
            ])

        for vf, value in view_factors.items():
            self.qa_pairs.extend([
                {"question": f"Is the scene dominated by {vf}?", "answer": "Yes" if value > 0.5 else "No", "qtype": "perceptual", "subtype": "viewfactor.dominance", "factor": vf},
                {"question": f"Does the scene have sparse {vf}?", "answer": "Yes" if value <= 0.2 else "No", "qtype": "perceptual", "subtype": "viewfactor.sparsity", "factor": vf},
                {"question": f"What is the proportion of {vf} in the scene?", "answer": f"{value:.2f}", "qtype": "perceptual", "subtype": "viewfactor.scalar", "factor": vf}
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
            for prompt in ["What is the average depth of visible buildings?", "On average, how far are the buildings in the image?"]:
                self.qa_pairs.append({"question": prompt, "answer": f"{avg:.2f}", "qtype": "perceptual", "subtype": "depth.average_per_object", "object": "building"})
            
        if "closest_object" in depth_stats:
            self.qa_pairs.append({
                "question": "Which object is closest to the camera?",
                "answer": depth_stats["closest_object"],
                "qtype": "perceptual",
                "subtype": "depth.closest_object",
                "object": depth_stats["closest_object"]
            })


        if obj_counts.get("car", 0) > 0 and obj_counts.get("person", 0) > 0:
            for q in ["Are both cars and pedestrians present?", "Do you see people and vehicles in the same scene?"]:
                self.qa_pairs.append({
                    "question": q,
                    "answer": "Yes",
                    "qtype": "perceptual",
                    "subtype": "object.cooccurrence",
                    "object1": "car",
                    "object2": "person"
                })

        for (a, b), occluded in occlusion_pairs.items():
            if (a == "pedestrian" and b == "vehicle") or (a == "tree" and b == "building"):
                for q in [f"Are the {a}s occluded by a {b}?", f"Is the view of {a}s partially blocked by a {b}?"]:
                    self.qa_pairs.append({
                        "question": q,
                        "answer": "Yes" if occluded else "No",
                        "qtype": "perceptual",
                        "subtype": "occlusion.binary",
                        "object1": a,
                        "object2": b
                    })

        for obj in ["building", "car"]:
            dist = spatial_distribution.get(obj)
            if dist:
                for q in [f"Are {obj}s mostly on the left side of the image?",
                          f"Are the {obj}s mainly concentrated on the left?",
                          f"Where are the {obj}s mostly located in the image? (e.g., left side, right side, or evenly spread)"]:
                    self.qa_pairs.append({
                        "question": q,
                        "answer": "Yes" if "left" in q.lower() and dist == "left side" else ("No" if "left" in q.lower() else dist),
                        "qtype": "perceptual",
                        "subtype": "spatial.distribution.label" if "Where" in q else "spatial.distribution",
                        "object": obj
                    })

        for obj in ["car", "building"]:
            if f"{obj}_foreground" in depth_per_class:
                self.qa_pairs.extend([
                    {"question": f"Is there a {obj} in the front of the scene?", "answer": "Yes" if depth_per_class[f"{obj}_foreground"] else "No", "qtype": "perceptual", "subtype": "depth.foreground_presence", "object": obj},
                    {"question": f"Is a {obj} visible close to the viewer?", "answer": "Yes" if depth_per_class[f"{obj}_foreground"] else "No", "qtype": "perceptual", "subtype": "depth.foreground_presence", "object": obj}
                ])
            if f"{obj}_background" in depth_per_class:
                self.qa_pairs.extend([
                    {"question": f"Are any {obj}s located far in the background?", "answer": "Yes" if depth_per_class[f"{obj}_background"] else "No", "qtype": "perceptual", "subtype": "depth.background_presence", "object": obj},
                    {"question": f"Do you see any distant {obj}s in the image?", "answer": "Yes" if depth_per_class[f"{obj}_background"] else "No", "qtype": "perceptual", "subtype": "depth.background_presence", "object": obj}
                ])

        if "skyline_clear" in vertical_layout:
            for q in ["Is the skyline clearly visible?", "Can you clearly see the skyline in the image?"]:
                self.qa_pairs.append({
                    "question": q,
                    "answer": "Yes" if vertical_layout["skyline_clear"] else "No",
                    "qtype": "perceptual",
                    "subtype": "scene.skyline_visibility"
                })
        if "top_sky" in vertical_layout:
            for q in ["Is the top of the image mostly sky?", "Does the top portion of the image consist mostly of sky?"]:
                self.qa_pairs.append({
                    "question": q,
                    "answer": "Yes" if vertical_layout["top_sky"] else "No",
                    "qtype": "perceptual",
                    "subtype": "segmentation.vertical_layout"
                })
        if "top_entity" in vertical_layout:
            self.qa_pairs.append({
                "question": "What object occupies the top part of the image?",
                "answer": vertical_layout["top_entity"],
                "qtype": "perceptual",
                "subtype": "segmentation.vertical_layout.entity"
            })

    def generate_reasoning_questions(self):
        obj_counts = self.meta.get("object_counts", {})
        view_factors = self.meta.get("view_factors", {})
        depth_order = self.meta.get("depth_order", {})

        # --- Multi-hop count comparisons ---
        if "person" in obj_counts and "car" in obj_counts:
            p, c = obj_counts["person"], obj_counts["car"]
            more = "people" if p > c else "cars"
            self.qa_pairs.extend([
                {"question": "Are there more people than cars in the image?", "answer": "Yes" if p > c else "No", "qtype": "reasoning", "subtype": "multi_hop.count_compare"},
                {"question": "Which is greater: the number of people or cars?", "answer": more, "qtype": "reasoning", "subtype": "multi_hop.which_is_more"}
            ])

        # --- Spatial relations from depth_order ---
        for a, b in [("tree", "building"), ("car", "pedestrian")]:
            if a in depth_order and b in depth_order:
                self.qa_pairs.extend([
                    {"question": f"Is the {a} in front of the {b}?", "answer": "Yes" if depth_order[a] < depth_order[b] else "No", "qtype": "reasoning", "subtype": "spatial.relation"},
                    {"question": f"Is the {a} not closer than the {b}?", "answer": "Yes" if depth_order[a] >= depth_order[b] else "No", "qtype": "reasoning", "subtype": "negation.spatial_refute"},
                    {"object1": a, "object2": b, "question": f"Are the {a}s occluded by a {b}?", "answer": "Yes" if depth_order[a] > depth_order[b] else "No", "qtype": "reasoning", "subtype": "occlusion.binary"}
                ])

        # --- Negation (absence) ---
        for obj in ["bicycle", "stop sign", "traffic light", "person"]:
            present = obj_counts.get(obj, 0) > 0
            self.qa_pairs.append({
                "question": f"Is there no {obj} visible in the scene?",
                "answer": "Yes" if not present else "No",
                "qtype": "reasoning", "subtype": "negation.absence",
                "object": obj
            })

        # --- Negation (logical conjunction) ---
        if view_factors.get("greenery", 0) < 0.2 or view_factors.get("sky", 0) < 0.2:
            self.qa_pairs.append({
                "question": "Is it incorrect to say the scene is green and open?",
                "answer": "Yes",
                "qtype": "reasoning",
                "subtype": "negation.conjunction"
            })

        # --- Negation (exclusion) ---
        excluded = next((obj for obj in ["tree", "car", "bench"] if obj_counts.get(obj, 0) == 0), None)
        if excluded:
            self.qa_pairs.append({
                "question": "Which of these is not present: a tree, a car, or a bench?",
                "answer": excluded,
                "qtype": "reasoning",
                "subtype": "negation.exclusion_choice"
            })

        # --- Counterfactuals ---
        if "person" in obj_counts:
            count = obj_counts["person"]
            self.qa_pairs.append({
                "question": "If two more people entered the scene, would the area look crowded?",
                "answer": "Yes" if count + 2 >= 5 else "No",
                "qtype": "reasoning", "subtype": "counterfactual.count_perturbation"
            })

        if view_factors.get("building", 0) > 0.3:
            self.qa_pairs.append({
                "question": "Would this scene feel more natural if buildings were not present?",
                "answer": "Yes",
                "qtype": "reasoning", "subtype": "counterfactual.absence_viewfactor",
                "factor": "building"
            })

        if view_factors.get("sky", 0) > 0.4:
            self.qa_pairs.append({
                "question": "If the sky were overcast instead of clear, would the scene feel less open?",
                "answer": "Yes",
                "qtype": "reasoning", "subtype": "counterfactual.attribute_substitution",
                "factor": "sky"
            })

        if "bus" in depth_order and "pedestrian" in depth_order:
            self.qa_pairs.append({
                "question": "If the bus were moved to the front of the scene, would it block the view?",
                "answer": "Yes",
                "qtype": "reasoning", "subtype": "counterfactual.occlusion_movement"
            })

        # --- Composite negation (inlined) ---
        gvi = view_factors.get("greenery", 0)
        bvf = view_factors.get("building", 0)
        svf = view_factors.get("sky", 0)

        if gvi >= 0.4 and bvf >= 0.4:
            self.qa_pairs.extend([
                {"question": "Is it incorrect to say the scene has fairly many greenery with fairly many buildings?", "answer": "No", "qtype": "reasoning", "subtype": "negation.composite_conjunction_1"},
                {"question": "Would it be wrong to describe the area as both green and built-up?", "answer": "No", "qtype": "reasoning", "subtype": "negation.composite_conjunction_1"},
                {"question": "Is it false to claim that there is a lot of vegetation and many buildings here?", "answer": "No", "qtype": "reasoning", "subtype": "negation.composite_conjunction_1"}
            ])

        if svf >= 0.5 and gvi <= 0.3:
            self.qa_pairs.extend([
                {"question": "Is it incorrect to say the scene has open sky but also with sparse greenery?", "answer": "No", "qtype": "reasoning", "subtype": "negation.composite_conjunction_2"},
                {"question": "Would it be inaccurate to describe the scene as open-sky but lacking in greenery?", "answer": "No", "qtype": "reasoning", "subtype": "negation.composite_conjunction_2"},
                {"question": "Is it wrong to claim that there's plenty of visible sky but not much greenery?", "answer": "No", "qtype": "reasoning", "subtype": "negation.composite_conjunction_2"}
            ])

        if bvf <= 0.3 and svf <= 0.3:
            self.qa_pairs.extend([
                {"question": "Is it incorrect to say the scene has a few buildings yet the sky is still not too visible?", "answer": "No", "qtype": "reasoning", "subtype": "negation.composite_conjunction_3"},
                {"question": "Would it be wrong to describe the area as low-rise and enclosed by the sky?", "answer": "No", "qtype": "reasoning", "subtype": "negation.composite_conjunction_3"},
                {"question": "Is it false to say that neither buildings nor sky dominate this scene?", "answer": "No", "qtype": "reasoning", "subtype": "negation.composite_conjunction_3"}
            ])


    def generate_llm_enhanced_paraphrases(self, max_paraphrases=10):
        if not self.gemini_api_key or not genai:
            return
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        random.shuffle(self.qa_pairs)
        added = 0

        for qa in self.qa_pairs:
            prompt = f"""
                        You are helping refine a dataset of urban scene question-answer pairs for a visual question answering (VQA) task. 
                        Paraphrase this question while keeping the answer identical so that it
                        - Sounds more natural and fluent
                        - Preserves the original meaning and answer
                        - Stays grounded in the visible content of the image
                        - Uses diverse syntax or style (e.g., passive voice, rhetorical form, clause rearrangement)

                        Do **not** introduce new information that is not inferable from the image.

                        Original: {qa['question']}
                        Answer: {qa['answer']}
                        Output ONE natural rephrasing only. Do not explain."""

            try:
                response = model.generate_content(prompt)
                output = response.text.strip()
                if output:
                    new_qa = {
                        **qa,
                        "question": output,
                        "qtype": "llm",
                        "subtype": "paraphrase.from_" + qa["subtype"],
                        "origin_subtype": qa["subtype"]
                    }
                    self.qa_pairs.append(new_qa)
                    added += 1
                    if added >= max_paraphrases:
                        break
            except Exception as e:
                print("Gemini error:", e)

# Worker Setup
_shared_dataset = None
def _init_worker(path, load_depth):
    global _shared_dataset
    _shared_dataset = UnifiedUrbanDataset(path, load_depth=load_depth)

def _task_wrapper(idx):
    global _shared_dataset
    features = _shared_dataset[idx][1]
    image_id = _shared_dataset.data[idx]["image_id"]
    generator = MultimodalQAGenerator(features, gemini_api_key=os.getenv("MY_API_KEY"), enable_llm=True)
    qa_pairs = generator.generate_all(image_index=idx)
    return [{"image_id": image_id, **qa} for qa in qa_pairs]

# Main builder
def build_qa_dataset_parallel(consolidated_path, output_path, load_depth=True, target_image_ids=None):
    temp_dataset = UnifiedUrbanDataset(consolidated_path, load_depth=load_depth)
    args_list = [i for i in range(len(temp_dataset)) if target_image_ids is None or temp_dataset.data[i]["image_id"] in target_image_ids]
    del temp_dataset
    
    print(f"🔄 Generating QA for {len(args_list)} images using {multiprocessing.cpu_count()} workers...")

    existing_questions = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as fin:
            for line in fin:
                try:
                    item = json.loads(line)
                    existing_questions.add((item['image_id'], item['question']))
                except:
                    continue
        print(f"Loaded {len(existing_questions):,} dedup keys")

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
                    #fout.write('\\n'.join(buffer) + '\\n')
                    fout.write('\n'.join(buffer) + '\n')
                    fout.flush()
                    buffer = []

            if buffer:
                fout.write('\\n'.join(buffer) + '\\n')
                fout.flush()

    print(f"✅ Done. QA dataset written to {output_path}")


# --- Entry Point ---
if __name__ == "__main__":
    import random

    # Load full dataset
    temp_dataset = UnifiedUrbanDataset("./data/legacy/final_dataset_2.0.0.json", load_depth=True)
    all_image_ids = [item["image_id"] for item in temp_dataset.data]
    del temp_dataset

    # Sample 50K unique image IDs
    subset_image_ids = set(random.sample(all_image_ids, k=50000))

    # Optional: Save this for reproducibility
    with open("./data/image_ids_10k.json", "w") as f:
        json.dump(list(subset_image_ids), f, indent=2)

    # Build QA dataset using just the 10K subset
    build_qa_dataset_parallel(
        consolidated_path="./data/legacy/final_dataset_2.0.0.json",
        output_path="./qa_dataset_50k_v3.jsonl",
        target_image_ids=subset_image_ids,
        load_depth=True
    )
