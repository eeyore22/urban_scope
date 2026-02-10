import json
import random
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from dataset.vqa_dataset import UnifiedUrbanDataset

# --- QA Generator using Gemini 1.5 Flash ---
class MultimodalQAGenerator:
    def __init__(self, metadata, gemini_api_key, llm_budget_usd=290.0):
        self.meta = metadata
        self.qa_pairs = []
        self.gemini_api_key = gemini_api_key
        self.llm_budget_usd = llm_budget_usd
        self.llm_usage_count = 0
        self.llm_max_calls = 50  # budget cap

    def generate_all(self):
        self.generate_basic_questions()
        self.generate_advanced_questions()
        if self.gemini_api_key:
            self.generate_llm_enhanced_paraphrases()
        return self.qa_pairs

    def generate_basic_questions(self):
        obj_counts = self.meta.get("object_counts", {})
        view_factors = self.meta.get("view_factors", {})
        depth_stats = self.meta.get("depth_stats", {})

        for obj, count in obj_counts.items():
            self.qa_pairs.extend([
                {"question": f"Is there a {obj} in the image?", "answer": "Yes" if count > 0 else "No", "qtype": "basic", "subtype": "presence.binary"},
                {"question": f"How many {obj}s are visible in the scene?", "answer": str(count), "qtype": "basic", "subtype": "count"}
            ])

        for vf, value in view_factors.items():
            self.qa_pairs.extend([
                {"question": f"Is the scene dominated by {vf}?", "answer": "Yes" if value > 0.3 else "No", "qtype": "basic", "subtype": "viewfactor.dominance"},
                {"question": f"What is the proportion of {vf} in the scene?", "answer": f"{value:.2f}", "qtype": "basic", "subtype": "viewfactor.scalar"}
            ])

        if "range" in depth_stats:
            rng = depth_stats["range"]
            label = "Deep" if rng > 20 else "Shallow"
            self.qa_pairs.append({"question": "Does the scene appear deep or shallow?", "answer": label, "qtype": "basic", "subtype": "depth.range"})

    def generate_advanced_questions(self):
        obj_counts = self.meta.get("object_counts", {})
        view_factors = self.meta.get("view_factors", {})
        depth_order = self.meta.get("depth_order", {})

        if "person" in obj_counts and "car" in obj_counts:
            p, c = obj_counts["person"], obj_counts["car"]
            more = "People" if p > c else "Cars"
            self.qa_pairs.extend([
                {"question": "Are there more people than cars in the image?", "answer": "Yes" if p > c else "No", "qtype": "advanced", "subtype": "multi_hop.count_compare"},
                {"question": "Which is greater: the number of people or cars?", "answer": more, "qtype": "advanced", "subtype": "multi_hop.which_is_more"}
            ])

        for a, b in [("tree", "building"), ("car", "pedestrian")]:
            if a in depth_order and b in depth_order:
                self.qa_pairs.extend([
                    {"question": f"Is the {a} in front of the {b}?", "answer": "Yes" if depth_order[a] < depth_order[b] else "No", "qtype": "advanced", "subtype": "spatial.relation"},
                    {"question": f"Is the {a} not closer than the {b}?", "answer": "Yes" if depth_order[a] >= depth_order[b] else "No", "qtype": "advanced", "subtype": "negation.spatial_refute"}
                ])

        for obj in ["bicycle", "stop sign", "traffic light", "person"]:
            present = obj_counts.get(obj, 0) > 0
            self.qa_pairs.append({
                "question": f"Is there no {obj} visible in the scene?",
                "answer": "Yes" if not present else "No",
                "qtype": "advanced", "subtype": "negation.absence"
            })

        if view_factors.get("greenery", 0) < 0.2 or view_factors.get("sky", 0) < 0.2:
            self.qa_pairs.append({
                "question": "Is it incorrect to say the scene is green and open?",
                "answer": "Yes",
                "qtype": "advanced", "subtype": "negation.conjunction"
            })

        if "person" in obj_counts:
            count = obj_counts["person"]
            self.qa_pairs.append({
                "question": "If two more people entered the scene, would the area look crowded?",
                "answer": "Yes" if count + 2 >= 5 else "No",
                "qtype": "advanced", "subtype": "counterfactual.count_perturbation"
            })

        if view_factors.get("building", 0) > 0.3:
            self.qa_pairs.append({
                "question": "Would this scene feel more natural if buildings were not present?",
                "answer": "Yes",
                "qtype": "advanced", "subtype": "counterfactual.absence_viewfactor"
            })

        if view_factors.get("sky", 0) > 0.4:
            self.qa_pairs.append({
                "question": "If the sky were overcast instead of clear, would the scene feel less open?",
                "answer": "Yes",
                "qtype": "advanced", "subtype": "counterfactual.attribute_substitution"
            })

        if "bus" in depth_order and "pedestrian" in depth_order:
            self.qa_pairs.append({
                "question": "If the bus were moved to the front of the scene, would it block the view?",
                "answer": "Yes",
                "qtype": "advanced", "subtype": "counterfactual.occlusion_movement"
            })

        self.qa_pairs.append({
            "question": "Which of these is not present: a tree, a car, or a bench?",
            "answer": next(obj for obj in ["tree", "car", "bench"] if obj_counts.get(obj, 0) == 0),
            "qtype": "advanced", "subtype": "negation.exclusion_choice"
        })

        self.append_composite_negation_questions()

    def append_composite_negation_questions(self):
        vf = self.meta.get("view_factors", {})
        gvi = vf.get("greenery", 0)
        bvf = vf.get("building", 0)
        svf = vf.get("sky", 0)

        if gvi >= 0.4 and bvf >= 0.4:
            self.qa_pairs.extend([
                {"question": "Is it incorrect to say the scene has fairly many greenery with fairly many buildings?", "answer": "No", "qtype": "advanced", "subtype": "negation.composite_conjunction_1"},
                {"question": "Would it be wrong to describe the area as both green and built-up?", "answer": "No", "qtype": "advanced", "subtype": "negation.composite_conjunction_1"},
                {"question": "Is it false to claim that there is a lot of vegetation and many buildings here?", "answer": "No", "qtype": "advanced", "subtype": "negation.composite_conjunction_1"}
            ])

        if svf >= 0.5 and gvi <= 0.3:
            self.qa_pairs.extend([
                {"question": "Is it incorrect to say the scene has open sky but also with sparse greenery?", "answer": "No", "qtype": "advanced", "subtype": "negation.composite_conjunction_2"},
                {"question": "Would it be inaccurate to describe the scene as open-sky but lacking in greenery?", "answer": "No", "qtype": "advanced", "subtype": "negation.composite_conjunction_2"},
                {"question": "Is it wrong to claim that there's plenty of visible sky but not much greenery?", "answer": "No", "qtype": "advanced", "subtype": "negation.composite_conjunction_2"}
            ])

        if bvf <= 0.3 and svf <= 0.3:
            self.qa_pairs.extend([
                {"question": "Is it incorrect to say the scene has a few buildings yet the sky is still not too visible?", "answer": "No", "qtype": "advanced", "subtype": "negation.composite_conjunction_3"},
                {"question": "Would it be wrong to describe the area as low-rise and enclosed by the sky?", "answer": "No", "qtype": "advanced", "subtype": "negation.composite_conjunction_3"},
                {"question": "Is it false to say that neither buildings nor sky dominate this scene?", "answer": "No", "qtype": "advanced", "subtype": "negation.composite_conjunction_3"}
            ])

    def generate_llm_enhanced_paraphrases(self, max_paraphrases=15):
        import google.generativeai as genai

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        count = 0

        candidates = [
            qa for qa in self.qa_pairs
            if qa["qtype"] == "advanced" and (
                "counterfactual" in qa["subtype"] or
                "negation" in qa["subtype"]
            )
        ]
        random.shuffle(candidates)

        for qa in candidates:
            if count >= max_paraphrases or self.llm_usage_count >= self.llm_max_calls:
                break

            prompt = f"""You are helping refine a dataset of urban scene question-answer pairs for a visual question answering (VQA) task.

                        This question involves **negation** or a **counterfactual 'what if'** scenario — these are especially important for linguistic diversity and reasoning.

                        Please **paraphrase** the following question so that it:

                        - Sounds more natural and fluent
                        - Preserves the original meaning and answer
                        - Stays grounded in the visible content of the image
                        - Uses diverse syntax or style (e.g., passive voice, rhetorical form, clause rearrangement)
                        - Maintains the counterfactual or negation intent, but phrases it differently

                        Do **not** introduce new information that is not inferable from the image.

                        ---
                        Original Question: {qa['question']}
                        Answer: {qa['answer']}
                        ---
                        Provide ONE paraphrased version of the question only. Do not explain or justify."""

            try:
                response = model.generate_content(prompt)
                output = response.text.strip()
                if output:
                    self.qa_pairs.append({
                        "question": output,
                        "answer": qa["answer"],
                        "qtype": "llm",
                        "subtype": "paraphrase.from_advanced",
                        "origin_subtype": qa["subtype"]
                    })
                    count += 1
                    self.llm_usage_count += 1
            except Exception as e:
                print("Gemini error:", e)


# --- Multiprocessing logic ---
def task(args):
    idx, data, api_key, budget = args
    _, features = data[idx]
    image_id = data.data[idx]["image_id"]
    generator = MultimodalQAGenerator(features, gemini_api_key=api_key, llm_budget_usd=budget)
    qa_pairs = generator.generate_all()
    return [{"image_id": image_id, **qa} for qa in qa_pairs]

def build_qa_dataset_parallel(consolidated_path, output_path, gemini_api_key=None, llm_budget_usd=10.0, load_depth=True):
    dataset = UnifiedUrbanDataset(consolidated_path, load_depth=load_depth)
    args_list = [(idx, dataset, gemini_api_key, llm_budget_usd) for idx in range(len(dataset))]

    print(f"🔄 Generating QA pairs for {len(dataset)} images using {multiprocessing.cpu_count()} workers...")

    MAX_QA_PAIRS = 10_000_000
    total_written = 0

    with open(output_path, 'w') as fout:
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for result in tqdm(executor.map(task, args_list), total=len(dataset)):
                if total_written >= MAX_QA_PAIRS:
                    print(f"🛑 Reached limit of {MAX_QA_PAIRS} QA pairs. Stopping early.")
                    break
                for qa in result:
                    if total_written >= MAX_QA_PAIRS:
                        break
                    fout.write(json.dumps(qa) + '\n')
                    fout.flush()
                    total_written += 1

    print(f"✅ Saved QA pairs to {output_path}")

# --- Main Entry ---
if __name__ == "__main__":
    build_qa_dataset_parallel(
        consolidated_path="./data/final_dataset.json",
        output_path="./qa_dataset_3.0.0_gemini_flash.jsonl",
        gemini_api_key="MY_API_KEY",
        llm_budget_usd=290.0
    )
