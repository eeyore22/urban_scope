import os
import json
import random
from tqdm import tqdm
import multiprocessing
import google.generativeai as genai
import openai

# Set your Gemini API key
genai.configure(api_key="MY_API_KEY")

# Load QA and Metadata
QA_PATH = "./eval/qa_dataset_4.0.0_train.json"
METADATA_PATH = "./final_dataset_2.0.0.json"
OUTPUT_PATH = "./eval/qa_dataset_4.1.0_train_cot.jsonl"
BATCH_SIZE = 10000

with open(QA_PATH, "r") as f:
    qa_dataset = [json.loads(line) for line in f]

with open(METADATA_PATH, "r") as f:
    metadata_entries = json.load(f)
    metadata_dict = {entry["image_id"].replace(".jpg", ""): entry for entry in metadata_entries}

# Subtype to rule description
subtype_to_rule = {
    "presence.binary": "If the object count is greater than 0, the answer is 'Yes'. Otherwise, 'No'.",
    "count": "The answer is the total count of the specified object.",
    "viewfactor.dominance": "If the view factor proportion is greater than 0.5, the answer is 'Yes'. Otherwise, 'No'.",
    "viewfactor.sparsity": "If the view factor proportion is less than or equal to 0.2, the answer is 'Yes'. Otherwise, 'No'.",
    "viewfactor.scalar": "The answer is the numerical proportion of the specified view factor, rounded to two decimal places.",
    "depth.range": "If the depth range is greater than 20, the answer is 'Complex'. Otherwise, 'Simple'.",
    "depth.variation": "If the depth range is greater than 20, the answer is 'Yes'. Otherwise, 'No'.",
    "depth.range.label": "If the depth range is greater than 40, label is 'high'. If greater than 20, label is 'moderate'. Otherwise, 'low'.",
    "depth.average_per_object": "The answer is the average depth of the specified object, rounded to two decimal places.",
    "depth.closest_object": "The answer is the object listed as closest in the image.",
    "object.cooccurrence": "If both 'car' and 'person' have a count greater than 0, the answer is 'Yes'. Otherwise, 'No'.",
    "occlusion.binary": "If the occlusion flag for the object pair is true, the answer is 'Yes'. Otherwise, 'No'.",
    "spatial.distribution": "If the spatial distribution for the object is 'left side', the answer is 'Yes'. Otherwise, 'No'.",
    "spatial.distribution.label": "The answer is the spatial distribution label (left side, right side, or evenly spread).",
    "depth.foreground_presence": "If the object is present in the foreground, the answer is 'Yes'. Otherwise, 'No'.",
    "depth.background_presence": "If the object is present in the background, the answer is 'Yes'. Otherwise, 'No'.",
    "scene.skyline_visibility": "If 'skyline_clear' is true, the answer is 'Yes'. Otherwise, 'No'.",
    "segmentation.vertical_layout": "If 'top_sky' is true, the answer is 'Yes'. Otherwise, 'No'.",
    "segmentation.vertical_layout.entity": "The answer is the 'top_entity' visible in the image.",
    "negation.absence": "If the object count is 0, the answer is 'Yes'. Otherwise, 'No'.",
    "negation.conjunction": "If the greenery or sky view factor is less than 0.2, the answer is 'Yes'.",
    "negation.exclusion_choice": "The answer is the object that is missing among the listed options.",
    "counterfactual.count_perturbation": "If the number of people plus two is greater than or equal to five, the answer is 'Yes'. Otherwise, 'No'.",
    "counterfactual.absence_viewfactor": "If the building view factor is greater than 0.3, the answer is 'Yes'. Otherwise, 'No'.",
    "counterfactual.attribute_substitution": "If the sky view factor is greater than 0.4, the answer is 'Yes'. Otherwise, 'No'.",
    "counterfactual.occlusion_movement": "If both 'bus' and 'pedestrian' are present in the depth order, the answer is 'Yes'.",
    "multi_hop.count_compare": "Compare the number of people and cars. If the number of people is greater, the answer is 'Yes'. Otherwise, 'No'.",
    "multi_hop.which_is_more": "Compare the number of people and cars. Answer which one is greater.",
    "spatial.relation": "Compare the depth order of the specified objects. If the first object is in front, answer 'Yes'. Otherwise, 'No'."
}

def build_cot_prompt(metadata, question, answer, subtype):
    rule = subtype_to_rule.get(subtype, "Use the image information to reason step-by-step and reach the answer.")
    prompt = f"""You are an assistant that generates chain-of-thought (CoT) answers for visual question answering tasks.

Given:
- Metadata: {json.dumps(metadata, indent=2)}
- Question: {question}
- Answer: {answer}

Your task:
- Derive the answer based only on the provided metadata, but when answering, you must always pretend that the information is directly seen from the image.
- You must not mention metadata or structured data. Always phrase the answer as if you are directly observing the image.
- You must use the rule that strictly corresponds to the question's subtype: {rule}
- You must not refer to or apply any other rule from other subtypes.
- Start with a step-by-step reasoning using the image content and the rule.
- Conclude with the final answer explicitly as 'Answer: <final answer>'.

Requirements:
- Do not invent additional details.
- Do not mention metadata, rule, or dataset. Only refer to visual observations from the image.
- Only apply the rule for the given subtype.

Please generate the chain-of-thought answer."""
    return prompt

# Worker function
import time

def process_single_qa(qa, queue):
    image_id_key = qa["image_id"].replace(".jpg", "")
    metadata = metadata_dict.get(image_id_key)

    if metadata is None:
        queue.put(("error", f"⚠️ Metadata not found for image {qa['image_id']}"))
        return

    question = qa["question"]
    answer = qa["answer"]
    subtype = qa.get("subtype", "")

    prompt = build_cot_prompt(metadata, question, answer, subtype)

    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt)
        cot_answer = response.text.strip()

        result = {
            "image_id": qa["image_id"],
            "question": question,
            "answer": answer,
            "subtype": subtype,
            "cot_answer": cot_answer
        }
        queue.put(("success", result))

        time.sleep(0.03)  # Throttle to ~33 requests per second

    except Exception as e:
        queue.put(("error", f"⚠️ Gemini error on image {qa['image_id']}: {e}"))
        time.sleep(2)  # Backoff on error

def load_existing_keys(output_path):
    existing_keys = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                item = json.loads(line)
                key = (item["image_id"], item["question"], item["answer"], item["subtype"])
                existing_keys.add(key)
    return existing_keys

def writer(queue, output_path):
    existing_keys = load_existing_keys(output_path)

    with open(output_path, "a") as f:
        while True:
            msg = queue.get()
            if msg == "DONE":
                break

            msg_type, content = msg

            if msg_type == "success":
                key = (content["image_id"], content["question"], content["answer"], content["subtype"])
                if key in existing_keys:
                    print(f"⚠️ Skipping duplicate: Image ID: {content['image_id']}, Subtype: {content['subtype']}, Question: {content['question']}, Answer: {content['answer']}")
                    continue
                json.dump(content, f)
                f.write("\n")
                existing_keys.add(key)

            elif msg_type == "error":
                print(content)


def run_batch(batch, batch_idx):
    print(f"🚀 Starting Batch {batch_idx + 1} with {len(batch)} QA pairs")

    queue = multiprocessing.Manager().Queue()

    writer_process = multiprocessing.Process(target=writer, args=(queue, OUTPUT_PATH))
    writer_process.start()

    with multiprocessing.Pool(processes=10) as pool:
        for qa in tqdm(batch, desc=f"Processing Batch {batch_idx + 1}"):
            pool.apply_async(process_single_qa, args=(qa, queue))

        pool.close()
        pool.join()

    # Signal writer to stop
    queue.put("DONE")
    writer_process.join()

    print(f"✅ Completed Batch {batch_idx + 1}")


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    batches = list(chunk_list(qa_dataset, BATCH_SIZE))

    for batch_idx, batch in enumerate(batches):
        run_batch(batch, batch_idx)

    print(f"🎉 All batches completed. CoT results saved to {OUTPUT_PATH}")

