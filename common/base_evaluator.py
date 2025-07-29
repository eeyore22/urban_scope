# common/base_evaluator.py

import os, json, torch, torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import re
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, classification_report
from process import *
from helpers import *

# --- Centralize CoT settings
COT_PROMPT_PREFIX = "Please explain your reasoning step by step. Answer: "
COT_GEN_KWARGS = dict(
    max_new_tokens=128,
    do_sample=False,
    num_beams=3,
    repetition_penalty=1.2,
    length_penalty=1.0,
    early_stopping=True,
    pad_token_id=None, # fill later
)
DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=20,
    do_sample=False,
    num_beams=1,
    early_stopping=True,
    pad_token_id=None, # fill later
)

# ---Dataset---
class VQADataset(Dataset):
    def __init__(self, qa_list, image_root):
        self.qa_list = qa_list 
        self.image_root = image_root 
    def __len__(self):
            return len(self.qa_list)
    def __getitem__(self, idx):
        return self.qa_list[idx]

# ---Distributed Sampler---
class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset 
        self.num_replicas = num_replicas or get_world_size()
        self.rank = rank if rank is not None else get_rank()
        self.total_size = len(self.dataset)
        self.indices = list(range(self.total_size))[self.rank::self.num_replicas]
        self.num_samples = len(self.indices)
    def __iter__(self):
        assert len(self.indices) == self.num_samples
        return iter(self.indices)
    def __len__(self):
        return self.num_samples 
    
# ---Evaluator---
class VQAEvaluator:
    def __init__(self):
        self.results = defaultdict(list)
        self.labels = defaultdict(list)
        self.by_subtype = defaultdict(lambda: {"preds": [], "labels": []}) 
    
    def add(self, group, pred, label, subtype=None):
        self.results[group].append(pred)
        self.labels[group].append(label)
        if subtype is not None:
            self.by_subtype[subtype]["preds"].append(pred)
            self.by_subtype[subtype]["labels"].append(label)
    
    def compute(self, verbose=False):
        for group, preds in self.results.items():
            gts = self.labels[group]
            if not preds:
                continue 
            
            # Normalize type before metrics 
            # try:
            #     if all(isinstance(x, (int, float)) for x in gts+preds):
            #         gts = [float(x) for x in gts]
            #         preds = [float(x) for x in preds]
            #     else:
            #         gts = [str(x) for x in gts]
            #         preds = [str(x) for x in preds]
            # except Exception as e:
            #     print(f"Skipping group {group} due to error: {e}")
            #     continue 
            

            print(f"\n[{group}] ({len(preds)} samples)")
            if "scalar" in group or "numeric" in group:
                mae = mean_absolute_error(gts, preds)
                print(f"   MAE={mae:.4f}")
            
            elif "binary" in group:
                acc = accuracy_score(gts, preds)
                f1 = f1_score(gts, preds)
                
                # #dynamically determine the pos_label 
                # label_set = set(gts+preds)
                # if all(isinstance(x, str) for x in label_set):
                #     f1 = f1_score(gts, preds, average="binary", pos_label="yes")
                # elif all(isinstance(x, (int, float)) for x in label_set):
                #     # assume positive class is 1.0 for numeric labels 
                #     f1 = f1_score(gts, preds, average="binary", pos_label=1.0)
                # else:
                #     # fall back to macro average if mixed types
                #     f1 = f1_score(gts, preds, average="macro")
                
                print(f"   Acc={acc:.4f} | F1={f1:.4f}")
                
            else: # multi-class
                print(classification_report(gts, preds, zero_division=0, output_dict=False))
                
        print("\n--- Per Subtype Results ---")
        for subtype, data in self.by_subtype.items():
            preds, gts = data["preds"], data["labels"]
            if not preds:
                continue 
            
            # try:
            #     if all(isinstance(x, (int, float)) for x in gts + preds):
            #         gts = [float(x) for x in gts]
            #         preds = [float(x) for x in preds]
            #     else:
            #         gts = [str(x) for x in gts]
            #         preds = [str(x) for x in preds]
            # except Exception as e:
            #     print(f"⚠️ Skipping subtype {subtype} due to error: {e}")
            #     continue
            
            print(f"\n[{subtype}] ({len(preds)} samples)")
            if any(isinstance(x, float) for x in preds):
                try:
                    mae = mean_absolute_error(gts, preds)
                    print(f"   MAE={mae:.4f}")
                except:
                    print(f"   Skipped non-numeric values")
            #elif set(gts).issubset({0.0, 1.0, "0", "1"}):
            elif set(gts).issubset({0.0, 1.0}):
                acc = accuracy_score(gts, preds)
                f1 = f1_score(gts, preds)
                print(f"   Acc={acc:.4f} | F1={f1:.4f}")
            else:
                print(classification_report(gts, preds, zero_division=0, output_dict=False))
                
def prepare_batch_data(qa_list, image_root: str, use_cot: bool):
    batch_images, batch_prompts, batch_meta = [], [], []
    total_items = len(qa_list)
    missing_images, missing_keys, invalid_eval_group, successful_items = 0, 0, 0, 0

    answer_key = "cot_answer" if use_cot else "answer"
    for qa in qa_list:
        img_path = os.path.join(image_root, qa["image_id"])
        if not os.path.exists(img_path):
            missing_images += 1
            continue

        subtype = qa.get("subtype", "")
        # strip off paraphrase prefix
        if subtype.startswith("paraphrase.from_"):
            subtype = subtype.replace("paraphrase.from_", "")

        prompt_template = PROMPT_MAP.get(subtype, qa.get("question", ""))
        if prompt_template and "{" in prompt_template:
            missing_keys = [k for k in re.findall(r"{(.*?)}", prompt_template) if k not in qa]
            if missing_keys:
                print(f"Skipping {qa['image_id']}: missing keys {missing_keys}")
                continue
            try:
                prompt = prompt_template.format(**qa)
            except Exception as e:
                print(f"Skipping {qa['image_id']}: format error {e}")
                continue
        else:
            prompt = prompt_template
                
        eval_group = get_eval_group(subtype)
        if eval_group is None:
            invalid_eval_group += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            batch_images.append(image)
            batch_prompts.append(prompt)
            batch_meta.append({
                "qa":         qa,
                "answer":     qa.get(answer_key, qa["answer"]),
                "eval_group": eval_group,
                "subtype":    subtype,
            })
            successful_items += 1
        except Exception as e:
            print(f"Error loading image {qa['image_id']}: {e}")
            continue

    # Debug logging 
    if total_items > 0:
        print(f"Batch processing stats: {total_items} total, {successful_items} successful, "
              f"{missing_images} missing images, {invalid_eval_group} invalid eval_groups")
        
    return batch_images, batch_prompts, batch_meta


def process_single_item(model, processor, device, image, prompt: str, qa_info: dict, adapter, use_cot: bool=False):
    """Run one example through model.generate, decode, parse, return a result dict."""

    try:
        raw_inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )

        processed = adapter.post_process_inputs(raw_inputs) # DefaultAdapter는 no-op, LlavaAdapter는 pixel_values.half()
        inputs = {k: v.to(device) for k, v in processed.items()}
        
        gen_kwargs = COT_GEN_KWARGS if use_cot else DEFAULT_GEN_KWARGS
        gen_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id
        
        with torch.no_grad():
            outputs = (model.module if isinstance(model, DDP) else model).generate(
                **inputs,
                **gen_kwargs
            )
            
        raw_pred = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        parsed_pred = parse_answer(raw_pred, qa_info["eval_group"], qa_info["subtype"])
        parsed_gt = parse_answer(qa_info["answer"], qa_info["eval_group"], qa_info["subtype"])
        if parsed_pred is None or parsed_gt is None:
            print(f"[DEBUG] Failed to parse prediction or label for {qa_info['qa']['image_id']}: pred='{raw_pred}', GT='{qa_info['answer']}', group='{qa_info['eval_group']}'")
            return None 
        
        return {
            "image_id": qa_info["qa"]["image_id"],
            "question": prompt,
            "gt_answer": qa_info["answer"],
            "pred_answer": raw_pred,
            "parsed_pred": parsed_pred,
            "parsed_gt": parsed_gt,
            "eval_group": qa_info["eval_group"],
            "subtype": qa_info["subtype"] 
        }
    except Exception as e:
        print(f"[DEBUG] Error processing single item {qa_info['qa']['image_id']}: {e}")
        return None 
        
        
def process_batch(model, processor, device, batch_images, batch_prompts, batch_meta, adapter, use_cot: bool=False, verbose: bool=False):
    """Run a batch at once via model.generate, decode, parse all."""

    # 1) raw processor call (returns a dict of tensors)
    raw_inputs = processor(
        images=batch_images,
        text=batch_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    
    # 2) model-specific post-processing
    processed = adapter.post_process_inputs(raw_inputs)
    inputs = {k: v.to(device) for k, v in processed.items()}
    
    gen_kwargs = COT_GEN_KWARGS if use_cot else DEFAULT_GEN_KWARGS
    gen_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id
    
    with torch.no_grad():
        outs = (model.module if isinstance(model, DDP) else model).generate(
            **inputs,
            **gen_kwargs
        )
        
    raw_pred = processor.tokenizer.batch_decode(outs, skip_special_tokens=True)
    results = []
    parsing_failures = 0
    
    for i, (raw_pred, qa_info) in enumerate(zip(raw_pred, batch_meta)):
        raw_pred = raw_pred.strip().lower()
        qa = qa_info["qa"]
        answer = qa_info["answer"]
        eval_group = qa_info["eval_group"]
        subtype = qa_info["subtype"]
        
        parsed_pred = parse_answer(raw_pred, eval_group, subtype)
        parsed_gt = parse_answer(answer, eval_group, subtype)

        if parsed_pred is None or parsed_gt is None:
            parsing_failures += 1 
            print(f"Failed to parse batch item {qa['image_id']}: pred='{raw_pred}', label='{answer}', group='{eval_group}'")
            continue 
        
        results.append({
            "image_id": qa["image_id"],
            "question": batch_prompts[i],
            "gt_answer": answer,
            "pred_answer": raw_pred,
            "parsed_pred": parsed_pred,
            "parsed_gt": parsed_gt,
            "eval_group": eval_group,
            "subtype": subtype
        })
        
        if verbose:
            print(f"\n Image: {qa['image_id']} | Subtype: {subtype}")
            print(f"🟩 GT: {answer} | 🟥 Prediction: {raw_pred}")
            print(f"✅ Parsed GT: {parsed_gt} | ✅ Parsed Pred: {parsed_pred}")
            verdict = "✔️  Correct" if is_correct(parsed_pred, parsed_gt, subtype) else "❌ Incorrect"
            print(f"Verdict: {verdict}")
            
    if parsing_failures > 0:
        print(f"Batch processing: {len(batch_meta)} total, {len(results)} successful, {parsing_failures} parsing failures")
        
    return results


def evaluate_model(
    model_dir: str,
    test_json: str,
    image_dir: str,
    adapter,
    processor_fn,               # e.g. lambda d: Blip2Processor.from_pretrained(d)
    model_fn,                   # e.g. lambda d: Blip2ForConditionalGeneration.from_pretrained(d)
    log_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    mode: str = "zero_shot",
    use_cot: bool = False
):
    # DDP setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1 and not dist.is_initialized():
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda:0")
        
    # Load data 
    with open(test_json, 'r') as f:
        qa_data = [json.loads(l) for l in f]

    dataset = VQADataset(qa_data, image_dir)
    sampler = DistributedEvalSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x, num_workers=num_workers, pin_memory=True)
    
    # Load model & processor 
    processor = processor_fn(model_dir)
    model = model_fn(model_dir).to(device)
    model.eval()
    model = model.half()
    # if world_size == 1:
    #     try: model = torch.compile(model, mode="reduce-overhead")
    #     except: pass 
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except:
        pass
    if world_size > 1 and any(p.requires_grad for p in model.parameters()):
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
        
    evaluator = VQAEvaluator()
    all_preds = []
    
    if is_main_process():
        print(f"Starting evaluation on {len(qa_data)} examples across {world_size} GPU(s)")
    pbar = tqdm(loader, disable=not is_main_process(), desc=f"Evaluating (rank {get_rank()})")
    
    for batch in pbar:
        images, raw_prompts, meta = prepare_batch_data(batch, image_dir, use_cot)
        if not images:
            continue 
        
        # Format each prompt with the adapter 
        prompts = [
            adapter.format_prompt(raw, m["qa"], use_cot)
            for raw, m in zip(raw_prompts, meta)
        ]
        
        results = process_batch(model, processor, device, images, prompts, meta, adapter, use_cot, verbose=False)
        if results is None:
            results = []
            for img, prompt, info in zip(images, prompts, meta):
                r = process_single_item(model, processor, device, img, prompt, info, adapter, use_cot)
                if r:
                    results.append(r)
        for r in results:
            evaluator.add(r["eval_group"], r["parsed_pred"], r["parsed_gt"], subtype=r["subtype"])
        all_preds.extend(results)
        
    # Write out predictions 
    os.makedirs(log_dir, exist_ok=True)
    pred_path = os.path.join(log_dir, f"preds_rank{get_rank()}.jsonl")
    with open(pred_path, "w") as f:
        for entry in all_preds:
            f.write(json.dumps(entry) + "\n")
            
    # Only rank 0 writes summary (aggregating all ranks)
    if is_main_process():
        import glob, sys, io
        # 1) gather all per-rank prediction files 
        pattern = os.path.join(log_dir, "preds_rank*.jsonl")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f" No prediction files found with pattern {pattern}")
            return 
        
        # 2) load them all into one big list 
        all_entries = []
        for fpath in files:
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    all_entries.append(json.loads(line))
        
        # 3) save the merged JSONL for record 
        agg_jsonl = os.path.join(log_dir, f"{mode}_aggregated_predictions.jsonl")
        with open(agg_jsonl, "w") as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved aggregated predictions to {agg_jsonl}")
        
        # 4) re-compute metrics on the full set 
        agg_eval = VQAEvaluator()
        for entry in all_entries:
            agg_eval.add(
                entry["eval_group"],
                entry["parsed_pred"],
                entry["parsed_gt"],
                subtype=entry.get("subtype")
            )
            
        # 5) write the aggregated report
        print(f"\n=== Aggregated Evaluation Results for {mode} mode ===")
        print(f"Total samples across all ranks: {len(all_entries)}")
        report_path = os.path.join(log_dir, f"{mode}_aggregated_report.txt")
        with open(report_path, "w") as fw:
            old_stdout = sys.stdout 
            sys.stdout = io.TextIOWrapper(fw.buffer, encoding="utf-8")
            agg_eval.compute(verbose=False)
            sys.stdout.flush()
            sys.stdout = old_stdout
        print(f"Aggregated evaluation report saved to {report_path}")
        
    if world_size > 1:
        dist.barrier()
        cleanup_ddp()
        
    
# === Prompt Rewriting Map ===
PROMPT_MAP = {
    # Object Presence & Counts
    "presence.binary":
        "Is there a {object} in the image? Respond with exactly \"yes\" or \"no.\"",
    "count":
        "How many {object} instances are visible in the image? Respond with an integer.",

    # View Factor Analysis
    "viewfactor.dominance":
        "Is the scene dominated by {factor}? Respond with exactly \"yes\" or \"no.\"",
    "viewfactor.sparsity":
        "Does the scene have sparse {factor}? Respond with exactly \"yes\" or \"no.\"",
    "viewfactor.scalar":
        "What is the view factor for {factor}? Respond with a decimal number between 0 and 1.",

    # Depth Complexity
    "depth.range":
        "Does the scene appear visually complex? Respond with \"low,\" \"moderate,\" or \"high.\"",
    "depth.variation":
        "Is there a large depth variation in the scene? Respond with exactly \"yes\" or \"no.\"",
    "depth.range.label":
        "Label the overall depth complexity of this scene: low, moderate, or high.",
    "depth.average_per_object":
        "What is the average depth of visible buildings? Provide a numeric estimate.",
    "depth.closest_object":
        "Which object is closest to the camera? Respond with the object name.",
    "depth.foreground_presence":
        "Is there a {object} in the foreground of the scene? Respond with \"yes\" or \"no.\"",
    "depth.background_presence":
        "Are any {object}s located far in the background? Respond with \"yes\" or \"no.\"",

    # Segmentation-Based Layout
    "spatial.distribution":
        "Are {object}s primarily on the left half of the image? Respond with \"yes\" or \"no.\"",
    "spatial.distribution.label":
        "Where are the {object}s mostly located in the image? Choose one: left, right, or evenly spread.",
    "segmentation.vertical_layout":
        "Is the top portion of the image mostly sky? Respond with \"yes\" or \"no.\"",
    "segmentation.vertical_layout.entity":
        "Which object occupies the uppermost region of the image? Respond with the object name.",

    # Object Co-occurrence
    "object.cooccurrence":
        "Are both a {object1} and a {object2} present in the image? Respond \"yes\" or \"no.\"",
    "scene.skyline_visibility":
        "Is the city skyline clearly visible? Respond with \"yes\" or \"no.\"",
    "occlusion.binary":
        "Are the {object1}s occluded by a {object2}? Respond with \"yes\" or \"no.\"",

    # Advanced Reasoning
    "negation.absence":
        "Is there no {object} visible in the scene? Respond \"yes\" or \"no.\"",
    "negation.conjunction":
        "Is the statement \"the scene is green and open\" incorrect? Respond \"yes\" or \"no.\"",
    "negation.exclusion_choice":
        "Which of these is not present: a tree, a car, or a bench? Respond with one of: tree, car, bench.",
    "negation.composite_conjunction_1":
        "Is it false that the scene has both abundant vegetation and many buildings? Respond \"yes\" or \"no.\"",
    "negation.composite_conjunction_2":
        "Is it incorrect to claim the sky is open while greenery is sparse? Respond \"yes\" or \"no.\"",
    "negation.composite_conjunction_3":
        "Would it be wrong to describe the area as low-rise and enclosed by sky? Respond \"yes\" or \"no.\"",

    # Multi-Hop & Comparison
    "multi_hop.count_compare":
        "Are there more people than cars in the image? Respond \"yes\" or \"no.\"",
    "multi_hop.which_is_more":
        "Which is greater: the number of people or the number of cars? Respond \"people\" or \"cars.\"",
    "spatial.relation":
        "Is the tree positioned in front of the building? Respond \"yes\" or \"no.\"",

    # Counterfactuals
    "counterfactual.count_perturbation":
        "If two more people entered the scene, would it look crowded? Respond \"yes\" or \"no.\"",
    "counterfactual.absence_viewfactor":
        "Would the scene feel more natural if buildings were absent? Respond \"yes\" or \"no.\"",
    "counterfactual.attribute_substitution":
        "If the sky were overcast instead of clear, would the scene feel less open? Respond \"yes\" or \"no.\"",
    "counterfactual.occlusion_movement":
        "If the bus were moved to the front, would it block the view? Respond \"yes\" or \"no.\"",
}


def get_eval_group(subtype):
    if subtype == "viewfactor.scalar":
        return "view_factor_scalar"
    elif subtype in ["viewfactor.dominance", "viewfactor.sparsity"]:
        return "view_factor_binary"
    elif subtype == "presence.binary":
        return "presence_binary"
    elif subtype == "count":
        return "count_numeric"
    elif subtype in ["depth.range", "depth.variation"]:
        return "depth_binary"
    elif subtype == "depth.range.label":
        return "depth_categorical"
    elif subtype == "depth.average_per_object":
        return "depth_numeric"
    elif subtype == "depth.closest_object":
        return "depth_closest_object"
    elif subtype in ["depth.foreground_presence", "depth.background_presence"]:
        return "depth_binary"
    elif subtype == "spatial.distribution":
        return "layout_binary"
    elif subtype == "spatial.distribution.label":
        return "layout_text"
    elif subtype in ["segmentation.vertical_layout", "scene.skyline_visibility"]:
        return "layout_binary"
    elif subtype == "segmentation.vertical_layout.entity":
        return "layout_top_entity"
    elif subtype == "object.cooccurrence":
        return "cooccurrence_binary"
    elif subtype == "occlusion.binary":
        return "occlusion_binary"
    elif "negation" in subtype:
        return "advanced_negation"
    elif "multi_hop" in subtype:
        return "advanced_multihop"
    elif "counterfactual" in subtype:
        return "advanced_counterfactual"
    elif subtype == "spatial.relation":
        return "advanced_spatial"
    return None

