# trainers/blip2.py
import torch
import os 
import argparse
from transformers import BitsAndBytesConfig, Blip2ForConditionalGeneration, Blip2Processor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from common.base_trainer import BaseTrainer
from dataset.vqa_dataset import VQADataset
from dataset.vqa_dataset_cot import VQADatasetCoT
from common.adapters import DefaultAdapter
from common.base_evaluator import evaluate_model

class BLIP2Trainer(BaseTrainer):
    def build_processor(self):
        return Blip2Processor.from_pretrained(self.args.model)

    def build_model(self):
        # 1) LoRA + 8-bit path
        if self.args.lora:
            # a) load in 8-bit
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            model = Blip2ForConditionalGeneration.from_pretrained(
                self.args.model,
                quantization_config=bnb_cfg,
                device_map={"": self.rank}
            )
            # b) prepare for k-bit training
            model = prepare_model_for_kbit_training(model)
            # c) attach LoRA adapters
            lora_cfg = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
                bias="none",
                task_type="CAUSAL_LM",  # keep causal
            )
            model = get_peft_model(model, lora_cfg)
        # 2) Full-precision / mixed-precision path
        else:
            dtype = (
                torch.bfloat16 if self.args.bf16
                else torch.float16 if self.args.fp16
                else torch.float32
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                self.args.model,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(self.device)

        # 3) freeze vision if requested
        if self.args.freeze_vision:
            for name, param in model.named_parameters():
                if "vision_model" in name:
                    param.requires_grad_(False)

        # 4) gradient checkpointing
        if self.args.ckpt:
            model.gradient_checkpointing_enable()

        return model

    def build_dataset(self, json_path, img_dir):
        DatasetClass = VQADatasetCoT if self.args.cot else VQADataset
        return DatasetClass(json_path, img_dir, self.proc)

    def evaluate(self):
        adapter = DefaultAdapter()
        evaluate_model(
            model_dir=os.path.join(self.args.out_dir, "best"),
            test_json=self.args.test_json,
            image_dir=self.args.image_dir,
            adapter=adapter,
            processor_fn=lambda d: Blip2Processor.from_pretrained(d),
            model_fn=lambda d: Blip2ForConditionalGeneration.from_pretrained(d),
            log_dir=self.args.out_dir, 
            mode="finetuned",
            use_cot=self.args.cot 
        )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--out_dir", type=str, default="outputs/blip2")
    parser.add_argument("--train_json",type=str, required=False, default=None)
    parser.add_argument("--val_json", type=str, required=False, default=None)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="images_v3_50k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cot", action="store_true", help="Enable chain‑of‑thought training/eval")
    
    # Training
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    
    # Memory/precision
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--ckpt", action="store_true")
    parser.add_argument("--bucket", type=int, default=40, help="Bucket size (MB) for DDP gradient all-reduce")
    
    # Distributed 
    parser.add_argument("--fsdp", action="store_true", help="Enable fully-sharded training (skip with --lora)")
    
    # LoRA
    parser.add_argument("--lora", action="store_true", help="Enable LoRA adapters + 8-bit base weights")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.lora:
        args.fsdp = False 
    if args.train_json and args.val_json: 
        # Fine-tune mode
        trainer = BLIP2Trainer(args)
        trainer.train()
    else: 
        # Zero-shot evaluation mode
        adapter = DefaultAdapter()
        evaluate_model(
            model_dir=args.model,
            test_json=args.test_json,
            image_dir=args.image_dir,
            adapter=adapter,
            processor_fn=lambda d: Blip2Processor.from_pretrained(d),
            model_fn=lambda d: Blip2ForConditionalGeneration.from_pretrained(d),
            log_dir=args.out_dir,
            mode="zero_shot",
            use_cot=args.cot
        )
        
        
