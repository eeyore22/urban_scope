# Pytorch implementation for evaluating VLM on a synthetic dataset
This repository covers pytorch implementation for evaluating and fine-tuning BLIP2, InstructBLIP and LLaVA-1.5 on a synthetic instruction dataset. 
This study was published at ICCV 2025 Workshop (Multimodal Reasoning and Slow Thinking in Large Model Era: Towards System 2 and Beyond).

## Preliminary
While VLMs are generally strong at recognizing what objects are present in an image, they often struggle to interpret where those objects are located and how they are spatially arranged.
Urban scenes contain repetitive elements such as sky, buildings, and greenery, but meaningful understanding depends on fine-grained spatial cues—relative proportions, occlusion (e.g., how much buildings block the sky), depth structure, and left–right layout.
These spatial relationships are crucial for urban planning [1], walkability assessment [2], and urban heat-island analysis [3], yet existing models rarely reason over such cues in a systematic way.
Therefore, we 1) constructed a synthetic dataset tailored to urban scenes and relevant question-answer pairs and 2) ran zero-shot evaluation and fine-tuning on representative open-source VLMs to test whether domain-specific instruction tuning using a synthetic dataset can help close the domain gap.

![Domain Gap](intro_figure.png)

## Dependency
<pre> pip install -r requirements.txt </pre>

## 🚀 Usage

The launcher script `run.sh` provides a unified interface for all open-source models.  
Simply change `--framework <blip2|instructblip|llava>` to switch between models.

---

### 🔵 Fine-tuning (example: BLIP-2)

```bash
./run.sh \
  --framework blip2 \
  --mode finetune \
  --gpus 0,1 \
  --ngpus 2 \
  --train_json path/to/train.json \
  --val_json   path/to/val.json \
  --test_json  path/to/test.json \
  --image_dir  path/to/images \
  --out_dir    outputs/blip2/finetune \
  [--cot] [--lora] [--fp16|--bf16] [--ckpt] \
  --epochs 20 \
  --batch 16 \
  --lr 1e-4 \
  --seed 42

./run.sh \
  --framework blip2 \
  --mode zero_shot \
  --gpus 0 \
  --ngpus 1 \
  --test_json path/to/test.json \
  --image_dir path/to/images \
  --out_dir   outputs/blip2/zero_shot \
  [--cot] [--fp16|--bf16] \
  --seed 42
