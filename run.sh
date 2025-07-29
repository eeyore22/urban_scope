#!/usr/bin/env bash
set -euo pipefail

function usage(){
  cat <<EOF
Usage: $0 --framework <blip2|instructblip|llava> --mode <finetune|zero_shot> \\
           --gpus <cuda_ids> [--ngpus N] \\
           --test_json PATH [--train_json PATH --val_json PATH] \\
           --image_dir DIR [--out_dir DIR] [--cot] [--lora] [--fp16] [--bf16] [--ckpt] [--fsdp] \\
           [--bucket N] [--epochs N] [--batch N] [--lr LR] [--grad_accum N] [--seed N]
EOF
  exit 1
}

# defaults
NGPUS=1
OUT_DIR=""
COT_FLAG=""
LORA_FLAG=""
FP16_FLAG=""
BF16_FLAG=""
CKPT_FLAG=""
FSDP_FLAG=""
BUCKET=40
EPOCHS=20
BATCH=16
LR=1e-4
GRAD_ACCUM=1
SEED=42

# parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --framework)    FRAMEWORK="$2"; shift 2;;
    --mode)         MODE="$2";      shift 2;;
    --gpus)         GPUS="$2";      shift 2;;
    --ngpus)        NGPUS="$2";     shift 2;;
    --train_json)   TRAIN_JSON="$2";shift 2;;
    --val_json)     VAL_JSON="$2";  shift 2;;
    --test_json)    TEST_JSON="$2"; shift 2;;
    --image_dir)    IMAGE_DIR="$2"; shift 2;;
    --out_dir)      OUT_DIR="$2";   shift 2;;
    --cot)          COT_FLAG="--cot";      shift;;
    --lora)         LORA_FLAG="--lora";    shift;;
    --fp16)         FP16_FLAG="--fp16";    shift;;
    --bf16)         BF16_FLAG="--bf16";    shift;;
    --ckpt)         CKPT_FLAG="--ckpt";    shift;;
    --fsdp)         FSDP_FLAG="--fsdp";    shift;;
    --bucket)       BUCKET="$2";    shift 2;;
    --epochs)       EPOCHS="$2";    shift 2;;
    --batch)        BATCH="$2";     shift 2;;
    --lr)           LR="$2";        shift 2;;
    --grad_accum)   GRAD_ACCUM="$2";shift 2;;
    --seed)         SEED="$2";      shift 2;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

# required checks
[[ -z "${FRAMEWORK:-}" || -z "${MODE:-}" || -z "${GPUS:-}" || -z "${TEST_JSON:-}" ]] && usage

# default out_dir if none provided
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="outputs/${FRAMEWORK}/${MODE}${COT_FLAG:+_cot}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES="$GPUS"

# build common args string
COMMON_ARGS="--test_json $TEST_JSON \
             --image_dir $IMAGE_DIR \
             --out_dir $OUT_DIR \
             --batch $BATCH \
             --lr $LR \
             --seed $SEED \
             $COT_FLAG $LORA_FLAG $FP16_FLAG $BF16_FLAG $CKPT_FLAG $FSDP_FLAG \
             --bucket $BUCKET \
             --epochs $EPOCHS \
             --grad_accum $GRAD_ACCUM"

if [[ "$MODE" == "finetune" ]]; then
  # finetune => use torchrun and require train/val json
  if [[ -z "${TRAIN_JSON:-}" || -z "${VAL_JSON:-}" ]]; then
    echo "For finetune you must supply --train_json and --val_json"
    exit 1
  fi
  CMD="torchrun --nproc_per_node=$NGPUS eval/trainers/${FRAMEWORK}.py \
        --train_json $TRAIN_JSON --val_json $VAL_JSON \
        $COMMON_ARGS"
#else
  # zero-shot => single-GPU python
  #CMD="python trainers/${FRAMEWORK}.py $COMMON_ARGS"
#fi

else
  # zero-shot via torchrun too
  CMD="torchrun --nproc_per_node=$NGPUS eval/trainers/${FRAMEWORK}.py \
        $COMMON_ARGS"
fi

echo ">>> $CMD"
eval $CMD



# For fine-tuning, run with: (ex: BLIP2 with CoT on 4 GPUs)
# ./run.sh \
#   --framework blip2 \
#   --mode finetune \
#   --gpus 4,5,6,7 \
#   --ngpus 4 \
#   --train_json qa_dataset_5.0.0_train.json \
#   --val_json   qa_dataset_5.0.0_val.json \
#   --test_json  qa_dataset_5.0.0_test.json \
#   --image_dir  images_v3_50k \
#   --out_dir    outputs/blip2/finetune_cot \
#   --cot \
#   --lora --bf16 --ckpt \
#   --bucket 40 \
#   --epochs 2 \
#   --batch 16 \
#   --lr 1e-4 \
#   --grad_accum 1 \
#   --seed 42


# For zero-shot evaluation, run with (ex: InstructBLIP on 2 GPUs):
# ./run.sh \
#   --framework instructblip \
#   --mode zero_shot \
#   --gpus 6,7 \
#   --ngpus 2 \
#   --test_json eval/qa_dataset_4.0.0_test.json \
#   --image_dir eval/images_v3_50k \
#   --out_dir   outputs/instructblip/zero_shot \
#   --seed 42

# For either options, make it executable beforehand
# chmod +x run.sh