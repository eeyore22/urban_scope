import torch.distributed as dist
import os 
import torch 
import json 

# === DDP Helpers ===
def setup_ddp(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
        
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

# === Data Loading ===
# def read_jsonl_safe(path):
#     from safe_json_reader import read_jsonl_safe as _reader
#     return _reader(path)

def read_jsonl_safe(path):
    """
    Read a JSONL file where each line is (ideally) a standalone JSON object.
    Lines that fail to parse will be skipped with a warning.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed line and warn once
                if i == 1 or i % 1000 == 0:
                    print(f"⚠️ helpers.read_jsonl_safe: failed to parse JSON on line {i!r} of {path}")
                continue
    return data

def normalise(lst):
    """
    Cast everything to string so sklearn never sees a type-mix.
    Also trims whitespace and lowers case.
    """
    return [str(x).strip().lower() for x in lst]
