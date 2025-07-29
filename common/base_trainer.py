# common/base_trainer.py
import os
import numpy as np
import random
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from utils.eta import compute_eta, format_eta
import abc

class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.rank, self.world = self._ddp_setup()
            
        # Device + output dir
        self.device = torch.device("cuda")
        os.makedirs(args.out_dir, exist_ok=True)
        
        if getattr(args, "seed", -1) >= 0:
            self._set_seed(args.seed, self.rank)
        #self.grad_accum = getattr(args, "grad_accum", 1)

        # these will be filled in by subclass
        self.proc     = self.build_processor()
        self.model    = self.build_model().to(self.device)
        self.train_ds = self.build_dataset(args.train_json, args.image_dir)
        self.val_ds   = self.build_dataset(args.val_json,   args.image_dir)

        self._wrap_fsdp_or_ddp()
        
        # Optimizer & scaler
        self.opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(getattr(args, "fp16", False) or getattr(args, "bf16", False))
        )

    def _ddp_setup(self):
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        if world > 1:
            dist.init_process_group("nccl")
            torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world
    
    def _cleanup(self):
        """Shut down the process group."""
        if dist.is_initialized(): 
            dist.barrier()
            dist.destroy_process_group()
    
    def _is_main(self):
        """Are we rank 0?"""
        return self.rank == 0

    def _wrap_fsdp_or_ddp(self):
        # FSDP branch
        if getattr(self.args, "fsdp", False) and not getattr(self.args, "lora", False):
            self.model = FSDP(
                self.model,
                auto_wrap_policy=size_based_auto_wrap_policy(1e7),
                mixed_precision=True,
                device_id=self.device
            )
        # DDP branch
        elif self.world > 1:
            self.model = DistributedDataParallel(
                self.model, 
                device_ids=[self.device.index], 
                bucket_cap_mb=self.args.bucket, 
                find_unused_parameters=self.find_unused_parameters()
            )

    def _set_seed(self, seed: int, rank: int = 0):
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cudnn.deterministic = True   # more reproducible, ~1-3% slower
        torch.backends.cudnn.benchmark     = False
    
    def train(self):
        # Prepare loaders
        tr_samp = DistributedSampler(self.train_ds, num_replicas=self.world, rank=self.rank, shuffle=True)
        va_samp = DistributedSampler(self.val_ds, num_replicas=self.world, rank=self.rank, shuffle=False)
        
        tr_ld = DataLoader(self.train_ds, 
                           batch_size=self.args.batch, 
                           sampler=tr_samp,
                           collate_fn=self.collate_fn(), 
                           num_workers=2, 
                           pin_memory=True,
                           worker_init_fn=self._worker_init_fn)
        va_ld = DataLoader(self.val_ds,   
                           batch_size=self.args.batch, 
                           sampler=va_samp,
                           collate_fn=self.collate_fn(), 
                           num_workers=2, 
                           pin_memory=True,
                           worker_init_fn=self._worker_init_fn)

        best_val, best_ep = float('inf'), -1
        epoch_times = [] # ETA tracking
        
        for ep in range(1, self.args.epochs+1):
            epoch_start = time.time()
            tr_samp.set_epoch(ep)
            
            self.model.train()
            total_loss = 0
            cnt = 0
            it_start = time.time()
            
            for i, batch in enumerate(tr_ld, 1):
                batch = {k:v.to(self.device, non_blocking=True) for k,v in batch.items()}
                
                # Forward/backward
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    loss = self.model(**batch).loss / self.args.grad_accum
                self.scaler.scale(loss).backward()
                cnt += 1
                total_loss += loss.item() * self.args.grad_accum
                
                if cnt % self.args.grad_accum == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad()
                    
                if self._is_main():
                    eta = compute_eta(time.time()-it_start, i, len(tr_ld))
                    print(f"Epoch {ep}/{self.args.epochs} | Iter {i}/{len(tr_ld)} | Loss {loss.item():.4f} | ETA {format_eta(eta)}")

            # Aggregate train loss across GPUs
            t = torch.tensor(total_loss, device=self.device)
            if self.world > 1:
                dist.all_reduce(t)
            train_loss = t.item() / self.world / len(tr_ld)
            
            # Validation
            val_loss = self._validate(va_ld)
            epoch_times.append(time.time()-epoch_start)
            
            # Epoch-level logging
            if self._is_main():
                avg_epoch = sum(epoch_times) / len(epoch_times)
                remaining = avg_epoch * (self.args.epochs-ep)
                print(f"Epoch {ep} done. Train {total_loss/len(tr_ld):.4f} | Val {val_loss:.4f} | Remain {format_eta(remaining)}")
                if val_loss < best_val:
                    best_val, best_ep = val_loss, ep
                    self._save_checkpoint("best")
        
        # Final checkpoint & eval
        if self._is_main():
            self._save_checkpoint("final")
            self.evaluate()
            
        self._cleanup()

    def _validate(self, loader):
        self.model.eval()
        total = 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            for batch in loader:
                batch = {k:v.to(self.device, non_blocking=True) for k,v in batch.items()}
                total += self.model(**batch).loss.item()
        
        # Aggregate across GPUs
        t = torch.tensor(total, device=self.device)
        if self.world > 1:
            dist.all_reduce(t)
        return t.item() / self.world / len(loader)

    # ——— Abstract methods. Must be implemented per model. —————————————————————————————————————————
    @abc.abstractmethod
    def build_processor(self):
        raise NotImplementedError

    @abc.abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def build_dataset(self, json_path, img_dir):
        raise NotImplementedError

    def collate_fn(self):
        """Return None or a custom collate function."""
        return None

    def find_unused_parameters(self):
        """DDP find_unused_parameters flag (some models need True)."""
        return False

    def _worker_init_fn(self, worker_id):
        if getattr(self.args, "seed", -1) >= 0:
            seed = self.args.seed + self.rank * 1000 + worker_id
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _save_checkpoint(self, name):
        ckpt_dir = os.path.join(self.args.out_dir, name)
        os.makedirs(ckpt_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(ckpt_dir)
        self.proc.save_pretrained(ckpt_dir)

    def evaluate(self):
        """Subclasses should override this to call into their evaluator."""
        raise NotImplementedError
