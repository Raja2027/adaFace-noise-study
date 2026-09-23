#!/usr/bin/env python3
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import gc
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from training.image_dataset import ImageManifestDataset
from net import build_model
from head import AdaFace

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def benchmark_batch_size(batch_size, device, dataset, max_steps=500):
    print(f"\n[{batch_size}] Testing Batch Size: {batch_size}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    loader_iter = iter(loader)
    
    # Re-initialize models and optimizers to avoid memory fragmentation and state carry-over
    backbone = build_model('ir_50').to(device)
    head = AdaFace(embedding_size=512, classnum=2000, m=0.4, h=0.333, s=64.0, t_alpha=0.01).to(device)
    
    # Parameter groups for BN weight decay = 0
    backbone_params = []
    backbone_bn_params = []
    for name, param in backbone.named_parameters():
        if len(param.shape) == 1 or name.endswith(".bias"):
            backbone_bn_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'weight_decay': 5e-4},
        {'params': backbone_bn_params, 'weight_decay': 0.0},
        {'params': head.parameters(), 'weight_decay': 5e-4}
    ], lr=0.01, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    
    backbone.train()
    head.train()
    
    start_event_fwd = torch.cuda.Event(enable_timing=True)
    end_event_fwd = torch.cuda.Event(enable_timing=True)
    
    start_event_bwd = torch.cuda.Event(enable_timing=True)
    end_event_bwd = torch.cuda.Event(enable_timing=True)
    
    start_event_opt = torch.cuda.Event(enable_timing=True)
    end_event_opt = torch.cuda.Event(enable_timing=True)

    time_dataload = 0.0
    time_fwd = 0.0
    time_bwd = 0.0
    time_opt = 0.0
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Warmup
    try:
        for _ in range(5):
            batch = next(loader_iter)
            imgs = batch[0].to(device)
            labels = batch[1].to(device)
            optimizer.zero_grad()
            emb, norm = backbone(imgs)
            out, _ = head(emb, norm, labels)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"[{batch_size}] OOM during warmup!")
            return False, None
        raise e
        
    torch.cuda.synchronize()
    
    print(f"[{batch_size}] Warmup complete. Benchmarking {max_steps} steps...")
    
    total_start = time.perf_counter()
    
    for step in range(max_steps):
        try:
            # DataLoader timing (CPU)
            t0_dl = time.perf_counter()
            batch = next(loader_iter)
            imgs = batch[0].to(device)
            labels = batch[1].to(device)
            t1_dl = time.perf_counter()
            time_dataload += (t1_dl - t0_dl)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward timing (GPU)
            torch.cuda.synchronize()
            start_event_fwd.record()
            emb, norm = backbone(imgs)
            out, _ = head(emb, norm, labels)
            loss = criterion(out, labels)
            end_event_fwd.record()
            
            # Backward timing (GPU)
            torch.cuda.synchronize()
            start_event_bwd.record()
            loss.backward()
            end_event_bwd.record()
            
            # Optimizer timing (GPU)
            torch.cuda.synchronize()
            start_event_opt.record()
            optimizer.step()
            end_event_opt.record()
            
            torch.cuda.synchronize()
            time_fwd += start_event_fwd.elapsed_time(end_event_fwd) / 1000.0
            time_bwd += start_event_bwd.elapsed_time(end_event_bwd) / 1000.0
            time_opt += start_event_opt.elapsed_time(end_event_opt) / 1000.0
            
            if (step + 1) % 100 == 0:
                print(f"[{batch_size}] Step {step+1}/{max_steps} complete.")
                
        except StopIteration:
            loader_iter = iter(loader)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"[{batch_size}] OOM during step {step}!")
                return False, None
            raise e

    total_time = time.perf_counter() - total_start
    total_images = max_steps * batch_size
    imgs_sec = total_images / total_time
    
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    
    results = {
        "batch_size": batch_size,
        "DataLoader wait": time_dataload,
        "Forward": time_fwd,
        "Backward": time_bwd,
        "Optimizer": time_opt,
        "Total iteration": total_time,
        "Images/sec": imgs_sec,
        "Peak Mem (MB)": peak_mem
    }
    
    print(f"\n--- RESULTS FOR BATCH SIZE {batch_size} ---")
    for k, v in results.items():
        if k in ["DataLoader wait", "Forward", "Backward", "Optimizer", "Total iteration"]:
            print(f"  {k}: {v:.2f} s")
        else:
            print(f"  {k}: {v:.2f}")
            
    return True, results

def main():
    print("=" * 50)
    print("100K TRAINING BENCHMARK")
    print("=" * 50)
    
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA NOT AVAILABLE. Exiting.")
        sys.exit(1)
        
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    
    # We will test on 0% noise HQ train dataset just for speed
    manifest_path = str(PROJECT_ROOT / "data" / "splits" / "100k" / "0pct" / "clean_high_train.csv")
    dataset = ImageManifestDataset([manifest_path], PROJECT_ROOT, is_train=True)
    
    candidate_batch_sizes = [32, 64, 128]
    successful_results = []
    
    for bs in candidate_batch_sizes:
        success, res = benchmark_batch_size(bs, device, dataset, max_steps=100)
        
        # Cleanup aggressively
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if success:
            successful_results.append(res)
        else:
            print(f"Batch size {bs} failed (OOM). Stopping search.")
            break
            
    if not successful_results:
        print("All batch sizes failed!")
        sys.exit(1)
        
    print("\n" + "=" * 50)
    print("FINAL BENCHMARK DECISION")
    print("=" * 50)
    best_run = successful_results[-1] # Largest stable batch size
    
    # Save the selected batch size to a config or json so the notebook can read it
    out_path = PROJECT_ROOT / "benchmark_selected.json"
    with open(out_path, "w") as f:
        json.dump(best_run, f)
        
    print(f"Selected Batch Size: {best_run['batch_size']}")
    print(f"Images/sec: {best_run['Images/sec']:.2f}")
    print(f"Peak GPU Mem: {best_run['Peak Mem (MB)']:.2f} MB")
    
if __name__ == "__main__":
    main()
