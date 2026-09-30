#!/usr/bin/env python3
"""
100k Training Benchmark
=======================
Benchmarks the full training pipeline at batch sizes 32, 64, 128.
Reports separate DataLoader fetch, H2D transfer, Forward, Backward,
Optimizer timings plus Images/sec and peak GPU memory.

This is a BENCHMARK ONLY — no full training is launched.
"""
import sys
import time
import inspect
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
from losses.adaface import AdaFace

NUM_WORKERS = 2


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_adaface_module():
    """Assert that AdaFace resolves to losses/adaface.py, NOT third_party."""
    module_path = inspect.getfile(AdaFace)
    print(f"AdaFace module path: {module_path}")
    assert "losses" in module_path and "adaface" in module_path, (
        f"FATAL: AdaFace resolved to {module_path} — expected losses/adaface.py"
    )
    print("[OK] AdaFace module verified: losses/adaface.py")


def verify_label_range(labels, classnum=2000):
    """Assert labels are in valid range [0, classnum)."""
    lmin = labels.min().item()
    lmax = labels.max().item()
    print(f"Label range: min={lmin}, max={lmax} (valid: [0, {classnum}))")
    assert lmin >= 0, f"FATAL: label min {lmin} < 0"
    assert lmax < classnum, f"FATAL: label max {lmax} >= {classnum}"
    print("[OK] Label range verified.")


def benchmark_batch_size(batch_size, device, dataset, max_steps=100):
    print(f"\n{'='*50}")
    print(f"[{batch_size}] Testing Batch Size: {batch_size}")
    print(f"{'='*50}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    loader_iter = iter(loader)

    # Re-initialize models and optimizers to avoid memory fragmentation
    backbone = build_model('ir_50').to(device)
    head = AdaFace(
        embedding_size=512, classnum=2000,
        m=0.4, h=0.333, s=64.0, t_alpha=0.01
    ).to(device)

    # Parameter groups: BN weight decay = 0
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

    # CUDA events for GPU-side timing
    start_event_fwd = torch.cuda.Event(enable_timing=True)
    end_event_fwd = torch.cuda.Event(enable_timing=True)
    start_event_bwd = torch.cuda.Event(enable_timing=True)
    end_event_bwd = torch.cuda.Event(enable_timing=True)
    start_event_opt = torch.cuda.Event(enable_timing=True)
    end_event_opt = torch.cuda.Event(enable_timing=True)

    # Accumulators (separate DL fetch vs H2D transfer)
    time_dl_fetch = 0.0
    time_h2d = 0.0
    time_fwd = 0.0
    time_bwd = 0.0
    time_opt = 0.0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # --- Label range check on first batch ---
    try:
        first_batch = next(loader_iter)
        verify_label_range(first_batch[1], classnum=2000)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"[{batch_size}] OOM fetching first batch!")
            return False, None
        raise

    # --- Warmup (5 steps, results discarded) ---
    try:
        # Use the first batch we already fetched for warmup step 1
        imgs = first_batch[0].to(device)
        labels = first_batch[1].to(device)
        optimizer.zero_grad()
        emb, norm = backbone(imgs)
        out, _ = head(emb, norm, labels)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        for _ in range(4):
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
        raise

    torch.cuda.synchronize()
    print(f"[{batch_size}] Warmup complete. Benchmarking {max_steps} steps...")

    total_start = time.perf_counter()

    for step in range(max_steps):
        try:
            # A. DataLoader fetch time (CPU)
            t0_dl = time.perf_counter()
            batch = next(loader_iter)
            t1_dl = time.perf_counter()
            time_dl_fetch += (t1_dl - t0_dl)

            # B. Host-to-GPU transfer time (CPU)
            t0_h2d = time.perf_counter()
            imgs = batch[0].to(device)
            labels = batch[1].to(device)
            t1_h2d = time.perf_counter()
            time_h2d += (t1_h2d - t0_h2d)

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

            if (step + 1) % 50 == 0:
                print(f"[{batch_size}] Step {step+1}/{max_steps} complete.")

        except StopIteration:
            loader_iter = iter(loader)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"[{batch_size}] OOM during step {step}!")
                return False, None
            raise

    total_time = time.perf_counter() - total_start
    total_images = max_steps * batch_size
    imgs_sec = total_images / total_time

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    results = {
        "batch_size": batch_size,
        "status": "PASS",
        "DataLoader fetch (s)": round(time_dl_fetch, 3),
        "H2D transfer (s)": round(time_h2d, 3),
        "Forward (s)": round(time_fwd, 3),
        "Backward (s)": round(time_bwd, 3),
        "Optimizer (s)": round(time_opt, 3),
        "Total iteration (s)": round(total_time, 3),
        "Images/sec": round(imgs_sec, 2),
        "Peak Mem (MB)": round(peak_mem, 2),
        "steps": max_steps,
    }

    print(f"\n--- RESULTS FOR BATCH SIZE {batch_size} ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Cleanup model from GPU before next batch size
    del backbone, head, optimizer, loader, loader_iter
    gc.collect()
    torch.cuda.empty_cache()

    return True, results


def main():
    print("=" * 60)
    print("100K TRAINING BENCHMARK")
    print("=" * 60)

    # --- Verify AdaFace module ---
    verify_adaface_module()

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA NOT AVAILABLE. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(device)
    vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_mb:.0f} MB")
    print(f"Workers: {NUM_WORKERS}")

    # Load the 0% noise HQ train dataset for benchmarking
    manifest_path = str(PROJECT_ROOT / "data" / "splits" / "100k" / "0pct" / "clean_high_train.csv")
    dataset = ImageManifestDataset([manifest_path], PROJECT_ROOT, is_train=True)

    candidate_batch_sizes = [32, 64, 128]
    all_results = []

    for bs in candidate_batch_sizes:
        success, res = benchmark_batch_size(bs, device, dataset, max_steps=100)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if success:
            all_results.append(res)
        else:
            all_results.append({
                "batch_size": bs,
                "status": "OOM",
            })
            print(f"Batch size {bs} failed (OOM). Continuing to report.")
            # Don't break — record OOM and continue if possible

    successful_results = [r for r in all_results if r["status"] == "PASS"]

    if not successful_results:
        print("All batch sizes failed!")
        sys.exit(1)

    # --- Save full benchmark table ---
    results_path = PROJECT_ROOT / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull benchmark results saved to: {results_path}")

    # --- Select largest stable batch size ---
    best_run = successful_results[-1]

    selected_path = PROJECT_ROOT / "benchmark_selected.json"
    with open(selected_path, "w") as f:
        json.dump(best_run, f, indent=2)
    print(f"Selected batch size saved to: {selected_path}")

    # --- Final Report ---
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK REPORT")
    print("=" * 60)
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_mb:.0f} MB")
    print(f"AdaFace module: {inspect.getfile(AdaFace)}")
    print(f"Workers: {NUM_WORKERS}")
    print()

    for res in all_results:
        bs = res["batch_size"]
        status = res["status"]
        print(f"--- Batch {bs} [{status}] ---")
        if status == "PASS":
            print(f"  DataLoader fetch: {res['DataLoader fetch (s)']} s")
            print(f"  H2D transfer:     {res['H2D transfer (s)']} s")
            print(f"  Forward:          {res['Forward (s)']} s")
            print(f"  Backward:         {res['Backward (s)']} s")
            print(f"  Optimizer:        {res['Optimizer (s)']} s")
            print(f"  Total iteration:  {res['Total iteration (s)']} s")
            print(f"  Images/sec:       {res['Images/sec']}")
            print(f"  Peak Mem:         {res['Peak Mem (MB)']} MB")
        print()

    print(f"Selected Batch Size: {best_run['batch_size']}")
    print(f"Images/sec: {best_run['Images/sec']}")
    print(f"Peak GPU Mem: {best_run['Peak Mem (MB)']} MB")
    print()
    print("BENCHMARK COMPLETE. STOP.")


if __name__ == "__main__":
    main()
