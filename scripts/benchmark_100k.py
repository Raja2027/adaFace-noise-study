#!/usr/bin/env python3
import sys
import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from training.image_dataset import ImageManifestDataset
from net import build_model

def main():
    print("=" * 50)
    print("100K DATALOADER BENCHMARK")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_workers = 0
    
    manifest_path = str(PROJECT_ROOT / "data" / "splits" / "100k" / "0pct" / "clean_high_train.csv")
    dataset = ImageManifestDataset([manifest_path], PROJECT_ROOT, is_train=True)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    backbone = build_model('ir_50').to(device)
    backbone.train()
    
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Device: {device}")
    
    start_time = time.time()
    steps = 0
    max_steps = 500 # benchmark 500 steps
    
    print(f"Benchmarking for {max_steps} steps...")
    try:
        for batch in loader:
            imgs = batch[0].to(device)
            # Dummy forward
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                out = backbone(imgs)
            steps += 1
            if steps % 50 == 0:
                print(f"  Step {steps}/{max_steps}...")
            if steps >= max_steps:
                break
    except KeyboardInterrupt:
        pass
        
    elapsed = time.time() - start_time
    imgs_sec = (steps * batch_size) / elapsed
    
    print("\n--- RESULTS ---")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Steps: {steps}")
    print(f"Images/sec: {imgs_sec:.2f}")
    if torch.cuda.is_available():
        print(f"Max GPU Mem: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
