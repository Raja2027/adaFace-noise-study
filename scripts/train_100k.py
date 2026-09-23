#!/usr/bin/env python3
"""
Main training script for the 100k AdaFace noise study.
"""
import sys
import os
import argparse
import time
import json
import shutil
import subprocess
from pathlib import Path

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from net import build_model
from losses.adaface import AdaFace
from training.image_dataset import ImageManifestDataset
from training.trainer import Trainer
from training.diagnostics import DiagnosticRunner

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    return params_decay, params_no_decay

def get_git_commit(cwd):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    except Exception:
        return "Not available"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quality', type=str, choices=['HQ', 'LQ'], required=True)
    parser.add_argument('--noise_level', type=str, choices=['0', '5', '10', '20'], required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dry_run', action='store_true', help="Run 2 epochs of 10 batches for smoke testing")
    args = parser.parse_args()

    seed = 42
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    experiment_name = f"{args.quality}-{args.noise_level}"
    sync_dir = PROJECT_ROOT / "results" / "100k" / experiment_name
    sync_dir.mkdir(parents=True, exist_ok=True)
    (sync_dir / 'checkpoint').mkdir(exist_ok=True)
    (sync_dir / 'diagnostics').mkdir(exist_ok=True)
    (sync_dir / 'evaluation').mkdir(exist_ok=True)

    level_dir = PROJECT_ROOT / "data" / "splits" / "100k" / f"{args.noise_level}pct"
    
    if args.quality == 'HQ':
        train_manifests = [str(level_dir / "clean_high_train.csv"), str(level_dir / "noisy_high_train.csv")]
    else:
        train_manifests = [str(level_dir / "clean_low_train.csv"), str(level_dir / "noisy_low_train.csv")]
        
    heldout_manifest = str(PROJECT_ROOT / "data" / "splits" / "100k" / "heldout.csv")
    probe_conditions = str(PROJECT_ROOT / "data" / "splits" / "100k" / "probe_conditions.csv")
    
    train_dataset = ImageManifestDataset(train_manifests, PROJECT_ROOT, is_train=True)
    val_dataset = ImageManifestDataset(heldout_manifest, PROJECT_ROOT, is_train=False)
    
    if args.dry_run:
        print("DRY RUN: Limiting train to 10 batches")
        train_dataset.df = train_dataset.df.head(args.batch_size * 10)
        val_dataset.df = val_dataset.df.head(args.batch_size * 5)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    backbone = build_model('ir_50').to(device)
    head = AdaFace(embedding_size=512, classnum=2000, m=0.4, h=0.333, s=64.0, t_alpha=0.01).to(device)
    
    paras_wo_bn, paras_only_bn = split_parameters(backbone)
    optimizer = optim.SGD([
        {'params': paras_wo_bn + [head.kernel], 'weight_decay': 5e-4},
        {'params': paras_only_bn, 'weight_decay': 0.0}
    ], lr=args.lr, momentum=0.9)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 35], gamma=0.1)
    trainer = Trainer(backbone, head, optimizer, device)
    diag_runner = DiagnosticRunner(probe_conditions, PROJECT_ROOT, device)
    
    manifest = {
        'experiment_name': experiment_name,
        'seed': seed,
        'git_commit': get_git_commit(PROJECT_ROOT),
        'noise_level': args.noise_level,
        'quality_condition': args.quality,
        'train_count': len(train_dataset),
        'validation_count': len(val_dataset),
        'batch_size': args.batch_size,
        'epochs': args.epochs if not args.dry_run else 2,
        'optimizer': 'SGD',
        'scheduler': 'MultiStepLR',
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(sync_dir / 'run_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4)
        
    start_epoch = 1
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [], 'time': []}
    
    latest_ckpt = sync_dir / 'checkpoint' / 'latest.pt'
    if latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        backbone.load_state_dict(ckpt['backbone'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        history = ckpt.get('history', history)
        
    print(f"\n--- STARTING {experiment_name} (Epoch {start_epoch}) ---")
    best_acc = 0.0
    
    epochs = 2 if args.dry_run else args.epochs
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = trainer.train_one_epoch(train_loader, epoch)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        val_loss, val_acc = trainer.validate(val_loader, epoch)
        
        elapsed = time.time() - epoch_start
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['time'].append(elapsed)
        
        print(f"Epoch {epoch}/{epochs} - {elapsed:.1f}s | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        ckpt_state = {
            'epoch': epoch,
            'backbone': backbone.state_dict(),
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history': history
        }
        
        torch.save(ckpt_state, latest_ckpt)
        
        if not args.dry_run:
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(ckpt_state, sync_dir / 'checkpoint' / 'best.pt')
                
            if epoch % 5 == 0 or epoch == epochs:
                torch.save(ckpt_state, sync_dir / 'checkpoint' / f'epoch_{epoch:02d}.pt')
                
            pd.DataFrame(history).to_csv(sync_dir / 'training_log.csv', index=False)
            
            # Diagnostics every 5 epochs
            if epoch % 5 == 0 or epoch == epochs:
                diag_df = diag_runner.run(backbone, head, epoch)
                diag_df.to_csv(sync_dir / 'diagnostics' / f'epoch_{epoch:02d}.csv', index=False)

    if args.dry_run:
        # Test diagnostics runs successfully without crashing in dry_run
        diag_runner.loader.dataset.df = diag_runner.loader.dataset.df.head(10)
        diag_df = diag_runner.run(backbone, head, epoch)
        print("Dry run complete. Diagnostics ran successfully.")
        
    print("\n--- TRAINING COMPLETE ---")

if __name__ == '__main__':
    main()
