"""
Local training launcher for AdaFace Noise Study.
Trains on local GPU, saves checkpoints to Google Drive folder.
"""
import sys
import os
import yaml
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from net import build_model
from losses.adaface import AdaFace
from training.record_dataset import RecordDataset
from training.trainer import Trainer

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

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT).decode('ascii').strip()
    except Exception:
        return "Not available"

def get_adaface_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_ROOT / 'third_party' / 'AdaFace'
        ).decode('ascii').strip()
    except Exception:
        return "Not available"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    parser.add_argument('--sync_dir', type=str, required=True, 
                        help="Directory to save checkpoints/logs (e.g. Google Drive path)")
    parser.add_argument('--noise_map', type=str, default=None,
                        help="Path to JSON noise map (mapping record_id -> assigned_label)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sync_dir = Path(args.sync_dir)
    sync_dir.mkdir(parents=True, exist_ok=True)
    
    # Noise map loading
    noise_map_dict = {}
    if args.noise_map:
        with open(args.noise_map, 'r') as f:
            noise_map_dict = {int(k): int(v) for k, v in json.load(f).items()}
            
    # 1. Dataset
    rec_path = PROJECT_ROOT / config['data']['rec_path']
    idx_path = PROJECT_ROOT / config['data']['idx_path']
    label_map_path = PROJECT_ROOT / config['data']['label_map_path']
    
    dataset = RecordDataset(rec_path, idx_path, label_map_path, noise_map=noise_map_dict)
    
    num_workers = config['training'].get('num_workers', 4)
    
    train_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    # 2. Model & Head
    backbone = build_model(config['model']['backbone']).to(device)
    head = AdaFace(
        embedding_size=config['model']['embedding_size'],
        classnum=config['model']['classnum'],
        m=config['model']['adaface_m'],
        h=config['model']['adaface_h'],
        s=config['model']['adaface_s'],
        t_alpha=config['model']['adaface_t_alpha']
    ).to(device)
    
    # 3. Optimizer & Scheduler
    paras_wo_bn, paras_only_bn = split_parameters(backbone)
    optimizer = optim.SGD([
        {'params': paras_wo_bn + [head.kernel], 'weight_decay': float(config['training']['weight_decay'])},
        {'params': paras_only_bn, 'weight_decay': 0.0}
    ], lr=config['training']['learning_rate'], momentum=config['training']['momentum'])

    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training']['lr_milestones'],
        gamma=config['training']['lr_gamma']
    )

    trainer = Trainer(backbone, head, optimizer, device)
    
    # 4. Fail-Safe Validation
    print("\n--- FAIL-SAFE VALIDATION ---")
    git_hash = get_git_commit()
    adaface_hash = get_adaface_commit()
    assert rec_path.exists(), f"CRITICAL: {rec_path} does not exist!"
    assert idx_path.exists(), f"CRITICAL: {idx_path} does not exist!"
    assert label_map_path.exists(), f"CRITICAL: {label_map_path} does not exist!"
    assert len(dataset) > 0, "CRITICAL: Dataset is empty!"
    assert config['model']['classnum'] == 10572, f"CRITICAL: Expected 10572 classes, got {config['model']['classnum']}"
    assert torch.cuda.is_available(), "CRITICAL: CUDA GPU is not available!"
    print("All fail-safe validations passed successfully.\n")

    # 5. Run Manifest
    manifest = {
        'experiment_name': config['experiment']['name'],
        'noise_percentage': config['experiment']['noise_percentage'],
        'dataset_rec': str(rec_path),
        'dataset_len': len(dataset),
        'seed': config['training']['seed'],
        'model_backbone': config['model']['backbone'],
        'embedding_dim': config['model']['embedding_size'],
        'adaface_m': config['model']['adaface_m'],
        'adaface_h': config['model']['adaface_h'],
        'adaface_s': config['model']['adaface_s'],
        'adaface_t_alpha': config['model']['adaface_t_alpha'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'optimizer': 'SGD',
        'scheduler': 'MultiStepLR',
        'epochs': config['training']['epochs'],
        'checkpoint_freq': config['training']['checkpoint_freq'],
        'sync_dir': str(sync_dir),
        'SOURCE_GIT_COMMIT': git_hash,
        'ADAFACE_UPSTREAM_COMMIT': adaface_hash,
        'gpu_model': torch.cuda.get_device_name(0),
        'platform': 'local'
    }
    
    print("="*50)
    print("RUN MANIFEST")
    print("="*50)
    for k, v in manifest.items():
        print(f"{k}: {v}")
    print("="*50 + "\n")
    
    with open(sync_dir / 'run_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4)
        
    shutil.copy(args.config, sync_dir / 'config.yaml')
        
    # 6. Resume Logic
    start_epoch = 1
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'lr': [], 'time': []}
    
    latest_ckpt = sync_dir / 'latest.pt'
    if latest_ckpt.exists():
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        backbone.load_state_dict(ckpt['backbone'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        history = ckpt.get('history', history)
        
    print(f"\n--- STARTING TRAINING FROM EPOCH {start_epoch} ---")
    
    import pandas as pd
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = trainer.train_one_epoch(train_loader, epoch)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        elapsed = time.time() - epoch_start
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(current_lr)
        history['time'].append(elapsed)
        
        print(f"Epoch {epoch}/{config['training']['epochs']} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}")
        
        # Save latest
        ckpt_state = {
            'epoch': epoch,
            'backbone': backbone.state_dict(),
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history': history
        }
        
        torch.save(ckpt_state, sync_dir / 'latest.pt')
        
        # Save intermediate
        if epoch % config['training']['checkpoint_freq'] == 0 or epoch == config['training']['epochs']:
            torch.save(ckpt_state, sync_dir / f'epoch_{epoch:02d}.pt')
            print(f"  [*] Saved epoch_{epoch:02d}.pt")
            
        # Write CSV log
        pd.DataFrame(history).to_csv(sync_dir / 'train_log.csv', index=False)
            
    print("\n--- TRAINING COMPLETE ---")

if __name__ == '__main__':
    main()
