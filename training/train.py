import sys
import yaml
import argparse
from pathlib import Path
import time
import json
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAFACE_ROOT = PROJECT_ROOT / 'third_party' / 'AdaFace'

# Append to sys.path to import official AdaFace backbone
sys.path.append(str(ADAFACE_ROOT))
from net import build_model

# Append to sys.path to import our local modules
sys.path.append(str(PROJECT_ROOT))
from losses.adaface import AdaFace
from training.dataset import PilotDataset
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pilot_adaface.yaml')
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset & Dataloader
    train_dataset = PilotDataset(
        csv_path=PROJECT_ROOT / config['data']['train_csv'],
        data_root=PROJECT_ROOT / config['data']['data_root']
    )
    val_dataset = PilotDataset(
        csv_path=PROJECT_ROOT / config['data']['val_csv'],
        data_root=PROJECT_ROOT / config['data']['data_root']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        drop_last=False
    )
    
    # 2. Model (Backbone)
    print(f"Initializing backbone: {config['model']['backbone']}")
    backbone = build_model(config['model']['backbone'])
    backbone = backbone.to(device)
    
    # 3. Head (AdaFace Loss)
    print("Initializing AdaFace head")
    head = AdaFace(
        embedding_size=config['model']['embedding_size'],
        classnum=config['model']['classnum'],
        m=config['model']['adaface_m'],
        h=config['model']['adaface_h'],
        s=config['model']['adaface_s'],
        t_alpha=config['model']['adaface_t_alpha']
    )
    head = head.to(device)
    
    # 4. Optimizer & LR Scheduler
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
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [], 'time': []
    }
    
    best_val_acc = 0.0
    best_epoch = -1
    
    ckpt_dir = PROJECT_ROOT / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    logs_dir = PROJECT_ROOT / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    print("\n--- STARTING TRAINING ---")
    start_train_time = time.time()
    
    for epoch in range(1, config['training']['epochs'] + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = trainer.train_one_epoch(train_loader, epoch)
        val_loss, val_acc = trainer.validate(val_loader, epoch)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        elapsed = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['time'].append(elapsed)
        
        print(f"Epoch {epoch}/{config['training']['epochs']} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'backbone': backbone.state_dict(),
                'head': head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc
            }, ckpt_dir / 'best_pilot_adaface.pt')
            print("  [*] Saved best checkpoint")
            
    total_time = time.time() - start_train_time
    
    # Save final
    torch.save({
        'epoch': config['training']['epochs'],
        'backbone': backbone.state_dict(),
        'head': head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_acc': history['val_acc'][-1]
    }, ckpt_dir / 'final_pilot_adaface.pt')
    
    with open(logs_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    print("\n--- TRAINING COMPLETE ---")
    print(f"Total time: {total_time/60:.2f} mins")
    print(f"Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}")

if __name__ == '__main__':
    main()
