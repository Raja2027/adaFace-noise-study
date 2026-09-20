import sys
import yaml
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from net import build_model
from losses.adaface import AdaFace
from training.record_dataset import RecordDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml used during training")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint (.pt) file to evaluate")
    parser.add_argument('--output', type=str, required=True, help="Path to save evaluation metrics (JSON)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*50)
    print("EVALUATE COLAB")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Dataset: {config['data']['rec_path']}")
    print("="*50)

    # 1. Load Model
    backbone = build_model(config['model']['backbone']).to(device)
    head = AdaFace(
        embedding_size=config['model']['embedding_size'],
        classnum=config['model']['classnum'],
        m=config['model']['adaface_m'],
        h=config['model']['adaface_h'],
        s=config['model']['adaface_s'],
        t_alpha=config['model']['adaface_t_alpha']
    ).to(device)
    
    print("Loading weights...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    head.load_state_dict(ckpt['head'])
    
    backbone.eval()
    head.eval()
    
    print("Evaluation placeholder executed. (Implement LFW/AgeDB/CFP-FP metrics here when binaries are ready).")
    
    results = {
        'checkpoint_evaluated': args.checkpoint,
        'config': args.config,
        'seed': config['training']['seed'],
        'metrics': {
            'accuracy': 'N/A' # To be filled with actual evaluation
        }
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved evaluation metrics to {output_path}")

if __name__ == "__main__":
    with torch.no_grad():
        main()
