#!/usr/bin/env python3
"""
Sanity check the 100k dataset's low-quality images using a trained AdaFace checkpoint.
Verifies that the blur transformation predictably lowers the feature norm (quality indicator).
"""

import sys
import yaml
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from net import build_model
from tqdm import tqdm

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_pilot_adaface.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot_adaface.yaml"
QUALITY_PAIRS = PROJECT_ROOT / "data" / "splits" / "100k" / "quality_pairs.csv"


class QualityPairsDataset(Dataset):
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path).head(8000) # Use the first 8000 for a quick but substantial check
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        hq_path = PROJECT_ROOT / row['image_path']
        lq_path = PROJECT_ROOT / row['low_quality_path']
        
        hq_img = Image.open(hq_path).convert('RGB')
        lq_img = Image.open(lq_path).convert('RGB')
        
        hq_img = np.array(hq_img).astype(np.float32) / 255.0
        lq_img = np.array(lq_img).astype(np.float32) / 255.0
        
        hq_img = (hq_img - 0.5) / 0.5
        lq_img = (lq_img - 0.5) / 0.5
        
        hq_img = torch.from_numpy(hq_img.transpose((2, 0, 1)))
        lq_img = torch.from_numpy(lq_img.transpose((2, 0, 1)))
            
        return hq_img, lq_img

def main():
    print("=" * 50)
    print("100K DATASET QUALITY SANITY CHECK")
    print("=" * 50)
    
    if not CHECKPOINT_PATH.exists():
        print(f"\n[!] SKIPPED: Trained checkpoint not found at {CHECKPOINT_PATH}")
        print("This sanity check requires a CLEAN trained AdaFace checkpoint.")
        print("Do NOT substitute a randomly initialized model.")
        print("Dataset validation is not failed for this reason.")
        sys.exit(0)
        
    if not QUALITY_PAIRS.exists():
        print(f"\n[!] SKIPPED: Quality pairs not found at {QUALITY_PAIRS}")
        sys.exit(0)
        
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    backbone = build_model(config['model']['backbone']).to(device)
    
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    backbone.eval()
    
    dataset = QualityPairsDataset(QUALITY_PAIRS)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    hq_norms = []
    lq_norms = []
    
    print(f"Running inference on {len(dataset)} HQ/LQ pairs...")
    with torch.no_grad():
        for hq_imgs, lq_imgs in tqdm(loader):
            hq_imgs = hq_imgs.to(device)
            lq_imgs = lq_imgs.to(device)
            
            _, hq_norm = backbone(hq_imgs)
            _, lq_norm = backbone(lq_imgs)
            
            hq_norms.extend(hq_norm.squeeze().cpu().tolist())
            lq_norms.extend(lq_norm.squeeze().cpu().tolist())
            
    import numpy as np
    hq_norms = np.array(hq_norms)
    lq_norms = np.array(lq_norms)
    diff = hq_norms - lq_norms
    
    print("\n" + "=" * 50)
    print("SANITY CHECK RESULTS (AdaFace Quality Indicator)")
    print("=" * 50)
    print("High-Quality (HQ) Images:")
    print(f"  Mean feature norm: {hq_norms.mean():.4f}")
    print(f"  Std dev:           {hq_norms.std():.4f}")
    print(f"  Median:            {np.median(hq_norms):.4f}")
    
    print("\nLow-Quality (LQ) Controlled Blur Images:")
    print(f"  Mean feature norm: {lq_norms.mean():.4f}")
    print(f"  Std dev:           {lq_norms.std():.4f}")
    print(f"  Median:            {np.median(lq_norms):.4f}")
    
    print("\nDifference (HQ - LQ):")
    print(f"  Mean norm diff:    {diff.mean():.4f}")
    print(f"  Median norm diff:  {np.median(diff):.4f}")
    
    if diff.mean() > 0:
        print("\n[OK] Blur transformation successfully lowers the AdaFace quality indicator.")
    else:
        print("\n[WARNING] Blur transformation did NOT lower the quality indicator.")
        
if __name__ == "__main__":
    main()
