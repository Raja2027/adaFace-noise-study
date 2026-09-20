import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset

class PilotDataset(Dataset):
    def __init__(self, csv_path, data_root):
        """
        Args:
            csv_path: path to train.csv or val.csv
            data_root: root directory containing the images/ folder
        """
        self.data = pd.read_csv(csv_path)
        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.data_root / row['image_path']
        
        # 1. Load image (RGB)
        img = Image.open(img_path).convert('RGB')
        
        # Verify shape
        w, h = img.size
        if w != 112 or h != 112:
            raise ValueError(f"Image {img_path} has incorrect size: {w}x{h}. Expected 112x112.")
            
        # 2. Swap to BGR to exactly match official AdaFace convention
        # Official code: sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])
        img_bgr = np.array(img)[:, :, ::-1].copy()
        
        # 3. Apply transforms (Replacing torchvision)
        # ToTensor(): [0, 255] uint8 -> [0.0, 1.0] float32, and [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(img_bgr).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        
        # Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):
        # y = (x - mean) / std
        tensor = (tensor - 0.5) / 0.5
        
        class_index = torch.tensor(row['class_index'], dtype=torch.long)
        image_id = row['image_id']
        
        return tensor, class_index, image_id
