import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageManifestDataset(Dataset):
    """
    Dataset that loads images from a CSV manifest containing:
    image_path, assigned_identity, true_identity
    """
    def __init__(self, manifest_paths, project_root, is_train=True):
        if isinstance(manifest_paths, str):
            manifest_paths = [manifest_paths]
            
        dfs = [pd.read_csv(p) for p in manifest_paths]
        self.df = pd.concat(dfs, ignore_index=True)
        self.project_root = project_root
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.project_root, row['image_path'])
        
        # We use PIL or cv2 for image loading
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (112, 112))
            
        img = np.array(img).astype(np.float32) / 255.0
        
        if self.is_train and np.random.rand() > 0.5:
            img = np.fliplr(img)
            
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
            
        true_label = row['true_identity']
        assigned_label = row['assigned_identity']
        image_id = row['image_id']
        
        return img, true_label, image_id, assigned_label
