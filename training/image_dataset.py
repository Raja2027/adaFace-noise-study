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
        
        # Build deterministic label map from master_100k.csv to map arbitrary CASIA IDs to 0..1999
        master_path = os.path.join(project_root, "data", "splits", "100k", "master_100k.csv")
        if os.path.exists(master_path):
            master_df = pd.read_csv(master_path)
            # The 2000 selected identities
            unique_ids = sorted(master_df['true_identity'].unique())
            self.label_map = {orig: new for new, orig in enumerate(unique_ids)}
        else:
            # Fallback if master doesn't exist (e.g. testing)
            unique_ids = sorted(self.df['true_identity'].unique())
            self.label_map = {orig: new for new, orig in enumerate(unique_ids)}

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
            
        true_label = self.label_map.get(row['true_identity'], 0)
        assigned_label = self.label_map.get(row['assigned_identity'], 0)
        image_id = row['image_id']
        
        return img, true_label, image_id, assigned_label
