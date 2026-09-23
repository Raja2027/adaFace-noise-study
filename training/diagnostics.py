import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

class DiagnosticProbeDataset(Dataset):
    def __init__(self, probe_conditions_csv, project_root):
        self.df = pd.read_csv(probe_conditions_csv)
        self.project_root = project_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        
        # Paths based on 100k dataset structure
        hq_path = os.path.join(self.project_root, "data", "splits", "100k", "images", "heldout", f"{image_id}.png")
        lq_path = os.path.join(self.project_root, "data", "corrupted", "100k_blur_15x15_s5", "heldout", f"{image_id}.png")
        
        hq_img = np.array(Image.open(hq_path).convert('RGB')).astype(np.float32) / 255.0
        lq_img = np.array(Image.open(lq_path).convert('RGB')).astype(np.float32) / 255.0
        
        hq_img = (hq_img - 0.5) / 0.5
        lq_img = (lq_img - 0.5) / 0.5
        
        hq_img = torch.from_numpy(hq_img.transpose((2, 0, 1)).copy())
        lq_img = torch.from_numpy(lq_img.transpose((2, 0, 1)).copy())
        
        true_id = row['true_identity']
        false_id = row['false_identity']
        
        return {
            'image_id': image_id,
            'hq_img': hq_img,
            'lq_img': lq_img,
            'true_id': true_id,
            'false_id': false_id
        }

def compute_gradients(embeddings, norms, labels, head, criterion):
    """
    Computes per-sample gradient norm w.r.t head kernel and embeddings.
    Assumes batch_size = 1.
    """
    head.zero_grad(set_to_none=True)
    if embeddings.grad is not None:
        embeddings.grad.zero_()
        
    out, _ = head(embeddings, norms, labels)
    loss = criterion(out, labels)
    
    grads = torch.autograd.grad(loss, (head.kernel, embeddings), create_graph=False)
    g_head = torch.norm(grads[0], p=2).item()
    g_emb = torch.norm(grads[1], p=2).item()
    
    return g_head, g_emb, loss.item()

class DiagnosticRunner:
    def __init__(self, probe_conditions_csv, project_root, device):
        self.dataset = DiagnosticProbeDataset(probe_conditions_csv, project_root)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def run(self, backbone, head, epoch):
        # Ensure eval mode so BN/Dropout behave deterministically
        backbone.eval()
        head.eval()
        
        # Turn off EMA updates during diagnostic
        head.update_ema = False
        
        results = []
        
        print(f"\nRunning Diagnostics for Epoch {epoch}...")
        
        for batch in tqdm(self.loader, desc="Diagnostics"):
            image_id = batch['image_id'][0]
            hq_img = batch['hq_img'].to(self.device)
            lq_img = batch['lq_img'].to(self.device)
            true_id = batch['true_id'].to(self.device)
            false_id = batch['false_id'].to(self.device)
            
            # 1. Forward backbone once for HQ and LQ
            with torch.no_grad():
                hq_emb, hq_norm = backbone(hq_img)
                lq_emb, lq_norm = backbone(lq_img)
                
            hq_emb.requires_grad_(True)
            lq_emb.requires_grad_(True)
            
            # Conditions:
            # A: HQ + true_id
            # B: HQ + false_id
            # C: LQ + true_id
            # D: LQ + false_id
            
            conditions = [
                ('A', hq_emb, hq_norm, true_id, 'HQ', 0),
                ('B', hq_emb, hq_norm, false_id, 'HQ', 1),
                ('C', lq_emb, lq_norm, true_id, 'LQ', 0),
                ('D', lq_emb, lq_norm, false_id, 'LQ', 1)
            ]
            
            for cond_name, emb, norm, label, qual, is_noisy in conditions:
                
                # --- ACTUAL ---
                head.force_q_zero = False
                g_actual_head, g_actual_emb, loss_actual = compute_gradients(emb, norm, label, head, self.criterion)
                q_actual = head.margin_scaler if hasattr(head, 'margin_scaler') else None
                # re-forward to get properties
                with torch.no_grad():
                    out_actual, q_actual = head(emb, norm, label)
                    p_assigned = torch.softmax(out_actual, dim=1)[0, label[0]].item()
                
                # --- NEUTRAL ---
                head.force_q_zero = True
                g_neutral_head, g_neutral_emb, _ = compute_gradients(emb, norm, label, head, self.criterion)
                
                delta_g_head = g_actual_head - g_neutral_head
                delta_g_emb = g_actual_emb - g_neutral_emb
                
                results.append({
                    'image_id': image_id,
                    'true_identity': batch['true_id'].item(),
                    'assigned_identity': label.item(),
                    'quality_condition': qual,
                    'is_noisy': is_noisy,
                    'condition': cond_name,
                    'epoch': epoch,
                    'raw_feature_norm': norm.item(),
                    'AdaFace_quality_indicator': q_actual.item() if q_actual is not None else 0.0,
                    'P_assigned': p_assigned,
                    'per_sample_loss': loss_actual,
                    'G_actual': g_actual_head,
                    'G_neutral': g_neutral_head,
                    'Delta_G': delta_g_head,
                    'G_embedding_actual': g_actual_emb,
                    'G_embedding_neutral': g_neutral_emb,
                    'Delta_G_embedding': delta_g_emb
                })
                
        head.update_ema = True
        head.force_q_zero = False
        
        return pd.DataFrame(results)
