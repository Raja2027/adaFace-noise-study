import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, backbone, head, optimizer, device):
        self.backbone = backbone
        self.head = head
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def train_one_epoch(self, dataloader, epoch, log_interval=10):
        self.backbone.train()
        self.head.train()
        
        total_loss = 0.
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Train")
        for batch_idx, (images, true_labels, ids, assigned_labels) in enumerate(pbar):
            images = images.to(self.device)
            assigned_labels = assigned_labels.to(self.device)
            
            # Forward pass through backbone
            # Backbone returns (normalized_embeddings, norms)
            embeddings, norms = self.backbone(images)
            
            # Forward pass through AdaFace head
            # Head returns scaled logits and margin_scaler (which we ignore here)
            head_out = self.head(embeddings, norms, assigned_labels)
            if isinstance(head_out, tuple):
                logits = head_out[0]
            else:
                logits = head_out
                
            # Compute loss
            loss = self.criterion(logits, assigned_labels)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == assigned_labels).sum().item()
            total += images.size(0)
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({'Loss': loss.item(), 'Acc': correct/total})
                
        return total_loss / total, correct / total

    def validate(self, dataloader, epoch):
        self.backbone.eval()
        self.head.eval()
        
        total_loss = 0.
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch} Val")
            for images, true_labels, ids, assigned_labels in pbar:
                images = images.to(self.device)
                assigned_labels = assigned_labels.to(self.device)
                
                embeddings, norms = self.backbone(images)
                head_out = self.head(embeddings, norms, assigned_labels)
                if isinstance(head_out, tuple):
                    logits = head_out[0]
                else:
                    logits = head_out
                
                loss = self.criterion(logits, assigned_labels)
                
                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == assigned_labels).sum().item()
                total += images.size(0)
                
                pbar.set_postfix({'Loss': loss.item(), 'Acc': correct/total})
                
        return total_loss / total, correct / total
