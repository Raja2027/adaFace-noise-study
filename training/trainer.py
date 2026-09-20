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
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass through backbone
            # Backbone returns (normalized_embeddings, norms)
            embeddings, norms = self.backbone(images)
            
            # Forward pass through AdaFace head
            # Head returns scaled logits
            logits = self.head(embeddings, norms, labels)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
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
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                embeddings, norms = self.backbone(images)
                logits = self.head(embeddings, norms, labels)
                
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)
                
                pbar.set_postfix({'Loss': loss.item(), 'Acc': correct/total})
                
        return total_loss / total, correct / total
