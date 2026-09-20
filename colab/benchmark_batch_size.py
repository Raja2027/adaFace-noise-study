import sys
import gc
from pathlib import Path
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from net import build_model
from losses.adaface import AdaFace
from training.record_dataset import RecordDataset

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Cannot run memory benchmark.")
        sys.exit(1)
        
    config_path = PROJECT_ROOT / "configs/noise_0.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    rec_path = PROJECT_ROOT / config['data']['rec_path']
    idx_path = PROJECT_ROOT / config['data']['idx_path']
    label_map_path = PROJECT_ROOT / config['data']['label_map_path']
    
    dataset = RecordDataset(rec_path, idx_path, label_map_path)
    
    batch_sizes = [64, 128, 256, 512]
    best_batch = 64
    
    print("="*50)
    print("AdaFace Physical Batch Size Benchmark")
    print("="*50)
    
    for bs in batch_sizes:
        print(f"\nTesting Batch Size: {bs}")
        try:
            # Clear cache before each attempt
            gc.collect()
            torch.cuda.empty_cache()
            
            # Setup Models
            backbone = build_model(config['model']['backbone']).cuda()
            head = AdaFace(
                embedding_size=config['model']['embedding_size'],
                classnum=config['model']['classnum'],
                m=config['model']['adaface_m'],
                h=config['model']['adaface_h'],
                s=config['model']['adaface_s'],
                t_alpha=config['model']['adaface_t_alpha']
            ).cuda()
            
            optimizer = optim.SGD([{'params': backbone.parameters()}, {'params': head.parameters()}], lr=0.1)
            
            # Setup DataLoader
            loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
            
            # Simulate 3 forward/backward passes
            backbone.train()
            head.train()
            
            for i, (images, true_labels, ids, assigned_labels) in enumerate(loader):
                if i >= 3:
                    break
                    
                images = images.cuda()
                assigned_labels = assigned_labels.cuda()
                
                optimizer.zero_grad()
                embeddings, norms = backbone(images)
                head_out = head(embeddings, norms, assigned_labels)
                if isinstance(head_out, tuple):
                    loss = head_out[0]
                else:
                    loss = head_out
                loss.mean().backward()
                optimizer.step()
                
            print(f"  [SUCCESS] Batch size {bs} completed 3 iterations stably.")
            best_batch = bs
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  [OOM] Batch size {bs} caused Out of Memory error.")
                break
            else:
                print(f"  [ERROR] {e}")
                break
                
        finally:
            del loader
            del optimizer
            del head
            del backbone
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "="*50)
    print(f"RECOMMENDED PHYSICAL BATCH SIZE: {best_batch}")
    print("Update your config.yaml with this batch_size before running train_colab.py!")
    print("="*50)

if __name__ == "__main__":
    main()
