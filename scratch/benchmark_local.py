"""Benchmark maximum physical batch size for local RTX 3050 6GB GPU."""
import sys
import torch
import torch.optim as optim
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from net import build_model
from losses.adaface import AdaFace

def benchmark():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    backbone = build_model('ir_50').to(device)
    head = AdaFace(
        embedding_size=512, classnum=10572,
        m=0.4, h=0.333, s=64.0, t_alpha=0.01
    ).to(device)
    
    optimizer = optim.SGD([
        {'params': backbone.parameters()},
        {'params': head.parameters()}
    ], lr=0.1)
    
    best = 0
    for bs in [16, 32, 48, 64, 96, 128]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(3):
                images = torch.randn(bs, 3, 112, 112, device=device)
                labels = torch.randint(0, 10572, (bs,), device=device)
                optimizer.zero_grad()
                emb, norms = backbone(images)
                head_out = head(emb, norms, labels)
                loss = head_out[0] if isinstance(head_out, tuple) else head_out
                loss.mean().backward()
                optimizer.step()
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  batch_size={bs:>3d}: [SUCCESS] Peak VRAM: {peak_mb:.0f} MB")
            best = bs
        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={bs:>3d}: [OOM]")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  batch_size={bs:>3d}: [ERROR] {e}")
            break
    
    print(f"\nRecommended batch size: {best}")

if __name__ == '__main__':
    benchmark()
