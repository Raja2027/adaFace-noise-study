import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
import copy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from net import build_model
from losses.adaface import AdaFace
from training.diagnostics import compute_g_head

def main():
    print("=" * 50)
    print("DIAGNOSTIC INTEGRITY TEST")
    print("=" * 50)
    
    device = torch.device('cpu')
    
    # 1. Initialize tiny model
    print("Initializing components...")
    backbone = build_model('ir_50').to(device)
    head = AdaFace(
        embedding_size=512,
        classnum=2000,
        m=0.4,
        h=0.333,
        s=64.0,
        t_alpha=0.01
    ).to(device)
    
    backbone.eval()
    head.eval()
    head.update_ema = False
    
    criterion = nn.CrossEntropyLoss()
    
    # Dummy inputs
    emb = torch.randn(1, 512)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1) * 30.0 # Make norm around 30 so q != 0
    emb.requires_grad_(True)
    norm = torch.norm(emb, p=2, dim=1, keepdim=True)
    label = torch.tensor([42])
    
    # Copy state to verify nothing changes
    state_before = copy.deepcopy(head.state_dict())
    
    # ACTUAL
    print("Computing G_actual...")
    head.force_q_zero = False
    g_actual, loss_actual = compute_g_head(emb, norm, label, head, criterion)
    
    _, q_actual = head(emb, norm, label)
    q_actual = q_actual.item()
    print(f"  q_actual: {q_actual:.4f}")
    
    # NEUTRAL
    print("Computing G_neutral...")
    head.force_q_zero = True
    g_neutral, loss_neutral = compute_g_head(emb, norm, label, head, criterion)
    
    _, q_neutral = head(emb, norm, label)
    q_neutral = q_neutral.item()
    print(f"  q_neutral: {q_neutral:.4f}")
    
    # Restore
    head.force_q_zero = False
    
    # State check
    state_after = head.state_dict()
    state_changed = False
    for k in state_before:
        if not torch.equal(state_before[k], state_after[k]):
            print(f"  [!] State changed: {k}")
            state_changed = True
            
    # Validations
    print("\n--- VALIDATION ---")
    assert q_actual != 0.0, "q_actual is zero!"
    assert q_neutral == 0.0, "q_neutral is not zero!"
    assert not state_changed, "Model state changed during diagnostic!"
    print(f"G_actual: {g_actual:.4f}")
    print(f"G_neutral: {g_neutral:.4f}")
    print(f"Delta_G: {g_actual - g_neutral:.4f}")
    
    print("\n[OK] Diagnostic Integrity Test PASSED.")

if __name__ == "__main__":
    main()
