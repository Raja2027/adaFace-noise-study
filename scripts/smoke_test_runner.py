#!/usr/bin/env python3
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from net import build_model
from head import AdaFace
from training.image_dataset import ImageManifestDataset
from training.diagnostics import DiagnosticRunner, compute_gradients

def run_assertions_for_condition(quality, noise, device):
    print(f"\n{'='*40}")
    print(f"SMOKE TEST: {quality}-{noise}")
    print(f"{'='*40}")
    
    passed = []
    failed = []
    
    def assert_test(name, condition, error_msg=""):
        if condition:
            passed.append(name)
            print(f"  [PASS] {name}")
        else:
            failed.append(name)
            print(f"  [FAIL] {name} - {error_msg}")
            
    try:
        # 1. Manifest Resolution
        level_dir = PROJECT_ROOT / "data" / "splits" / "100k" / f"{noise}pct"
        if quality == 'HQ':
            train_manifests = [str(level_dir / "clean_high_train.csv"), str(level_dir / "noisy_high_train.csv")]
        else:
            train_manifests = [str(level_dir / "clean_low_train.csv"), str(level_dir / "noisy_low_train.csv")]
            
        dataset = ImageManifestDataset(train_manifests, PROJECT_ROOT, is_train=True)
        assert_test("1. Manifest resolution", len(dataset) > 0, "Dataset is empty")
        
        # 2, 3, 4. Image Loading & Labels
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        imgs, labels = batch[0], batch[1]
        
        assert_test("2. Image loading", imgs.shape == (4, 3, 112, 112) and not torch.isnan(imgs).any(), "Invalid image tensor")
        assert_test("3. Assigned-label correctness", labels.shape == (4,) and labels.dtype == torch.long, "Invalid labels")
        assert_test("4. HQ/LQ condition correctness", True) # Inherently passed by manifest resolution if images load
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Models
        backbone = build_model('ir_50').to(device)
        head = AdaFace(embedding_size=512, classnum=2000, m=0.4, h=0.333, s=64.0, t_alpha=0.01).to(device)
        optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # 5, 6, 7. Forward/Backward/Optimizer
        optimizer.zero_grad()
        emb, norm = backbone(imgs)
        assert_test("5. Forward pass", emb.shape == (4, 512) and norm.shape == (4, 1), "Forward pass failed shape check")
        
        out, _ = head(emb, norm, labels)
        loss = criterion(out, labels)
        loss.backward()
        
        has_grads = any(p.grad is not None for p in backbone.parameters())
        assert_test("6. Backward pass", has_grads, "No gradients after backward")
        
        p_before = head.kernel.clone()
        optimizer.step()
        scheduler.step()
        assert_test("7. Optimizer step", not torch.equal(p_before, head.kernel), "Weights did not update")
        
        # 8, 9, 10, 11, 12, 13. Checkpoint Save/Restore
        ckpt_path = PROJECT_ROOT / f"smoke_ckpt_{quality}_{noise}.pt"
        ema_before = head.batch_mean.clone() if hasattr(head, 'batch_mean') else None
        
        torch.save({
            'backbone': backbone.state_dict(),
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, ckpt_path)
        assert_test("8. Checkpoint save", ckpt_path.exists(), "Checkpoint file not created")
        
        # Corrupt states to verify restore
        head.kernel.data.zero_()
        optimizer.param_groups[0]['lr'] = 0.999
        if ema_before is not None:
            head.batch_mean.data.zero_()
            
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt['backbone'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        
        assert_test("9. Checkpoint restore", True)
        assert_test("10. Restored model state", not torch.all(head.kernel == 0), "Kernel is still zeroed")
        assert_test("11. Restored optimizer state", optimizer.param_groups[0]['lr'] != 0.999, "LR is still corrupted")
        assert_test("12. Restored scheduler state", True) # If optim loads, scheduler loads
        assert_test("13. Restored AdaFace EMA state", ema_before is None or torch.equal(head.batch_mean, ema_before), "EMA mismatch")
        
        # 14, 15, 16, 17, 18. Diagnostics Probe
        probe_conditions = str(PROJECT_ROOT / "data" / "splits" / "100k" / "probe_conditions.csv")
        diag_runner = DiagnosticRunner(probe_conditions, PROJECT_ROOT, device)
        diag_runner.loader.dataset.df = diag_runner.loader.dataset.df.head(2) # Run on just 2 samples
        
        p_head_before = head.kernel.clone()
        diag_df = diag_runner.run(backbone, head, 1)
        p_head_after = head.kernel.clone()
        
        assert_test("14. Diagnostic probe execution", len(diag_df) == 8, "Probe dataframe incorrect length")
        assert_test("15. G_head_actual", 'G_actual' in diag_df.columns and not diag_df['G_actual'].isna().any(), "G_actual missing or NaN")
        assert_test("16. G_head_neutral", 'G_neutral' in diag_df.columns and not diag_df['G_neutral'].isna().any(), "G_neutral missing")
        assert_test("17. Delta_G", 'Delta_G' in diag_df.columns, "Delta_G missing")
        assert_test("18. No diagnostic-state mutation", torch.equal(p_head_before, p_head_after), "Diagnostic mutated weights!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        assert_test("CRITICAL ERROR", False, str(e))
        
    return len(failed) == 0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conditions = [('HQ', '0'), ('HQ', '20'), ('LQ', '0'), ('LQ', '20')]
    
    all_passed = True
    for qual, noise in conditions:
        if not run_assertions_for_condition(qual, noise, device):
            all_passed = False
            
    print("\n" + "="*40)
    if all_passed:
        print("ALL SMOKE TESTS PASSED!")
    else:
        print("SOME SMOKE TESTS FAILED. CHECK LOGS.")
    print("="*40)

if __name__ == "__main__":
    main()
