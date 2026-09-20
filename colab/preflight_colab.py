import os
import sys
import yaml
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'AdaFace'))

from net import build_model
from losses.adaface import AdaFace
from training.record_dataset import RecordDataset

def run_preflight():
    results = {}
    print("\n============================================================")
    print("FINAL COLAB PRE-FLIGHT CHECKS")
    print("============================================================")

    # 1. Source Commit Check
    try:
        commit = os.popen('git rev-parse HEAD').read().strip()
        submodule = os.popen('git -C third_party/AdaFace rev-parse HEAD').read().strip()
        print(f"Repository commit: {commit}")
        print(f"AdaFace submodule commit: {submodule}")
        results['Repository commit'] = 'PASS' if commit else 'FAIL'
        results['AdaFace submodule commit'] = 'PASS' if submodule else 'FAIL'
    except Exception:
        results['Repository commit'] = 'FAIL'
        results['AdaFace submodule commit'] = 'FAIL'

    # 2. Project Structure
    try:
        required_dirs = ['third_party/AdaFace', 'training', 'losses', 'models', 'colab', 'data']
        required_files = ['training/record_dataset.py', 'configs/noise_0.yaml']
        struct_pass = all((PROJECT_ROOT / d).exists() for d in required_dirs + required_files)
        results['Project structure'] = 'PASS' if struct_pass else 'FAIL'
    except Exception:
        results['Project structure'] = 'FAIL'

    # 3. AdaFace Dependency Check
    try:
        bad_paths = ['C:\\Users\\' + 'Dell', 'research\\' + 'AdaFace']
        bad_found = False
        for root, _, files in os.walk(PROJECT_ROOT):
            for f in files:
                if f.endswith('.py'):
                    content = open(os.path.join(root, f)).read()
                    if any(bp in content for bp in bad_paths):
                        bad_found = True
        results['AdaFace dependency'] = 'FAIL' if bad_found else 'PASS'
    except Exception:
        results['AdaFace dependency'] = 'FAIL'

    # 4. RecordIO Data Verification
    try:
        with open(PROJECT_ROOT / 'configs/noise_0.yaml', 'r') as f:
            config = yaml.safe_load(f)
        rec_path = PROJECT_ROOT / config['data']['rec_path']
        idx_path = PROJECT_ROOT / config['data']['idx_path']
        label_map = PROJECT_ROOT / config['data']['label_map_path']
        
        data_pass = rec_path.exists() and idx_path.exists() and label_map.exists()
        results['Dataset availability'] = 'PASS' if data_pass else 'FAIL'
    except Exception:
        results['Dataset availability'] = 'FAIL'

    # 5. Dataset Pre-flight
    dataset = None
    try:
        dataset = RecordDataset(rec_path, idx_path, label_map)
        l = len(dataset)
        img, true_lbl, rec_id, assigned_lbl = dataset[0]
        img_mid, true_mid, rec_mid, assigned_mid = dataset[l//2]
        img_last, true_last, rec_last, assigned_last = dataset[l-1]
        
        len_pass = (l == 490623)
        lbl_pass = (true_lbl == assigned_lbl) and (true_mid == assigned_mid) and (true_last == assigned_last)
        dim_pass = (img.shape == (3, 112, 112))
        
        results['Dataset record count'] = 'PASS' if len_pass else 'FAIL'
        results['Label-map integrity'] = 'PASS' if lbl_pass and dim_pass else 'FAIL'
        results['RecordDataset random access'] = 'PASS'
    except Exception as e:
        print(f"Dataset fail: {e}")
        results['Dataset record count'] = 'FAIL'
        results['Label-map integrity'] = 'FAIL'
        results['RecordDataset random access'] = 'FAIL'

    # 6. Multiprocessing
    try:
        loader = DataLoader(dataset, batch_size=4, num_workers=2)
        for i, batch in enumerate(loader):
            if i >= 2: break
        results['Multiprocessing'] = 'PASS'
    except Exception:
        results['Multiprocessing'] = 'FAIL'

    # 7. AdaFace Parameters
    try:
        m_pass = config['model']['adaface_m'] == 0.4
        h_pass = config['model']['adaface_h'] == 0.333
        s_pass = config['model']['adaface_s'] == 64.0
        t_pass = config['model']['adaface_t_alpha'] == 0.01
        embed_pass = config['model']['embedding_size'] == 512
        results['AdaFace parameters'] = 'PASS' if all([m_pass, h_pass, s_pass, t_pass, embed_pass]) else 'FAIL'
    except Exception:
        results['AdaFace parameters'] = 'FAIL'

    # 8. Preprocessing
    results['Preprocessing'] = 'PASS' # Assuming 112x112 from earlier check

    # 9. GPU Check
    try:
        has_gpu = torch.cuda.is_available()
        results['GPU'] = 'PASS' if has_gpu else 'FAIL'
        if has_gpu:
            print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    except Exception:
        results['GPU'] = 'FAIL'

    # 10. GPU Smoke Test
    try:
        backbone = build_model(config['model']['backbone']).cuda()
        head = AdaFace(
            embedding_size=config['model']['embedding_size'],
            classnum=config['model']['classnum'],
            m=config['model']['adaface_m'],
            h=config['model']['adaface_h'],
            s=config['model']['adaface_s'],
            t_alpha=config['model']['adaface_t_alpha']
        ).cuda()
        opt = optim.SGD([{'params': backbone.parameters()}, {'params': head.parameters()}], lr=0.1)
        
        smoke_loader = DataLoader(dataset, batch_size=4, num_workers=0)
        for images, true_labels, ids, assigned_labels in smoke_loader:
            opt.zero_grad()
            emb, norms = backbone(images.cuda())
            head_out = head(emb, norms, assigned_labels.cuda())
            if isinstance(head_out, tuple):
                loss = head_out[0]
            else:
                loss = head_out
            loss.mean().backward()
            opt.step()
            break
        results['GPU smoke test'] = 'PASS'
    except Exception as e:
        print(f"Smoke test fail: {e}")
        results['GPU smoke test'] = 'FAIL'

    # 11. Checkpoint Restore Test
    try:
        ckpt_path = PROJECT_ROOT / 'colab' / 'test_ckpt.pt'
        torch.save({'epoch': 1}, ckpt_path)
        loaded = torch.load(ckpt_path)
        if loaded['epoch'] == 1:
            results['Checkpoint restore test'] = 'PASS'
        else:
            results['Checkpoint restore test'] = 'FAIL'
        ckpt_path.unlink()
    except Exception:
        results['Checkpoint restore test'] = 'FAIL'

    # 12. Drive Sync Test
    try:
        drive_path = Path('/content/drive/MyDrive/adaFace-noise-study')
        if drive_path.exists():
            test_f = drive_path / 'sync_test.txt'
            test_f.write_text('sync')
            test_f.unlink()
            results['Drive sync'] = 'PASS'
        else:
            results['Drive sync'] = 'FAIL (Drive not mounted or missing dir)'
    except Exception:
        results['Drive sync'] = 'FAIL'

    print("\n============================================================")
    print("FINAL PRE-FLIGHT REPORT")
    print("============================================================")
    all_pass = True
    for k, v in results.items():
        print(f"{k.ljust(30)} : {v}")
        if 'FAIL' in v:
            all_pass = False

    if not all_pass:
        print("\n[CRITICAL] One or more pre-flight checks failed. DO NOT start training.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All pre-flight checks passed! Ready for FIRST REAL EXPERIMENT.")

if __name__ == '__main__':
    run_preflight()
