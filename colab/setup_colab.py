import sys
import os
import subprocess
from pathlib import Path
import json

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("="*60)
    print("AdaFace Noise Study - Colab Environment Setup")
    print("="*60)
    
    # 1. Environment Verification
    print(f"Python Version: {sys.version}")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: No GPU detected. Colab runtime should be set to GPU.")
    except ImportError:
        print("PyTorch not installed yet. Will install via requirements.")

    # 2. Mount Google Drive (if in Colab)
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        print("Google Drive mounted at /content/drive")
    except ImportError:
        print("Not running in Google Colab environment or google.colab is not available.")
    
    # 3. Define Paths
    DRIVE_ROOT = Path('/content/drive/MyDrive/adaFace-noise-study')
    LOCAL_ROOT = Path('/content/adaFace-noise-study')
    
    # 4. Copy Codebase (Excluding data/raw to avoid unneeded sync overhead if it exists)
    if not LOCAL_ROOT.exists():
        print("Copying repository from Google Drive to local /content workspace...")
        # Assume the codebase is already zipped or cloned in Drive. 
        # For this script, we just copy the essential folders if they exist.
        LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        for folder in ['configs', 'losses', 'training', 'data']:
            src = DRIVE_ROOT / folder
            dst = LOCAL_ROOT / folder
            if src.exists() and not dst.exists():
                run_cmd(f"cp -r {src} {dst}")
                
    # 5. Copy Dataset
    print("Copying CASIA-WebFace dataset from Google Drive to local /content storage...")
    local_raw_dir = LOCAL_ROOT / 'data' / 'raw' / 'faces_webface_112x112'
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    
    for f in ['train.rec', 'train.idx']:
        src_f = DRIVE_ROOT / 'data' / 'raw' / 'faces_webface_112x112' / f
        dst_f = local_raw_dir / f
        if not dst_f.exists():
            if src_f.exists():
                run_cmd(f"cp {src_f} {dst_f}")
                print(f"Copied {f} successfully.")
            else:
                print(f"ERROR: Missing {src_f} on Google Drive!")
                sys.exit(1)
        else:
            print(f"{f} already exists locally.")
            
    # 6. Verify required files
    assert (local_raw_dir / 'train.rec').exists(), "train.rec not found locally!"
    assert (local_raw_dir / 'train.idx').exists(), "train.idx not found locally!"
    assert (LOCAL_ROOT / 'data/metadata/record_label_map.csv').exists(), "record_label_map.csv missing!"
    
    # 7. Reproducibility Summary
    print("\n--- Reproducibility Summary ---")
    print(f"Workspace initialized at: {LOCAL_ROOT}")
    print(f"Dataset path: {local_raw_dir}")
    print("Environment setup completed successfully. You are ready to run train_colab.py!")

if __name__ == '__main__':
    main()
