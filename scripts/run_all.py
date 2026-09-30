#!/usr/bin/env python3
"""
Runs the full 100k AdaFace noise study sequentially.
"""
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("STARTING FULL ADAFACE 100K NOISE STUDY MATRIX")
    print("=" * 60)
    
    experiments = [
        ('HQ', '0'),
        ('HQ', '5'),
        ('HQ', '10'),
        ('HQ', '20'),
        ('LQ', '0'),
        ('LQ', '5'),
        ('LQ', '10'),
        ('LQ', '20')
    ]
    
    project_root = Path(__file__).resolve().parent.parent
    train_script = project_root / 'scripts' / 'train_100k.py'
    
    for quality, noise in experiments:
        print(f"\n[{quality}-{noise}] Launching experiment...")
        cmd = [
            sys.executable, str(train_script),
            '--quality', quality,
            '--noise_level', noise
        ]
        
        try:
            # Check if this experiment is already marked COMPLETED in Google Drive
            # train_100k.py handles the actual skip/resume logic, but we run it sequentially.
            subprocess.run(cmd, check=True)
            print(f"[{quality}-{noise}] Execution finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[{quality}-{noise}] ERROR: Experiment failed with exit code {e.returncode}.")
            print("Stopping the matrix runner due to failure.")
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print("ALL 8 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
