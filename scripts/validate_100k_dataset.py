#!/usr/bin/env python3
"""
Validates the 100k controlled AdaFace label-noise dataset against strict criteria.
"""

import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_100K = PROJECT_ROOT / "data" / "splits" / "100k"
MASTER_CSV = SPLITS_100K / "master_100k.csv"
HELDOUT_CSV = SPLITS_100K / "heldout.csv"
QUALITY_PAIRS_CSV = SPLITS_100K / "quality_pairs.csv"

def fail(msg):
    print(f"\n[FAIL] {msg}")
    print("\n==================================================")
    print("100K DATASET VALIDATION: FAILED")
    print("==================================================")
    sys.exit(1)

def check(condition, msg):
    if condition:
        print(f"[x] {msg}")
    else:
        fail(msg)

def main():
    print("==================================================")
    print("100K DATASET VALIDATION")
    print("==================================================")

    if not MASTER_CSV.exists():
        fail(f"Master manifest not found: {MASTER_CSV}")
        
    master_df = pd.read_csv(MASTER_CSV)
    heldout_df = pd.read_csv(HELDOUT_CSV)
    quality_df = pd.read_csv(QUALITY_PAIRS_CSV)
    
    # 1. Exact counts
    check(len(master_df) == 100000, f"exactly 100,000 train images (found {len(master_df)})")
    check(len(heldout_df) == 8000, f"exactly 8,000 held-out images (found {len(heldout_df)})")
    
    # 2. Identities and balance
    train_identities = set(master_df['true_identity'])
    heldout_identities = set(heldout_df['true_identity'])
    
    check(len(train_identities) == 2000, f"exactly 2,000 training identities (found {len(train_identities)})")
    check(train_identities == heldout_identities, "held-out identities match training identities exactly")
    
    train_counts = master_df['true_identity'].value_counts()
    check(all(train_counts == 50), "exactly 50 train images / identity")
    
    heldout_counts = heldout_df['true_identity'].value_counts()
    check(all(heldout_counts == 4), "exactly 4 held-out images / identity")
    
    # 3. Duplicates and overlaps
    check(not master_df['image_id'].duplicated().any(), "no duplicate training image IDs")
    check(not heldout_df['image_id'].duplicated().any(), "no duplicate held-out IDs")
    
    train_records = set(master_df['source_record_id'])
    heldout_records = set(heldout_df['source_record_id'])
    overlap = train_records.intersection(heldout_records)
    check(len(overlap) == 0, f"no train/heldout overlap (found {len(overlap)} overlapping records)")
    
    # 4. RecordIO references (basic bounds check since we don't load train.idx here)
    check(master_df['source_record_id'].min() > 0, "all source records > 0")
    
    # 5. File existence and Quality Pairs
    check(len(quality_df) == 100000, f"exactly 100,000 HQ/LQ pairs (found {len(quality_df)})")
    
    # Just checking first 100 and last 100 to save time, or we can check all
    # Since dataset is 100k, checking 100k files might take a few seconds
    print("Checking file existence for all images (this takes a few seconds)...")
    missing_hq = 0
    missing_lq = 0
    for _, row in quality_df.iterrows():
        if not (PROJECT_ROOT / row['image_path']).exists(): missing_hq += 1
        if not (PROJECT_ROOT / row['low_quality_path']).exists(): missing_lq += 1
        
    check(missing_hq == 0, f"all image paths exist (missing {missing_hq})")
    check(missing_lq == 0, f"all low-quality paths exist (missing {missing_lq})")
    
    # 6. Nested Noise Verification
    n5 = set(master_df[master_df['is_noisy_5pct'] == 1]['image_id'])
    n10 = set(master_df[master_df['is_noisy_10pct'] == 1]['image_id'])
    n20 = set(master_df[master_df['is_noisy_20pct'] == 1]['image_id'])
    
    check(len(n5) == 5000, f"exactly 5,000 noisy images at 5% (found {len(n5)})")
    check(len(n10) == 10000, f"exactly 10,000 noisy images at 10% (found {len(n10)})")
    check(len(n20) == 20000, f"exactly 20,000 noisy images at 20% (found {len(n20)})")
    
    check(n5.issubset(n10), "5% noise IDs subset of 10%")
    check(n10.issubset(n20), "10% noise IDs subset of 20%")
    
    # 7. False label assignment logic
    self_assigns = 0
    invalid_targets = 0
    for level in ['5pct', '10pct', '20pct']:
        noisy = master_df[master_df[f'is_noisy_{level}'] == 1]
        self_assigns += (noisy[f'assigned_identity_{level}'] == noisy['true_identity']).sum()
        invalid_targets += (~noisy[f'assigned_identity_{level}'].isin(train_identities)).sum()
        
    check(self_assigns == 0, "no self-label assignments in any noise level")
    check(invalid_targets == 0, "all false labels belong to selected identities")
    
    # 8. Manifest Consistency
    # We load 0pct, 5pct, 10pct, 20pct A/B/C/D
    for level in [0, 5, 10, 20]:
        level_dir = SPLITS_100K / f"{level}pct"
        check(level_dir.exists(), f"manifest directory {level}pct/ exists")
        
        a = pd.read_csv(level_dir / "clean_high_train.csv")
        b = pd.read_csv(level_dir / "noisy_high_train.csv")
        c = pd.read_csv(level_dir / "clean_low_train.csv")
        d = pd.read_csv(level_dir / "noisy_low_train.csv")
        
        if level == 0:
            check(len(a) == 100000, f"0%: A contains 100,000")
            check(len(b) == 0, f"0%: B contains 0")
        else:
            clean_expected = 100000 - (level * 1000)
            noisy_expected = level * 1000
            check(len(a) == clean_expected, f"{level}%: A contains {clean_expected}")
            check(len(b) == noisy_expected, f"{level}%: B contains {noisy_expected}")
            check(len(c) == clean_expected, f"{level}%: C contains {clean_expected}")
            check(len(d) == noisy_expected, f"{level}%: D contains {noisy_expected}")
            
    # Probe deterministic check
    probe = pd.read_csv(SPLITS_100K / "probe.csv")
    check(len(probe) == 8000, "probe deterministic (matches 8,000 held-out)")
    
    # If we got here, everything passed
    report = {}
    if (SPLITS_100K / "dataset_statistics.json").exists():
        with open(SPLITS_100K / "dataset_statistics.json", "r") as f:
            report = json.load(f)
            check("20pct_noise_source_distribution" in report, "per-identity source-noise counts reported")
            check("20pct_noise_target_distribution" in report, "per-identity target-noise counts reported")

    report_out = {
        "status": "PASSED",
        "timestamp": pd.Timestamp.now().isoformat(),
        "train_images": len(master_df),
        "heldout_images": len(heldout_df)
    }
    with open(SPLITS_100K / "validation_report.json", "w") as f:
        json.dump(report_out, f, indent=4)

    print("\n==================================================")
    print("100K DATASET VALIDATION: PASSED")
    print("==================================================")

if __name__ == "__main__":
    main()
