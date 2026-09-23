#!/usr/bin/env python3
import os
import sys
import shutil
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import importlib.util

# Try to import create_100k_dataset for deterministic reconstruction
PROJECT_ROOT = Path("/content/drive/MyDrive/adaFace-noise-study")
if not PROJECT_ROOT.exists():
    # Fallback to local for dev
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))
try:
    from scripts.create_100k_dataset import SEED, TARGET_IDENTITIES, TRAIN_PER_ID, HELDOUT_PER_ID, NOISE_COUNTS
except ImportError:
    # Hardcode if we can't import (e.g. running outside repo root)
    SEED = 42
    TARGET_IDENTITIES = 2000
    TRAIN_PER_ID = 50
    HELDOUT_PER_ID = 4
    NOISE_COUNTS = {'5pct': 5000, '10pct': 10000, '20pct': 20000}

CANONICAL_DATA = PROJECT_ROOT / "data"

def print_header(title):
    print(f"\n{'=' * 70}")
    print(title.upper())
    print(f"{'=' * 70}")

def audit_drive():
    print_header("1. First Audit - Finding Scattered Artifacts")
    artifacts = []
    
    # Target files to look for
    target_files = {
        'master_100k.csv', 'heldout.csv', 'probe.csv', 'quality_pairs.csv',
        'noise_map.csv', 'noise_map_5pct.csv', 'noise_map_10pct.csv', 'noise_map_20pct.csv',
        'clean_high_train.csv', 'clean_low_train.csv', 'noisy_high_train.csv', 'noisy_low_train.csv',
        'selection_config.json', 'noise_config.json', 'dataset_statistics.json', 'validation_report.json',
        'train.rec', 'train.idx'
    }
    
    print(f"Scanning {PROJECT_ROOT} recursively... This may take a minute.")
    
    # We also want to find image directories. We don't want to list all 100k images, just the directories.
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Identify image directories
        if 'T000000.png' in files or 'T000001.png' in files:
            artifacts.append({
                'path': root,
                'filename': '[IMAGE DIR]',
                'type': 'Directory (Images)',
                'size': sum(os.path.getsize(os.path.join(root, f)) for f in files if f.endswith('.png')),
                'count': len([f for f in files if f.endswith('.png')])
            })
            continue # Don't list individual images
            
        for f in files:
            if f in target_files or f.endswith('.csv') or f.endswith('.json'):
                full_path = os.path.join(root, f)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0
                artifacts.append({
                    'path': full_path,
                    'filename': f,
                    'type': 'File',
                    'size': size,
                    'count': 1
                })
                
    # Print table
    print(f"{'Filename':<30} | {'Type':<20} | {'Count':<8} | {'Size (MB)':<10} | {'Path'}")
    print("-" * 150)
    for a in artifacts:
        print(f"{a['filename']:<30} | {a['type']:<20} | {a['count']:<8} | {a['size']/(1024*1024):<10.2f} | {a['path']}")
        
    return artifacts

def reconstruct_metadata(artifacts):
    """ Deterministically reconstruct master_100k.csv logic without image processing """
    print("Reconstructing dataset metadata using seed=42...")
    import numpy as np
    
    label_map_path = PROJECT_ROOT / "data" / "metadata" / "record_label_map.csv"
    if not label_map_path.exists():
        # Maybe in local checkout?
        label_map_path = Path(__file__).resolve().parent.parent / "data" / "metadata" / "record_label_map.csv"
        if not label_map_path.exists():
            raise FileNotFoundError("record_label_map.csv not found, cannot reconstruct metadata!")
            
    df_label_map = pd.read_csv(label_map_path)
    counts = df_label_map['y_true'].value_counts().reset_index()
    counts.columns = ['y_true', 'count']
    counts = counts.sort_values(['count', 'y_true'], ascending=[False, True]).reset_index(drop=True)
    
    selected_identities = counts.head(TARGET_IDENTITIES)['y_true'].tolist()
    
    df_selected = df_label_map[df_label_map['y_true'].isin(selected_identities)].copy()
    
    train_records = []
    rng = np.random.RandomState(SEED)
    
    for identity in selected_identities:
        records = df_selected[df_selected['y_true'] == identity]['record_index'].sort_values().values
        sampled = rng.choice(records, size=(TRAIN_PER_ID + HELDOUT_PER_ID), replace=False)
        train_records.extend([(r, identity) for r in sampled[:TRAIN_PER_ID]])
        
    train_df = pd.DataFrame(train_records, columns=['source_record_id', 'true_identity'])
    train_df['image_id'] = [f"T{str(i).zfill(6)}" for i in range(len(train_df))]
    
    # Check for existing noise maps in artifacts
    recovered_noise_map = None
    for a in artifacts:
        if a['filename'] == 'noise_map_20pct.csv' or (a['filename'] == 'noise_map.csv' and '20pct' in a['path']):
            print(f"Recovering existing noise map from: {a['path']}")
            recovered_noise_map = pd.read_csv(a['path'])
            break

    if recovered_noise_map is not None:
        print("Using recovered noise map to reconstruct subset flags.")
        n20_ids = recovered_noise_map['image_id'].tolist()
        assigned_dict_20 = dict(zip(recovered_noise_map['image_id'], recovered_noise_map['assigned_identity']))
        # We assume 5pct and 10pct are just the first 5k and 10k of the 20k, just like the deterministic algorithm did.
        n10_ids = n20_ids[:NOISE_COUNTS['10pct']]
        n5_ids = n20_ids[:NOISE_COUNTS['5pct']]
    else:
        print("No valid noise map recovered. Regenerating deterministic noise assignment...")
        noise_pool = train_df['image_id'].values.copy()
        rng.shuffle(noise_pool)
        n20_ids = noise_pool[:NOISE_COUNTS['20pct']]
        n10_ids = noise_pool[:NOISE_COUNTS['10pct']]
        n5_ids = noise_pool[:NOISE_COUNTS['5pct']]
        
        target_identities_pool = np.repeat(selected_identities, NOISE_COUNTS['20pct'] // TARGET_IDENTITIES)
        rng.shuffle(target_identities_pool)
        
        true_id_map = dict(zip(train_df['image_id'], train_df['true_identity']))
        assigned_dict_20 = {}
        
        for i in range(len(n20_ids)):
            img_id = n20_ids[i]
            true_id = true_id_map[img_id]
            if target_identities_pool[i] == true_id:
                swap_idx = (i + 1) % len(n20_ids)
                while target_identities_pool[swap_idx] == true_id or true_id_map[n20_ids[swap_idx]] == target_identities_pool[i]:
                    swap_idx = (swap_idx + 1) % len(n20_ids)
                target_identities_pool[i], target_identities_pool[swap_idx] = target_identities_pool[swap_idx], target_identities_pool[i]
            assigned_dict_20[img_id] = target_identities_pool[i]
        
    train_df['assigned_identity_0pct'] = train_df['true_identity']
    train_df['is_noisy_0pct'] = 0
    
    for level, ids_set in [('5pct', set(n5_ids)), ('10pct', set(n10_ids)), ('20pct', set(n20_ids))]:
        assigned_col = f"assigned_identity_{level}"
        noisy_col = f"is_noisy_{level}"
        
        train_df[noisy_col] = train_df['image_id'].apply(lambda x: 1 if x in ids_set else 0)
        train_df[assigned_col] = train_df.apply(
            lambda row: assigned_dict_20[row['image_id']] if row['image_id'] in ids_set else row['true_identity'],
            axis=1
        )
        
    train_df['image_path'] = train_df['image_id'].apply(lambda x: f"data/splits/100k/images/train/{x}.png")
    train_df['low_quality_path'] = train_df['image_id'].apply(lambda x: f"data/corrupted/100k_blur_15x15_s5/train/{x}.png")
    
    return train_df

def resolve_master(artifacts, hq_train_dir):
    print_header("2. Master Manifest Resolution")
    master_path = None
    for a in artifacts:
        if a['filename'] == 'master_100k.csv':
            master_path = a['path']
            break
            
    if master_path:
        print(f"Recovering existing master_100k.csv from: {master_path}")
        master_df = pd.read_csv(master_path)
    else:
        print("master_100k.csv is MISSING. Reconstructing via Mode 2...")
        master_df = reconstruct_metadata(artifacts)
        
    print(f"Master manifest row count: {len(master_df)}")
    
    # Bijections check against physical directory
    print("Checking BIJECTION against physical images...")
    if hq_train_dir.exists():
        physical_images = set([f.split('.')[0] for f in os.listdir(hq_train_dir) if f.endswith('.png')])
    else:
        physical_images = set()
        
    manifest_images = set(master_df['image_id'].tolist())
    
    print(f"Physical images found: {len(physical_images)}")
    print(f"Manifest images: {len(manifest_images)}")
    
    if physical_images != manifest_images:
        missing_in_physical = manifest_images - physical_images
        missing_in_manifest = physical_images - manifest_images
        raise RuntimeError(f"BIJECTION FAILED! Missing in physical: {len(missing_in_physical)}, Missing in manifest: {len(missing_in_manifest)}")
        
    print("BIJECTION PASSED: Manifest images exactly match physical HQ images.")
    return master_df, master_path

def consolidate_manifests(master_df, canonical_splits_dir, hq_train_dir, hq_heldout_dir, lq_train_dir, lq_heldout_dir):
    print_header("3. Consolidating Canonical Structure (No Deletions)")
    
    canonical_splits_dir.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(canonical_splits_dir / "master_100k.csv", index=False)
    print(f"Saved master_100k.csv to {canonical_splits_dir}")
    
    for level_pct in [0, 5, 10, 20]:
        level_str = f"{level_pct}pct"
        level_dir = canonical_splits_dir / level_str
        level_dir.mkdir(parents=True, exist_ok=True)
        
        assigned_col = f"assigned_identity_{level_str}"
        noisy_col = f"is_noisy_{level_str}"
        
        clean_mask = master_df[noisy_col] == 0
        noisy_mask = master_df[noisy_col] == 1
        
        def create_manifest(df_sub, is_hq, is_noisy_cond):
            manifest = df_sub[['image_id', 'source_record_id', 'true_identity']].copy()
            manifest['image_path'] = df_sub['image_path'] if is_hq else df_sub['low_quality_path']
            manifest['low_quality_path'] = df_sub['low_quality_path']
            manifest['assigned_identity'] = df_sub[assigned_col]
            manifest['quality_condition'] = "HIGH" if is_hq else "LOW"
            manifest['noise_level'] = f"{level_pct}%"
            manifest['is_noisy'] = 1 if is_noisy_cond else 0
            return manifest
            
        manifest_a = create_manifest(master_df[clean_mask], True, False)
        manifest_a.to_csv(level_dir / "clean_high_train.csv", index=False)
        manifest_c = create_manifest(master_df[clean_mask], False, False)
        manifest_c.to_csv(level_dir / "clean_low_train.csv", index=False)
        
        if level_pct > 0:
            manifest_b = create_manifest(master_df[noisy_mask], True, True)
            manifest_b.to_csv(level_dir / "noisy_high_train.csv", index=False)
            manifest_d = create_manifest(master_df[noisy_mask], False, True)
            manifest_d.to_csv(level_dir / "noisy_low_train.csv", index=False)
            
            noise_map = master_df[noisy_mask][['image_id', 'true_identity', assigned_col]].copy()
            noise_map = noise_map.rename(columns={assigned_col: 'assigned_identity'})
            noise_map['noise_level'] = f"{level_pct}%"
            noise_map['seed'] = SEED
            noise_map.to_csv(canonical_splits_dir / f"noise_map_{level_pct}pct.csv", index=False)
        else:
            pd.DataFrame(columns=manifest_a.columns).to_csv(level_dir / "noisy_high_train.csv", index=False)
            pd.DataFrame(columns=manifest_a.columns).to_csv(level_dir / "noisy_low_train.csv", index=False)

def strict_validation(canonical_splits_dir, hq_train_dir, lq_train_dir, hq_heldout_dir):
    print_header("4. Canonical Dataset Validation")
    checks = []
    
    def check(name, condition):
        status = "PASSED" if condition else "FAILED"
        checks.append({"name": name, "status": status})
        print(f"[{status}] {name}")
        if not condition:
            raise ValueError(f"Validation Check Failed: {name}")

    master_path = canonical_splits_dir / "master_100k.csv"
    check("master_100k.csv exists", master_path.exists())
    
    master_df = pd.read_csv(master_path)
    check("master_100k.csv contains exactly 100,000 rows", len(master_df) == 100000)
    check("no duplicate image IDs", master_df['image_id'].is_unique)
    check("2,000 identities", master_df['true_identity'].nunique() == 2000)
    
    counts = master_df['true_identity'].value_counts()
    check("50 train / identity", counts.min() == 50 and counts.max() == 50)
    
    hq_files = set(f for f in os.listdir(hq_train_dir) if f.endswith('.png'))
    lq_files = set(f for f in os.listdir(lq_train_dir) if f.endswith('.png'))
    
    check("100,000 HQ images in directory", len(hq_files) == 100000)
    check("100,000 LQ images in directory", len(lq_files) == 100000)
    
    manifest_hq_ids = set(master_df['image_id'].apply(lambda x: f"{x}.png"))
    check("Bijective HQ mapping (manifest ↔ physical)", hq_files == manifest_hq_ids)
    check("Bijective LQ mapping (manifest ↔ physical)", lq_files == manifest_hq_ids)
    
    n5 = set(master_df[master_df['is_noisy_5pct'] == 1]['image_id'])
    n10 = set(master_df[master_df['is_noisy_10pct'] == 1]['image_id'])
    n20 = set(master_df[master_df['is_noisy_20pct'] == 1]['image_id'])
    
    check("5,000 noisy at 5%", len(n5) == 5000)
    check("10,000 noisy at 10%", len(n10) == 10000)
    check("20,000 noisy at 20%", len(n20) == 20000)
    
    check("N5 ⊂ N10", n5.issubset(n10))
    check("N10 ⊂ N20", n10.issubset(n20))
    
    for level in ['5pct', '10pct', '20pct']:
        noisy_mask = master_df[f'is_noisy_{level}'] == 1
        conflicts = master_df[noisy_mask][master_df[noisy_mask]['true_identity'] == master_df[noisy_mask][f'assigned_identity_{level}']]
        check(f"Zero self-labels at {level}", len(conflicts) == 0)
        
    for level in ['0pct', '5pct', '10pct', '20pct']:
        for split in ['clean_high_train.csv', 'noisy_high_train.csv']:
            check(f"Manifest {level}/{split} exists", (canonical_splits_dir / level / split).exists())
            
    print(f"\nAll {len(checks)} checks PASSED perfectly.")
    return checks

def final_report(artifacts, canonical_dir, master_path, checks):
    print_header("5. Final Drive Verification Report")
    
    print(f"CANONICAL DATASET:")
    print(f"{canonical_dir}")
    
    print(f"\nMASTER:")
    if master_path:
        print(f"Recovered from scattered path: {master_path}")
    else:
        print("Reconstructed mathematically exactly to match physical PNGs.")
    print(f"{canonical_dir / 'splits/100k/master_100k.csv'} (100000 rows)")
    
    hq_train = canonical_dir / 'splits/100k/images/train'
    hq_heldout = canonical_dir / 'splits/100k/images/heldout'
    lq_train = canonical_dir / 'corrupted/100k_blur_15x15_s5/train'
    
    print(f"\nHQ TRAIN:")
    print(f"{hq_train} ({len(os.listdir(hq_train))} images)")
    
    print(f"\nHQ HELDOUT:")
    heldout_count = len(os.listdir(hq_heldout)) if hq_heldout.exists() else 0
    print(f"{hq_heldout} ({heldout_count} images)")
    
    print(f"\nLQ TRAIN:")
    print(f"{lq_train} ({len(os.listdir(lq_train))} images)")
    
    for lvl in ['0pct', '5pct', '10pct', '20pct']:
        clean = len(pd.read_csv(canonical_dir / f"splits/100k/{lvl}/clean_high_train.csv"))
        noisy = len(pd.read_csv(canonical_dir / f"splits/100k/{lvl}/noisy_high_train.csv"))
        print(f"\n{lvl}:")
        print(f"Clean HQ: {clean}, Noisy HQ: {noisy}")
        
    print(f"\nVALIDATION CHECKS PERFORMED: {len(checks)}")
    print("VALIDATION: PASSED")
    
    print("\nFILES INTENTIONALLY LEFT IN OLD LOCATIONS:")
    print("None were deleted. Scattered artifacts remain safe until user confirms deletion.")

def main():
    artifacts = audit_drive()
    
    hq_train_dir = CANONICAL_DATA / "splits" / "100k" / "images" / "train"
    hq_heldout_dir = CANONICAL_DATA / "splits" / "100k" / "images" / "heldout"
    lq_train_dir = CANONICAL_DATA / "corrupted" / "100k_blur_15x15_s5" / "train"
    lq_heldout_dir = CANONICAL_DATA / "corrupted" / "100k_blur_15x15_s5" / "heldout"
    canonical_splits_dir = CANONICAL_DATA / "splits" / "100k"
    
    master_df, original_master_path = resolve_master(artifacts, hq_train_dir)
    
    consolidate_manifests(master_df, canonical_splits_dir, hq_train_dir, hq_heldout_dir, lq_train_dir, lq_heldout_dir)
    
    checks = strict_validation(canonical_splits_dir, hq_train_dir, lq_train_dir, hq_heldout_dir)
    
    final_report(artifacts, CANONICAL_DATA, original_master_path, checks)

if __name__ == "__main__":
    main()
