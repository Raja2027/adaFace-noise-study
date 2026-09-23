#!/usr/bin/env python3
"""
Generate the 100k controlled AdaFace label-noise study dataset.
Extracts 100,000 training images + 8,000 held-out images.
Applies Gaussian blur for low-quality conditions.
Assigns nested, closed-set label noise (0%, 5%, 10%, 20%).
"""

import sys
import os
import json
import struct
import datetime
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SEED = 42
TARGET_IDENTITIES = 2000
TRAIN_PER_ID = 50
HELDOUT_PER_ID = 4
TOTAL_TRAIN = TARGET_IDENTITIES * TRAIN_PER_ID
TOTAL_HELDOUT = TARGET_IDENTITIES * HELDOUT_PER_ID

NOISE_COUNTS = {
    '5pct': 5000,
    '10pct': 10000,
    '20pct': 20000
}

BLUR_KERNEL = (15, 15)
BLUR_SIGMA = 5.0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "faces_webface_112x112"
REC_PATH = DATA_RAW / "train.rec"
IDX_PATH = DATA_RAW / "train.idx"
LABEL_MAP_PATH = PROJECT_ROOT / "data" / "metadata" / "record_label_map.csv"

SPLITS_100K = PROJECT_ROOT / "data" / "splits" / "100k"
CORRUPTED_DIR = PROJECT_ROOT / "data" / "corrupted" / "100k_blur_15x15_s5"

HQ_TRAIN_DIR = SPLITS_100K / "images" / "train"
HQ_HELDOUT_DIR = SPLITS_100K / "images" / "heldout"
LQ_TRAIN_DIR = CORRUPTED_DIR / "train"
LQ_HELDOUT_DIR = CORRUPTED_DIR / "heldout"

# RecordIO Constants
RECORDIO_MAGIC = 0xced7230a
IR_FORMAT = "IfQQ"
IR_SIZE = struct.calcsize(IR_FORMAT)


def set_deterministic():
    np.random.seed(SEED)
    import random
    random.seed(SEED)


def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT).decode('ascii').strip()
    except Exception:
        return "Not available"


def read_idx_file(idx_path: Path) -> dict:
    idx = {}
    with open(idx_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                idx[int(parts[0])] = int(parts[1])
    return idx


def read_record_header(rec_file, offset: int):
    rec_file.seek(offset)
    magic, lrecord = struct.unpack("<II", rec_file.read(8))
    assert magic == RECORDIO_MAGIC, f"Bad magic at offset {offset}: 0x{magic:08x}"
    length = lrecord & ((1 << 29) - 1)
    header_data = rec_file.read(IR_SIZE)
    flag, label_val, record_id, id2 = struct.unpack(IR_FORMAT, header_data)

    if flag > 0:
        label_arr = np.frombuffer(rec_file.read(flag * 4), dtype=np.float32).copy()
        data_offset = offset + 8 + IR_SIZE + flag * 4
        data_length = length - IR_SIZE - flag * 4
        return flag, label_arr, record_id, data_offset, data_length
    else:
        data_offset = offset + 8 + IR_SIZE
        data_length = length - IR_SIZE
        return flag, label_val, record_id, data_offset, data_length


def decode_image(rec_file, data_offset: int, data_length: int) -> np.ndarray:
    rec_file.seek(data_offset)
    img_bytes = rec_file.read(data_length)
    img_array = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        raise ValueError(f"Failed to decode image at offset {data_offset}")
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array


def main():
    set_deterministic()
    print("=" * 70)
    print("GENERATING 100K CONTROLLED DATASET")
    print("=" * 70)

    for d in [HQ_TRAIN_DIR, HQ_HELDOUT_DIR, LQ_TRAIN_DIR, LQ_HELDOUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Identity Selection
    print("\n--- Step 1: Identity Selection ---")
    df_label_map = pd.read_csv(LABEL_MAP_PATH)
    counts = df_label_map['y_true'].value_counts().reset_index()
    counts.columns = ['y_true', 'count']
    
    # Sort by count descending, then identity ascending (deterministic)
    counts = counts.sort_values(['count', 'y_true'], ascending=[False, True]).reset_index(drop=True)
    
    selected_identities = counts.head(TARGET_IDENTITIES)['y_true'].tolist()
    assert len(selected_identities) == TARGET_IDENTITIES
    assert counts.iloc[TARGET_IDENTITIES - 1]['count'] >= (TRAIN_PER_ID + HELDOUT_PER_ID)
    print(f"Selected {TARGET_IDENTITIES} identities.")
    print(f"Min images for selected identities: {counts.iloc[TARGET_IDENTITIES - 1]['count']}")

    # 2. Image Selection
    print("\n--- Step 2: Image Selection ---")
    df_selected = df_label_map[df_label_map['y_true'].isin(selected_identities)].copy()
    
    train_records = []
    heldout_records = []
    
    rng = np.random.RandomState(SEED)
    
    for identity in selected_identities:
        records = df_selected[df_selected['y_true'] == identity]['record_index'].sort_values().values
        sampled = rng.choice(records, size=(TRAIN_PER_ID + HELDOUT_PER_ID), replace=False)
        train_records.extend([(r, identity) for r in sampled[:TRAIN_PER_ID]])
        heldout_records.extend([(r, identity) for r in sampled[TRAIN_PER_ID:]])
        
    assert len(train_records) == TOTAL_TRAIN
    assert len(heldout_records) == TOTAL_HELDOUT
    
    train_df = pd.DataFrame(train_records, columns=['record_index', 'y_true'])
    heldout_df = pd.DataFrame(heldout_records, columns=['record_index', 'y_true'])
    
    # Assign Image IDs
    train_df['image_id'] = [f"T{str(i).zfill(6)}" for i in range(TOTAL_TRAIN)]
    heldout_df['image_id'] = [f"H{str(i).zfill(6)}" for i in range(TOTAL_HELDOUT)]
    
    # 3. Extraction & Blurring
    print("\n--- Step 3: Extracting and Blurring Images ---")
    idx = read_idx_file(IDX_PATH)
    rec_file = open(REC_PATH, "rb")
    
    def process_split(df, hq_dir, lq_dir):
        paths_hq = []
        paths_lq = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing"):
            record_index = row['record_index']
            image_id = row['image_id']
            
            flag, _, _, data_offset, data_length = read_record_header(rec_file, idx[record_index])
            img_array = decode_image(rec_file, data_offset, data_length)
            
            # Save HQ
            hq_rel = f"{image_id}.png"
            hq_abs = hq_dir / hq_rel
            Image.fromarray(img_array).save(hq_abs)
            paths_hq.append(os.path.relpath(hq_abs, PROJECT_ROOT).replace("\\", "/"))
            
            # Apply blur
            lq_array = cv2.GaussianBlur(img_array, BLUR_KERNEL, BLUR_SIGMA)
            lq_rel = f"{image_id}.png"
            lq_abs = lq_dir / lq_rel
            Image.fromarray(lq_array).save(lq_abs)
            paths_lq.append(os.path.relpath(lq_abs, PROJECT_ROOT).replace("\\", "/"))
            
        df['image_path'] = paths_hq
        df['low_quality_path'] = paths_lq
        return df
        
    print("Processing Training Split...")
    train_df = process_split(train_df, HQ_TRAIN_DIR, LQ_TRAIN_DIR)
    print("Processing Held-out Split...")
    heldout_df = process_split(heldout_df, HQ_HELDOUT_DIR, LQ_HELDOUT_DIR)
    rec_file.close()

    # 4. Noise Assignment
    print("\n--- Step 4: Nested Noise Assignment ---")
    
    noise_pool = train_df['image_id'].values.copy()
    rng.shuffle(noise_pool)
    
    n20_ids = noise_pool[:NOISE_COUNTS['20pct']]
    n10_ids = noise_pool[:NOISE_COUNTS['10pct']]
    n5_ids = noise_pool[:NOISE_COUNTS['5pct']]
    
    # Create target assignments for 20% level
    # 20,000 images, 2,000 identities -> 10 assignments per identity
    target_identities_pool = np.repeat(selected_identities, NOISE_COUNTS['20pct'] // TARGET_IDENTITIES)
    rng.shuffle(target_identities_pool)
    
    # Assign target labels, avoiding self-assignment
    # Map from image_id to true_identity
    true_id_map = dict(zip(train_df['image_id'], train_df['y_true']))
    
    assigned_dict_20 = {}
    
    # Simple derangement solver
    for i in range(len(n20_ids)):
        img_id = n20_ids[i]
        true_id = true_id_map[img_id]
        
        # If collision, swap with a random future element or previous element
        if target_identities_pool[i] == true_id:
            swap_idx = (i + 1) % len(n20_ids)
            while target_identities_pool[swap_idx] == true_id or true_id_map[n20_ids[swap_idx]] == target_identities_pool[i]:
                swap_idx = (swap_idx + 1) % len(n20_ids)
            # Swap
            target_identities_pool[i], target_identities_pool[swap_idx] = target_identities_pool[swap_idx], target_identities_pool[i]
            
        assigned_dict_20[img_id] = target_identities_pool[i]
        
    # Verify no self-assignment
    for k, v in assigned_dict_20.items():
        assert v != true_id_map[k], "Self-assignment detected!"

    # Populate master dataframe
    train_df = train_df.rename(columns={'record_index': 'source_record_id', 'y_true': 'true_identity'})
    
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
        
    # Save master manifest
    train_df.to_csv(SPLITS_100K / "master_100k.csv", index=False)
    print(f"Saved master manifest to {SPLITS_100K / 'master_100k.csv'}")

    # 5. Manifest Generation
    print("\n--- Step 5: Generating Noise Level Manifests ---")
    
    for level_pct in [0, 5, 10, 20]:
        level_dir = SPLITS_100K / f"{level_pct}pct"
        level_dir.mkdir(parents=True, exist_ok=True)
        level_str = f"{level_pct}pct"
        
        # A = Clean HQ
        # B = Noisy HQ
        # C = Clean LQ
        # D = Noisy LQ
        
        assigned_col = f"assigned_identity_{level_str}"
        noisy_col = f"is_noisy_{level_str}"
        
        clean_mask = train_df[noisy_col] == 0
        noisy_mask = train_df[noisy_col] == 1
        
        def create_manifest(df_sub, is_hq, is_noisy_cond):
            manifest = df_sub[['image_id', 'source_record_id', 'true_identity']].copy()
            manifest['image_path'] = df_sub['image_path'] if is_hq else df_sub['low_quality_path']
            manifest['low_quality_path'] = df_sub['low_quality_path']
            manifest['assigned_identity'] = df_sub[assigned_col]
            manifest['quality_condition'] = "HIGH" if is_hq else "LOW"
            manifest['noise_level'] = f"{level_pct}%"
            manifest['is_noisy'] = 1 if is_noisy_cond else 0
            return manifest
            
        # A: clean high
        manifest_a = create_manifest(train_df[clean_mask], True, False)
        manifest_a.to_csv(level_dir / "clean_high_train.csv", index=False)
        
        # C: clean low
        manifest_c = create_manifest(train_df[clean_mask], False, False)
        manifest_c.to_csv(level_dir / "clean_low_train.csv", index=False)
        
        if level_pct > 0:
            # B: noisy high
            manifest_b = create_manifest(train_df[noisy_mask], True, True)
            manifest_b.to_csv(level_dir / "noisy_high_train.csv", index=False)
            
            # D: noisy low
            manifest_d = create_manifest(train_df[noisy_mask], False, True)
            manifest_d.to_csv(level_dir / "noisy_low_train.csv", index=False)
            
            # Explicit noise map
            noise_map = train_df[noisy_mask][['image_id', 'true_identity', assigned_col]].copy()
            noise_map = noise_map.rename(columns={assigned_col: 'assigned_identity'})
            noise_map['noise_level'] = f"{level_pct}%"
            noise_map['seed'] = SEED
            noise_map.to_csv(SPLITS_100K / f"noise_map_{level_pct}pct.csv", index=False)
        else:
            # Write empty manifests for B/D to keep structure
            pd.DataFrame(columns=manifest_a.columns).to_csv(level_dir / "noisy_high_train.csv", index=False)
            pd.DataFrame(columns=manifest_a.columns).to_csv(level_dir / "noisy_low_train.csv", index=False)

    # 6. Ancillary Files
    # Quality Pairs
    quality_pairs = train_df[['image_id', 'image_path', 'low_quality_path', 'true_identity']]
    quality_pairs.to_csv(SPLITS_100K / "quality_pairs.csv", index=False)
    
    # Heldout / Probe
    heldout_df = heldout_df.rename(columns={'record_index': 'source_record_id', 'y_true': 'true_identity'})
    heldout_df['assigned_identity'] = heldout_df['true_identity']
    heldout_df.to_csv(SPLITS_100K / "heldout.csv", index=False)
    
    # Probe is identical to heldout as requested (4 per ID)
    heldout_df.to_csv(SPLITS_100K / "probe.csv", index=False)
    
    # 7. Reproducibility & Statistics
    selection_config = {
        "seed": SEED,
        "target_identities": TARGET_IDENTITIES,
        "train_images_per_identity": TRAIN_PER_ID,
        "heldout_images_per_identity": HELDOUT_PER_ID,
        "total_train_images": TOTAL_TRAIN,
        "total_heldout_images": TOTAL_HELDOUT,
        "blur_kernel": BLUR_KERNEL,
        "blur_sigma": BLUR_SIGMA,
        "source_dataset": str(REC_PATH),
        "creation_timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit()
    }
    with open(SPLITS_100K / "selection_config.json", "w") as f:
        json.dump(selection_config, f, indent=4)
        
    noise_config = {
        "seed": SEED,
        "noise_counts": NOISE_COUNTS,
        "strategy": "nested closed-set balanced assignment",
        "nested_guaranteed": True
    }
    with open(SPLITS_100K / "noise_config.json", "w") as f:
        json.dump(noise_config, f, indent=4)
        
    # Statistics
    src_noise_counts = {id_: 0 for id_ in selected_identities}
    tgt_noise_counts = {id_: 0 for id_ in selected_identities}
    
    for img_id in n20_ids:
        src = true_id_map[img_id]
        tgt = assigned_dict_20[img_id]
        src_noise_counts[src] += 1
        tgt_noise_counts[tgt] += 1
        
    stats = {
        "num_identities": TARGET_IDENTITIES,
        "train_images": TOTAL_TRAIN,
        "heldout_images": TOTAL_HELDOUT,
        "noise_counts": {
            "0pct": 0,
            "5pct": len(n5_ids),
            "10pct": len(n10_ids),
            "20pct": len(n20_ids)
        },
        "per_identity_train": {
            "min": TRAIN_PER_ID, "max": TRAIN_PER_ID, "mean": TRAIN_PER_ID, "median": TRAIN_PER_ID
        },
        "per_identity_heldout": {
            "min": HELDOUT_PER_ID, "max": HELDOUT_PER_ID, "mean": HELDOUT_PER_ID, "median": HELDOUT_PER_ID
        },
        "20pct_noise_source_distribution": {
            "min": min(src_noise_counts.values()), 
            "max": max(src_noise_counts.values()), 
            "mean": np.mean(list(src_noise_counts.values())),
            "median": np.median(list(src_noise_counts.values()))
        },
        "20pct_noise_target_distribution": {
            "min": min(tgt_noise_counts.values()), 
            "max": max(tgt_noise_counts.values()), 
            "mean": np.mean(list(tgt_noise_counts.values())),
            "median": np.median(list(tgt_noise_counts.values()))
        },
        "target_balancing_exact": (min(tgt_noise_counts.values()) == max(tgt_noise_counts.values()) == 10)
    }
    with open(SPLITS_100K / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("\n--- GENERATION COMPLETE ---")
    print("Files created in data/splits/100k/")

if __name__ == "__main__":
    main()
