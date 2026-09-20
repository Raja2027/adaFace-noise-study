import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(r'c:\Users\Dell\Downloads\research\adaFace-noise-study')
PILOT_100ID_META = PROJECT_ROOT / 'data' / 'splits' / 'pilot_100id' / 'metadata.csv'
PILOT_BALANCED_DIR = PROJECT_ROOT / 'data' / 'splits' / 'pilot_balanced'
PILOT_BALANCED_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SEED = 42
MIN_IMAGES = 30
TRAIN_IMAGES = 25
VAL_IMAGES = 5

def main():
    print("=" * 70)
    print("Creating Balanced Pilot Split")
    print("=" * 70)

    # 1. Load data
    df = pd.read_csv(PILOT_100ID_META)
    df = df.rename(columns={'y_true': 'casia_identity'})
    
    # 2. Retain identities with >= 30 images
    counts = df['casia_identity'].value_counts()
    valid_identities = counts[counts >= MIN_IMAGES].index.sort_values()
    
    df_filtered = df[df['casia_identity'].isin(valid_identities)].copy()
    
    # 3. Create class_index
    identity_map = {casia_id: idx for idx, casia_id in enumerate(valid_identities)}
    df_filtered['class_index'] = df_filtered['casia_identity'].map(identity_map)
    
    # 4. Sample and split
    np.random.seed(SEED)
    
    train_dfs = []
    val_dfs = []
    summary_data = []
    
    for casia_id in valid_identities:
        id_data = df_filtered[df_filtered['casia_identity'] == casia_id]
        
        # Randomly sample exactly 30 images using the seed
        sampled = id_data.sample(n=MIN_IMAGES, random_state=SEED)
        
        # Split into train (25) and val (5)
        train_sampled = sampled.iloc[:TRAIN_IMAGES].copy()
        val_sampled = sampled.iloc[TRAIN_IMAGES:].copy()
        
        train_sampled['split'] = 'train'
        val_sampled['split'] = 'val'
        
        train_dfs.append(train_sampled)
        val_dfs.append(val_sampled)
        
        summary_data.append({
            'casia_identity': casia_id,
            'class_index': identity_map[casia_id],
            'available_images': len(id_data),
            'selected_images': MIN_IMAGES,
            'train_images': len(train_sampled),
            'val_images': len(val_sampled)
        })
        
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = pd.concat(val_dfs, ignore_index=True)
    df_metadata = pd.concat([df_train, df_val], ignore_index=True)
    df_summary = pd.DataFrame(summary_data)
    
    # 5. Order columns
    cols = ['image_id', 'record_index', 'image_path', 'casia_identity', 'class_index', 'split']
    df_train = df_train[cols]
    df_val = df_val[cols]
    df_metadata = df_metadata[cols]
    
    # 6. Save CSVs
    df_train.to_csv(PILOT_BALANCED_DIR / 'train.csv', index=False)
    df_val.to_csv(PILOT_BALANCED_DIR / 'val.csv', index=False)
    df_metadata.to_csv(PILOT_BALANCED_DIR / 'metadata.csv', index=False)
    df_summary.to_csv(PILOT_BALANCED_DIR / 'summary.csv', index=False)
    
    # 7. Verifications
    print("\nVerifying constraints...")
    
    retained_identities = df_metadata['casia_identity'].nunique()
    print(f"Retained identity count = {retained_identities} (Expected: 83)")
    assert retained_identities == 83
    
    images_per_id = df_metadata.groupby('casia_identity').size()
    print(f"Exactly 30 images per retained identity: {all(images_per_id == 30)}")
    assert all(images_per_id == 30)
    
    train_per_id = df_train.groupby('casia_identity').size()
    print(f"Exactly 25 train images per identity: {all(train_per_id == 25)}")
    assert all(train_per_id == 25)
    
    val_per_id = df_val.groupby('casia_identity').size()
    print(f"Exactly 5 validation images per identity: {all(val_per_id == 5)}")
    assert all(val_per_id == 5)
    
    print(f"Train total = {len(df_train)} (Expected: 2075)")
    assert len(df_train) == 2075
    
    print(f"Validation total = {len(df_val)} (Expected: 415)")
    assert len(df_val) == 415
    
    print(f"Total = {len(df_metadata)} (Expected: 2490)")
    assert len(df_metadata) == 2490
    
    min_class = df_metadata['class_index'].min()
    max_class = df_metadata['class_index'].max()
    print(f"class_index range = {min_class}..{max_class} (Expected: 0..82)")
    assert min_class == 0 and max_class == 82
    
    dup_images = df_metadata['image_id'].duplicated().any()
    dup_records = df_metadata['record_index'].duplicated().any()
    print(f"No duplicate image_id: {not dup_images}")
    print(f"No duplicate record_index: {not dup_records}")
    assert not dup_images
    assert not dup_records
    
    train_records = set(df_train['record_index'])
    val_records = set(df_val['record_index'])
    overlap = len(train_records.intersection(val_records))
    print(f"No train/validation overlap: {overlap == 0}")
    assert overlap == 0
    
    print("Verifying all referenced files exist...")
    all_exist = True
    missing_count = 0
    for path_str in df_metadata['image_path']:
        full_path = PROJECT_ROOT / 'data' / 'splits' / 'pilot_100id' / path_str
        if not full_path.exists():
            all_exist = False
            missing_count += 1
    print(f"All referenced files exist: {all_exist} (Missing: {missing_count})")
    assert all_exist
    
    print(f"Selection seed used: {SEED}")
    
    print("\nFiles created in data/splits/pilot_balanced/:")
    print(" - train.csv")
    print(" - val.csv")
    print(" - metadata.csv")
    print(" - summary.csv")
    
if __name__ == '__main__':
    main()
