#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

PROBE_CSV = Path("data/splits/100k/probe.csv")
MASTER_CSV = Path("data/splits/100k/master_100k.csv")
OUT_CSV = Path("data/splits/100k/probe_conditions.csv")

def main():
    probe_df = pd.read_csv(PROBE_CSV)
    master_df = pd.read_csv(MASTER_CSV)
    
    identities = master_df['true_identity'].unique()
    assert len(identities) == 2000
    
    target_identities_pool = np.repeat(identities, len(probe_df) // len(identities) + 1)[:len(probe_df)]
    np.random.shuffle(target_identities_pool)
    
    probe_ids = probe_df['image_id'].values
    true_ids = probe_df['true_identity'].values
    
    assigned = []
    
    for i in range(len(probe_ids)):
        true_id = true_ids[i]
        
        if target_identities_pool[i] == true_id:
            swap_idx = (i + 1) % len(probe_ids)
            while target_identities_pool[swap_idx] == true_id or true_ids[swap_idx] == target_identities_pool[i]:
                swap_idx = (swap_idx + 1) % len(probe_ids)
            target_identities_pool[i], target_identities_pool[swap_idx] = target_identities_pool[swap_idx], target_identities_pool[i]
            
        assigned.append(target_identities_pool[i])
        
    probe_df['false_identity'] = assigned
    
    # Verify
    assert (probe_df['true_identity'] == probe_df['false_identity']).sum() == 0
    assert probe_df['false_identity'].isin(identities).all()
    
    probe_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

if __name__ == "__main__":
    main()
