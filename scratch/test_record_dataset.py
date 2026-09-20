import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.record_dataset import RecordDataset

def main():
    rec_path = PROJECT_ROOT / "data/raw/faces_webface_112x112/train.rec"
    idx_path = PROJECT_ROOT / "data/raw/faces_webface_112x112/train.idx"
    label_map_path = PROJECT_ROOT / "data/metadata/record_label_map.csv"
    
    # We will test noise override for a few sample IDs just to be sure it works
    noise_map = {
        10: 99999,
        20: 88888
    }
    
    dataset = RecordDataset(rec_path, idx_path, label_map_path, noise_map=noise_map)
    print(f"Dataset length: {len(dataset)}")
    
    # Verify random access & decoding
    for idx in [0, 100, 1000]:
        tensor, true_label, record_id, assigned_label = dataset[idx]
        print(f"Sample {idx}: record_id={record_id}, true={true_label}, assigned={assigned_label}, shape={tensor.shape}, min={tensor.min():.2f}, max={tensor.max():.2f}")
        
    # Verify noise map
    # Find indices for keys 10 and 20
    for idx, key in enumerate(dataset.keys):
        if key in noise_map:
            tensor, true_label, record_id, assigned_label = dataset[idx]
            print(f"\nNoise Override Test - record_id={record_id}:")
            print(f"True Label: {true_label}")
            print(f"Assigned Label: {assigned_label} (Expected: {noise_map[key]})")
            assert assigned_label.item() == noise_map[key], "Noise map failed!"
    
    # Verify DataLoader multiprocessing
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    print("\nTesting DataLoader Multiprocessing...")
    for i, (images, true_labels, ids, assigned) in enumerate(loader):
        print(f"Batch {i}: {images.shape}, {true_labels.shape}")
        if i == 1:
            break
            
    print("RecordDataset dry-run completed successfully!")

if __name__ == "__main__":
    main()
