import struct
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path

RECORDIO_MAGIC = 0xced7230a
IR_FORMAT = "IfQQ"
IR_SIZE = struct.calcsize(IR_FORMAT)

class RecordDataset(Dataset):
    def __init__(self, rec_path, idx_path, label_map_path, transform=None, noise_map=None):
        """
        Args:
            rec_path: Path to train.rec
            idx_path: Path to train.idx
            label_map_path: Path to record_label_map.csv
            transform: PyTorch transforms
            noise_map: Optional dict mapping `record_id` -> `assigned_label`.
                       If not provided, assigned_label == true_label.
        """
        self.rec_path = str(rec_path)
        self.transform = transform
        self.noise_map = noise_map or {}
        
        # Load the byte offsets
        offsets = {}
        with open(idx_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    offsets[int(parts[0])] = int(parts[1])
        
        # Load authoritative labels
        df = pd.read_csv(label_map_path)
        
        # Precompute compact list: (record_id, byte_offset, true_label)
        self.records = []
        for row in df.itertuples():
            record_id = row.record_index
            true_label = row.y_true
            if record_id in offsets:
                self.records.append((record_id, offsets[record_id], true_label))
        
        # We don't open the file in __init__ because Dataset workers need their own file handles
        self.rec_file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['rec_file'] = None
        return state
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def __len__(self):
        return len(self.records)

    def _open_rec(self):
        if self.rec_file is None:
            self.rec_file = open(self.rec_path, "rb")

    def __getitem__(self, idx):
        self._open_rec()
        record_id, offset, true_label = self.records[idx]
        
        self.rec_file.seek(offset)
        magic, lrecord = struct.unpack("<II", self.rec_file.read(8))
        if magic != RECORDIO_MAGIC:
            raise ValueError(f"Bad magic at offset {offset}: 0x{magic:08x}")
            
        length = lrecord & ((1 << 29) - 1)
        header_data = self.rec_file.read(IR_SIZE)
        flag, label_val, _, _ = struct.unpack(IR_FORMAT, header_data)
        
        if flag > 0:
            raise ValueError(f"Extended label records not supported in __getitem__. Key: {record_id}")
            
        # Verify true label matches authoritative map (optional sanity check)
        assert true_label == int(label_val), f"Label mismatch for {record_id}!"
        
        data_offset = offset + 8 + IR_SIZE
        data_length = length - IR_SIZE
        
        self.rec_file.seek(data_offset)
        img_bytes = self.rec_file.read(data_length)
        
        # Decode JPEG bytes → numpy array (OpenCV decodes to BGR)
        img_array = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError(f"Failed to decode image at offset {data_offset}")
            
        # Convert BGR -> RGB (Wait, the AdaFace standard is to feed BGR!)
        # The pilot dataset did:
        # img_bgr = np.array(img.convert('RGB'))[:, :, ::-1] 
        # OpenCV already decodes to BGR.
        
        tensor = torch.from_numpy(img_array).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5
        
        if self.transform:
            tensor = self.transform(tensor)
            
        assigned_label = self.noise_map.get(record_id, true_label)
        
        return tensor, torch.tensor(true_label, dtype=torch.long), record_id, torch.tensor(assigned_label, dtype=torch.long)
