#!/usr/bin/env python3
"""
Extract a pilot subset from CASIA-WebFace RecordIO archive.

Replicates the RecordIO reading logic from:
  - AdaFace/dataset/record_dataset.py  (BaseMXDataset.__init__, read_sample)
  - AdaFace/convert.py                 (save_rec_to_img_dir)

WITHOUT importing or modifying any AdaFace source code.
WITHOUT requiring MXNet (parses the binary RecordIO format directly).

The MXNet RecordIO format:
  - .idx file: text-based, each line is "key\toffset\n"
  - .rec file: sequence of records, each is:
      magic(4 LE) + lrecord(4 LE) + data(variable) + padding(0-3 bytes)
    where magic = 0xced7230a
          length = lrecord & ((1<<29)-1)
          cflag  = (lrecord >> 29) & 7
  - Each record's data starts with an IRHeader:
      struct 'IfQQ' = flag(uint32) + label(float32) + id(uint64) + id2(uint64)
    If flag > 0, the label field is ignored and replaced by `flag` float32
    values immediately following the IRHeader.
  - For image records (flag=0): label is the identity ID, remaining bytes are
    JPEG-encoded image data.
  - Key 0 is a special header record with flag=2 and extended labels
    [num_images, num_total_keys].

Usage:
    python data/splits/extract_pilot.py

Output:
    data/metadata/record_label_map.csv    (full dataset: record_index -> y_true)
    data/splits/pilot_100id/metadata.csv  (pilot subset metadata)
    data/splits/pilot_100id/images/       (extracted PNG images)
"""

import struct
import os
import sys
import random
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PILOT_LABELS = set(range(0, 100))  # Identity labels 0-99
IMAGE_FORMAT = "png"               # Lossless — no second compression step

# Paths (relative to project root)
SCRIPT_DIR = Path(__file__).resolve().parent          # data/splits/
PROJECT_ROOT = SCRIPT_DIR.parent.parent               # adaFace-noise-study/
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "faces_webface_112x112"
REC_PATH = DATA_RAW / "train.rec"
IDX_PATH = DATA_RAW / "train.idx"

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
PILOT_DIR = SCRIPT_DIR / "pilot_100id"
PILOT_IMAGES_DIR = PILOT_DIR / "images"
PILOT_METADATA_PATH = PILOT_DIR / "metadata.csv"
FULL_LABEL_MAP_PATH = METADATA_DIR / "record_label_map.csv"


# ---------------------------------------------------------------------------
# RecordIO binary parsing (replicates MXNet mx.recordio logic)
# ---------------------------------------------------------------------------
RECORDIO_MAGIC = 0xced7230a
IR_FORMAT = "IfQQ"  # flag(I=uint32), label(f=float32), id(Q=uint64), id2(Q=uint64)
IR_SIZE = struct.calcsize(IR_FORMAT)  # 24 bytes


def read_idx_file(idx_path: Path) -> dict:
    """Parse the text-based .idx file into {key: offset} dict.

    The .idx format is one "key\\toffset\\n" per line, both integers.
    This is the same format MXIndexedRecordIO reads.
    """
    idx = {}
    with open(idx_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                idx[int(parts[0])] = int(parts[1])
    return idx


def read_record_header(rec_file, offset: int):
    """Read a single RecordIO record header at the given byte offset.

    Returns:
        (flag, label_or_labels, record_id, data_offset, data_length)
        - flag: uint32, 0 for image records, >0 for extended-label records
        - label_or_labels: float (if flag==0) or np.array of floats (if flag>0)
        - record_id: uint64 from IRHeader
        - data_offset: byte offset where image data starts in the .rec file
        - data_length: number of image data bytes
    """
    rec_file.seek(offset)
    magic, lrecord = struct.unpack("<II", rec_file.read(8))
    assert magic == RECORDIO_MAGIC, f"Bad magic at offset {offset}: 0x{magic:08x}"

    length = lrecord & ((1 << 29) - 1)
    # cflag = (lrecord >> 29) & 7  # not needed for reading

    header_data = rec_file.read(IR_SIZE)
    flag, label_val, record_id, id2 = struct.unpack(IR_FORMAT, header_data)

    if flag > 0:
        # Extended label: `flag` float32 values follow the fixed header
        label_arr = np.frombuffer(rec_file.read(flag * 4), dtype=np.float32).copy()
        data_offset = offset + 8 + IR_SIZE + flag * 4
        data_length = length - IR_SIZE - flag * 4
        return flag, label_arr, record_id, data_offset, data_length
    else:
        data_offset = offset + 8 + IR_SIZE
        data_length = length - IR_SIZE
        return flag, label_val, record_id, data_offset, data_length


def decode_image(rec_file, data_offset: int, data_length: int) -> np.ndarray:
    """Read and decode JPEG image bytes from the .rec file.

    Returns an RGB numpy array (H, W, 3).
    """
    rec_file.seek(data_offset)
    img_bytes = rec_file.read(data_length)
    # Decode JPEG bytes → numpy array (OpenCV decodes to BGR)
    img_array = cv2.imdecode(
        np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if img_array is None:
        raise ValueError(f"Failed to decode image at offset {data_offset}")
    # Convert BGR → RGB to match what AdaFace convert.py does
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CASIA-WebFace Pilot Subset Extraction")
    print("=" * 70)

    # Verify source files exist
    for p in [REC_PATH, IDX_PATH]:
        if not p.exists():
            print(f"ERROR: Required file not found: {p}")
            sys.exit(1)
    print(f"[OK] Source: {DATA_RAW}")

    # ------------------------------------------------------------------
    # Step 1: Read the .idx file
    # ------------------------------------------------------------------
    print("\n--- Step 1: Reading .idx file ---")
    idx = read_idx_file(IDX_PATH)
    print(f"  Total keys in .idx: {len(idx)}")
    print(f"  Key range: {min(idx.keys())} – {max(idx.keys())}")

    # ------------------------------------------------------------------
    # Step 2: Read the header record (key 0) to determine record count
    # ------------------------------------------------------------------
    print("\n--- Step 2: Reading header record (key 0) ---")
    rec_file = open(REC_PATH, "rb")

    flag, labels, rec_id, _, _ = read_record_header(rec_file, idx[0])
    assert flag > 0, f"Expected header record to have flag > 0, got {flag}"
    num_images = int(labels[0])
    print(f"  Header flag: {flag}")
    print(f"  Header labels: {labels}")
    print(f"  Number of image records: {num_images}")
    print(f"  Image record keys: 1 – {num_images - 1}")

    # ------------------------------------------------------------------
    # Step 3 (Pass 1): Scan ALL record headers → full label map
    # ------------------------------------------------------------------
    print(f"\n--- Step 3 (Pass 1): Scanning all {num_images - 1} record headers ---")
    all_records = []  # (record_index, y_true)
    pilot_record_keys = []  # keys with label in PILOT_LABELS

    t0 = time.time()
    for key in tqdm(range(1, num_images), desc="Scanning headers", unit="rec"):
        if key not in idx:
            continue
        flag, label_val, rec_id, data_offset, data_length = read_record_header(
            rec_file, idx[key]
        )
        # For image records, flag == 0 and label_val is a float
        if flag == 0:
            y_true = int(label_val)
        else:
            # Should not happen for image records, but handle gracefully
            y_true = int(label_val[0]) if hasattr(label_val, '__len__') else int(label_val)

        all_records.append((key, y_true))

        if y_true in PILOT_LABELS:
            pilot_record_keys.append((key, y_true, data_offset, data_length))

    elapsed = time.time() - t0
    print(f"  Scanned {len(all_records)} image records in {elapsed:.1f}s")
    print(f"  Pilot records (labels 0–99): {len(pilot_record_keys)}")

    # ------------------------------------------------------------------
    # Step 4: Save full label map → data/metadata/record_label_map.csv
    # ------------------------------------------------------------------
    print(f"\n--- Step 4: Saving full label map ---")
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    df_full = pd.DataFrame(all_records, columns=["record_index", "y_true"])
    df_full.to_csv(FULL_LABEL_MAP_PATH, index=False)
    print(f"  Saved: {FULL_LABEL_MAP_PATH}")
    print(f"  Shape: {df_full.shape}")
    print(f"  Label range: {df_full['y_true'].min()} – {df_full['y_true'].max()}")
    print(f"  Unique labels: {df_full['y_true'].nunique()}")

    # ------------------------------------------------------------------
    # Step 5 (Pass 2): Extract pilot images as PNG
    # ------------------------------------------------------------------
    print(f"\n--- Step 5 (Pass 2): Extracting {len(pilot_record_keys)} pilot images ---")
    PILOT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    pilot_rows = []
    image_id = 0
    t0 = time.time()

    for key, y_true, data_offset, data_length in tqdm(
        pilot_record_keys, desc="Extracting images", unit="img"
    ):
        # Create label subdirectory
        label_dir = PILOT_IMAGES_DIR / str(y_true)
        label_dir.mkdir(exist_ok=True)

        # Decode and save image
        img_array = decode_image(rec_file, data_offset, data_length)
        rel_path = f"images/{y_true}/{key}.{IMAGE_FORMAT}"
        abs_path = PILOT_DIR / rel_path
        img_pil = Image.fromarray(img_array)
        img_pil.save(abs_path)

        pilot_rows.append({
            "image_id": image_id,
            "record_index": key,
            "y_true": y_true,
            "image_path": rel_path,
        })
        image_id += 1

    elapsed = time.time() - t0
    print(f"  Extracted {image_id} images in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Step 6: Save pilot metadata.csv
    # ------------------------------------------------------------------
    print(f"\n--- Step 6: Saving pilot metadata ---")
    df_pilot = pd.DataFrame(pilot_rows)
    df_pilot.to_csv(PILOT_METADATA_PATH, index=False)
    print(f"  Saved: {PILOT_METADATA_PATH}")

    # Close the .rec file
    rec_file.close()

    # ------------------------------------------------------------------
    # Step 7: Extraction statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXTRACTION STATISTICS")
    print("=" * 70)
    label_counts = Counter(df_pilot["y_true"])
    counts = list(label_counts.values())
    print(f"  Total identities:        {len(label_counts)}")
    print(f"  Total extracted images:   {len(df_pilot)}")
    print(f"  Label range:             {df_pilot['y_true'].min()} – {df_pilot['y_true'].max()}")
    print(f"  Min images/identity:     {min(counts)}")
    print(f"  Max images/identity:     {max(counts)}")
    print(f"  Mean images/identity:    {np.mean(counts):.1f}")
    print(f"  Median images/identity:  {np.median(counts):.1f}")
    print(f"  Std images/identity:     {np.std(counts):.1f}")
    print(f"  Records extracted:       {len(pilot_record_keys)}")

    # ------------------------------------------------------------------
    # Step 8: Verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # 8a: Check every metadata image_path exists
    print("\n[Check 1] Verifying all metadata image paths exist on disk...")
    missing = 0
    for _, row in df_pilot.iterrows():
        full_path = PILOT_DIR / row["image_path"]
        if not full_path.exists():
            print(f"  MISSING: {full_path}")
            missing += 1
    print(f"  Missing files: {missing}")
    if missing == 0:
        print(f"  [OK] All {len(df_pilot)} image paths verified.")

    # 8b: Randomly open 5 images and check dimensions + mode
    print("\n[Check 2] Spot-checking 5 random images (112x112 RGB)...")
    random.seed(42)
    sample_indices = random.sample(range(len(df_pilot)), min(5, len(df_pilot)))
    all_ok = True
    for i in sample_indices:
        row = df_pilot.iloc[i]
        full_path = PILOT_DIR / row["image_path"]
        try:
            img = Image.open(full_path)
            w, h = img.size
            mode = img.mode
            ok = (w == 112 and h == 112 and mode == "RGB")
            status = "[OK]" if ok else "[FAIL]"
            print(f"  {status} image_id={row['image_id']}, "
                  f"record_index={row['record_index']}, "
                  f"y_true={row['y_true']}, "
                  f"size={w}x{h}, mode={mode}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  [FAIL] image_id={row['image_id']}: {e}")
            all_ok = False
    if all_ok:
        print("  [OK] All spot-checked images are valid 112x112 RGB.")

    # 8c: Re-read RecordIO headers for 5 pilot records and cross-verify labels
    print("\n[Check 3] Cross-verifying labels against RecordIO headers...")
    rec_file = open(REC_PATH, "rb")
    sample_pilot = random.sample(range(len(df_pilot)), min(5, len(df_pilot)))
    verify_ok = True
    for i in sample_pilot:
        row = df_pilot.iloc[i]
        key = row["record_index"]
        expected_label = row["y_true"]
        flag, label_val, _, _, _ = read_record_header(rec_file, idx[key])
        actual_label = int(label_val) if flag == 0 else int(label_val[0])
        match = actual_label == expected_label
        status = "[OK]" if match else "[FAIL]"
        print(f"  {status} record_index={key}: "
              f"metadata y_true={expected_label}, RecordIO label={actual_label}")
        if not match:
            verify_ok = False
    rec_file.close()

    if verify_ok:
        print("  [OK] All cross-verified labels match.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Full label map:    {FULL_LABEL_MAP_PATH}")
    print(f"  Pilot metadata:    {PILOT_METADATA_PATH}")
    print(f"  Pilot images:      {PILOT_IMAGES_DIR}")
    print(f"  Total identities:  {len(label_counts)}")
    print(f"  Total images:      {len(df_pilot)}")
    all_checks_pass = (missing == 0 and all_ok and verify_ok)
    print(f"  All checks passed: {'[OK] YES' if all_checks_pass else '[FAIL] NO'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
