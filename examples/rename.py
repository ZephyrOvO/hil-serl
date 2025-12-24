#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from pathlib import Path


DATASET_ROOT = Path("/home/ruiqiang/workspaces/HK_TacExo_HAN/recorded_data/Ball_Pick/New_Ball_Pick_08_14_09_hcj")

count_total = 0
count_renamed = 0

for frame_dir in sorted(DATASET_ROOT.glob("frame_*")):
    if not frame_dir.is_dir():
        continue

    old_file = frame_dir / "is_successful.txt"
    new_file = frame_dir / "is_record_success.txt"

    if old_file.exists():
        try:
            old_file.rename(new_file)
            count_renamed += 1
        except Exception as e:
            print(f"[ERROR] Failed to rename {old_file}: {e}")
    count_total += 1

print(f"\n[INFO] Scanned {count_total} frame folders.")
print(f"[OK] Renamed {count_renamed} files from is_successful.txt → is_record_success.txt.")
