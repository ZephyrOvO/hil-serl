#!/usr/bin/env python3
# renumber_frames.py
import argparse
import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

FRAME_RE = re.compile(r"^frame_(\d+)$")

def find_frame_dirs(root: Path) -> List[Tuple[int, Path]]:
    items = []
    if not root.is_dir():
        raise FileNotFoundError(f"Root not found or not a directory: {root}")
    for p in root.iterdir():
        if p.is_dir():
            m = FRAME_RE.match(p.name)
            if m:
                idx = int(m.group(1))
                items.append((idx, p))
    items.sort(key=lambda x: x[0])
    return items

def plan_renames(root: Path, start: int = 0) -> List[Tuple[Path, Path]]:
    pairs = find_frame_dirs(root)
    renames = []
    for new_i, (_, old_path) in enumerate(pairs, start=start):
        new_name = f"frame_{new_i}"
        new_path = root / new_name
        if old_path == new_path:
            continue  # already correct
        renames.append((old_path, new_path))
    return renames

def two_phase_rename(renames: List[Tuple[Path, Path]], apply: bool, root: Path) -> dict:
    # 生成临时名，确保唯一
    tmp_pairs = []
    for i, (old_p, new_p) in enumerate(renames):
        tmp_p = old_p.with_name(f"__renaming__{i:06d}__")
        tmp_pairs.append((old_p, tmp_p, new_p))

    mapping = []  # 记录 old -> new

    if not apply:
        # Dry-run：打印计划
        for old_p, tmp_p, new_p in tmp_pairs:
            print(f"[DRY-RUN] {old_p.name}  ->  {new_p.name}")
        return {"applied": False, "mapping": []}

    # 第一阶段：old -> tmp
    for old_p, tmp_p, _ in tmp_pairs:
        if not old_p.exists():
            raise FileNotFoundError(f"Missing during phase1: {old_p}")
        if tmp_p.exists():
            raise FileExistsError(f"Temp already exists (unexpected): {tmp_p}")
        os.rename(old_p, tmp_p)

    # 第二阶段：tmp -> new
    for _, tmp_p, new_p in tmp_pairs:
        if new_p.exists():
            # 理论上此时不会存在（因为所有 frame_* 已转为临时名），但仍做保护
            raise FileExistsError(f"Target already exists: {new_p}")
        os.rename(tmp_p, new_p)
        mapping.append({"old": tmp_p.name, "new": new_p.name})

    # 写入映射文件
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = root / f"renumber_map_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"root": str(root), "mapping": mapping}, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote mapping: {out_json}")
    return {"applied": True, "mapping": mapping}

def process_root(root_str: str, start: int, apply: bool):
    root = Path(root_str).resolve()
    print(f"\n=== Processing: {root} ===")
    renames = plan_renames(root, start=start)
    if not renames:
        print("No renaming needed.")
        return
    res = two_phase_rename(renames, apply=apply, root=root)
    if not apply:
        print("[DRY-RUN] Nothing changed. Use --apply to execute.")

def main():
    parser = argparse.ArgumentParser(
        description="Renumber frame_<N> folders to a continuous sequence starting at 0 (per root)."
    )
    parser.add_argument("roots", nargs="+", help="One or more root directories to process.")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0).")
    parser.add_argument("--apply", action="store_true", help="Actually perform renaming.")
    args = parser.parse_args()

    for r in args.roots:
        process_root(r, start=args.start, apply=args.apply)

if __name__ == "__main__":
    main()
