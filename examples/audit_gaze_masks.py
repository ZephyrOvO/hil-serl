#!/usr/bin/env python3
import os, sys, pickle as pkl, json
from pathlib import Path
import numpy as np
import cv2
from collections import Counter, defaultdict

PKL = sys.argv[1] if len(sys.argv) > 1 else "./demo_data/tennis_ball_pick_5_demos_xxx.pkl"
FRAME_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else None   # 可选：传你的 frame_* 根目录
# 例如：python audit_gaze_masks.py ./demo_data/xxx.pkl /mnt/data3/hcj/recorded_data_...

def get_mask_from_obs(obs):
    """容错抽取 (128,128,3) 的 gaze_mask；若无则返回 None。"""
    if not isinstance(obs, dict): 
        return None
    if "images" in obs and isinstance(obs["images"], dict) and "gaze_mask" in obs["images"]:
        return obs["images"]["gaze_mask"]
    if "gaze_mask" in obs:
        return obs["gaze_mask"]
    return None

def norm_mask(m):
    """统一成 (H,W) uint8 灰度，便于比较与统计；不改原对象。"""
    if m is None: 
        return None
    x = m
    if isinstance(x, np.ndarray) and x.size > 0:
        if x.ndim == 3 and x.shape[2] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        elif x.ndim == 3 and x.shape[2] == 4:
            x = cv2.cvtColor(x, cv2.COLOR_BGRA2GRAY)
        elif x.ndim == 3:
            x = x[..., 0]
        if x.dtype != np.uint8:
            x = x.astype(np.uint8, copy=False)
        return x
    return None

def load_disk_mask(fid, out_hw=None):
    """从磁盘 frame_{fid}/gaze_mask.png 读灰度；可选 resize 到 out_hw。"""
    if FRAME_ROOT is None: 
        return None
    p = FRAME_ROOT / f"frame_{int(fid)}/gaze_mask.png"
    if not p.exists(): 
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None: 
        return None
    if out_hw is not None and (m.shape[0], m.shape[1]) != out_hw:
        m = cv2.resize(m, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m

with open(PKL, "rb") as f:
    transitions = pkl.load(f)

n = len(transitions)
present_obs = 0
present_next = 0
shape_counter_obs = Counter()
shape_counter_next = Counter()
nonzero_hist = defaultdict(int)

mismatch_disk = 0
checked_disk = 0

def nz_ratio(m):
    if m is None: 
        return None
    return float((m > 0).mean())

for i, tr in enumerate(transitions):
    obs  = tr.get("observations", {})
    nxt  = tr.get("next_observations", {})
    info = tr.get("infos", {})

    m_obs = get_mask_from_obs(obs)
    m_nxt = get_mask_from_obs(nxt)

    if m_obs is not None:
        present_obs += 1
        shape_counter_obs[m_obs.shape] += 1
    if m_nxt is not None:
        present_next += 1
        shape_counter_next[m_nxt.shape] += 1

    # 统计非零比例（灰度后）
    mo = norm_mask(m_obs)
    mn = norm_mask(m_nxt)
    r1 = nz_ratio(mo);  r2 = nz_ratio(mn)
    if r1 is not None: nonzero_hist[round(r1, 3)] += 1
    if r2 is not None: nonzero_hist[round(r2, 3)] += 1

    # 和磁盘比对（可选）
    if FRAME_ROOT is not None and "frame_idx" in info:
        fid = int(info["frame_idx"])
        dm = load_disk_mask(fid, out_hw=mo.shape if mo is not None else None)
        if dm is not None and mo is not None:
            checked_disk += 1
            if not np.array_equal(dm, mo):
                # 允许 next_observations 对齐到 fid+1；再试比对一下
                dm2 = load_disk_mask(fid+1, out_hw=mn.shape if mn is not None else None)
                if dm2 is None or (mn is not None and not np.array_equal(dm2, mn)):
                    mismatch_disk += 1

print("==== GAZE MASK AUDIT ====")
print(f"Transitions: {n}")
print(f"OBS  masks present: {present_obs} / {n}")
print(f"NEXT masks present: {present_next} / {n}")
print("OBS  shape counts:", dict(shape_counter_obs))
print("NEXT shape counts:", dict(shape_counter_next))
if nonzero_hist:
    # 给出一个粗糙的直方统计：0.0 表示全黑，>0 表示有前景
    zeros = sum(cnt for r,cnt in nonzero_hist.items() if r == 0.0)
    nonzeros = sum(cnt for r,cnt in nonzero_hist.items() if r > 0.0)
    print(f"Non-zero ratio histogram (rounded): samples={zeros+nonzeros}, zeros={zeros}, nonzeros={nonzeros}")
    # 可选：打印前 10 个桶
    top = sorted(nonzero_hist.items(), key=lambda x: (-x[1], x[0]))[:10]
    print("Top bins:", top)

if FRAME_ROOT is not None:
    print(f"Disk compare checked: {checked_disk}, mismatches: {mismatch_disk}")
    if mismatch_disk == 0 and checked_disk > 0:
        print("Disk masks are consistent with PKL-packed masks ✅")
    elif checked_disk == 0:
        print("No on-disk masks checked (missing frame_idx or files).")
    else:
        print("Found mismatches between PKL and disk masks. Inspect those fids.")
