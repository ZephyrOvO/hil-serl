#!/usr/bin/env python3
# examples/export_gaze_dataset.py

import os, re, json, glob, pickle as pkl
import numpy as np
import cv2
from pathlib import Path
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("root_dir", "/mnt/data3/hcj/recorded_data/Ball_Pick",
                    "根目录：包含多个采集会话，每个会话下有 frame_* 目录")
flags.DEFINE_string("out_dir", "./gaze_cls_data", "输出 pkl 放这里")
flags.DEFINE_integer("shard_size", 5000, "每个 pkl 包含的样本数")
flags.DEFINE_string("image_key", "front_camera", "观测里前视相机的键名")
flags.DEFINE_integer("resize_w", 128, "训练尺寸 W")
flags.DEFINE_integer("resize_h", 128, "训练尺寸 H")

def _sorted_frames(session_dir: Path):
    frames = []
    for d in session_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"frame_(\d+)$", d.name)
        if m:
            frames.append((int(m.group(1)), d))
    frames.sort(key=lambda x: x[0])
    return [p for _, p in frames]

def _load_color_bgr(frame_dir: Path):
    p = frame_dir / "color_image.jpg"
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    return img

def _mask_centroid(mask_path: Path):
    """从 mask PNG 里估计质心；失败时返回 None。"""
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    _, bw = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
    M = cv2.moments(bw)
    if M["m00"] <= 1e-5:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return cx, cy

def _read_gaze(frame_dir: Path):
    """
    返回 (hit_class, xy_pix)。
    统一约定：hit_class: 0=None, 1=obj0, 2=obj1。
    xy_pix: (x,y) 像素坐标；若 None 返回 (-1,-1)。
    """
    j = frame_dir / "gaze_contact.json"
    if not j.exists():
        return 0, np.array([-1.0, -1.0], np.float32)

    try:
        d = json.loads(j.read_text())
    except Exception:
        return 0, np.array([-1.0, -1.0], np.float32)

    rs_id = d.get("rs_object_id", None)  # 0 或 1，或 None
    cx, cy = d.get("centroid_x", None), d.get("centroid_y", None)

    # 优先用 json 里的质心；缺失时回退到 rs_mask 的质心
    if rs_id in (0, 1):
        if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
            mapped = 1 if rs_id == 0 else 2  # 0(obj0)->1, 1(obj1)->2
            return mapped, np.array([float(cx), float(cy)], np.float32)
        else:
            mask_name = "rs_mask_obj0.png" if rs_id == 0 else "rs_mask_obj1.png"
            cen = _mask_centroid(frame_dir / mask_name)
            if cen is not None:
                mapped = 1 if rs_id == 0 else 2
                return mapped, np.array(cen, np.float32)

    # 无命中或无法求质心
    return 0, np.array([-1.0, -1.0], np.float32)

def _normalize_xy(xy_pix, src_w, src_h):
    # (-1,-1) 表示 None，占位直接返回
    if xy_pix[0] < 0 or xy_pix[1] < 0:
        return np.array([-1.0, -1.0], np.float32)
    x = np.clip(xy_pix[0] / max(1.0, src_w - 1), 0.0, 1.0)
    y = np.clip(xy_pix[1] / max(1.0, src_h - 1), 0.0, 1.0)
    return np.array([x, y], np.float32)

def main(_):
    root = Path(FLAGS.root_dir)
    out = Path(FLAGS.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shard, shard_id = [], 0
    total = 0

    for session in sorted(root.iterdir()):
        if not session.is_dir():
            continue
        frames = _sorted_frames(session)
        for fdir in frames:
            img_bgr = _load_color_bgr(fdir)
            if img_bgr is None:
                continue

            h0, w0 = img_bgr.shape[:2]
            hit_class, xy_pix = _read_gaze(fdir)
            xy_norm = _normalize_xy(xy_pix, w0, h0)

            # 统一 resize（RGB，uint8）
            img_resz = cv2.resize(img_bgr, (FLAGS.resize_w, FLAGS.resize_h),
                                  interpolation=cv2.INTER_LINEAR)
            img_rgb = img_resz[..., :-1:-1].copy()  # BGR->RGB 的更快写法

            sample = {
                "observations": {
                    FLAGS.image_key: img_rgb,         # [H, W, 3] uint8
                },
                "gaze_hit_class": np.int32(hit_class),   # 0/1/2
                "gaze_xy": xy_norm.astype(np.float32),   # [2] in [0,1] or (-1,-1)
            }
            shard.append(sample)
            total += 1

            if len(shard) >= FLAGS.shard_size:
                p = out / f"gaze_samples_{shard_id:05d}.pkl"
                with open(p, "wb") as f:
                    pkl.dump(shard, f)
                print(f"[dump] {p}  ({len(shard)} samples)")
                shard, shard_id = [], shard_id + 1

    if shard:
        p = out / f"gaze_samples_{shard_id:05d}.pkl"
        with open(p, "wb") as f:
            pkl.dump(shard, f)
        print(f"[dump] {p}  ({len(shard)} samples)")

    print(f"Done. total samples = {total}, shards = {shard_id + (1 if shard else 0)}")

if __name__ == "__main__":
    app.run(main)
