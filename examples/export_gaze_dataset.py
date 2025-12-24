#!/usr/bin/env python3
# examples/export_gaze_dataset.py
from __future__ import annotations
import os, re, json, glob, pickle as pkl, random
import numpy as np
import cv2
from pathlib import Path
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("root_dir", "/mnt/data3/hcj/recorded_data/recorded_data_robot",
                    "根目录：包含多个采集会话，每个会话下有 frame_* 目录")
flags.DEFINE_string("out_dir", "./gaze_cls_data", "输出 pkl 放这里")
flags.DEFINE_integer("shard_size", 5000, "每个 pkl 包含的样本数")
flags.DEFINE_string("image_key", "front_camera", "观测里前视相机的键名")
flags.DEFINE_integer("resize_w", 128, "训练尺寸 W")
flags.DEFINE_integer("resize_h", 128, "训练尺寸 H")
flags.DEFINE_float("val_ratio", 0.2, "验证集比例(0~1)")
flags.DEFINE_integer("seed", 42, "数据切分随机种子")
flags.DEFINE_boolean("stratify", True, "是否按 gaze_hit_class 分层切分")
flags.DEFINE_float("bbox_min_size_ratio", 0.05, "当没有掩码时，基于质心回退生成一个最小正方框的边长比例（相对短边）")
 

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
    """从 mask PNG 里估计质心（取最大连通域）；失败时返回 None。"""
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    bw = (m > 0).astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return None
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 跳过背景
    cx, cy = cents[k]
    return float(cx), float(cy)

def _mask_bbox(mask_path: Path) -> tuple[float,float,float,float] | None:
    """
    返回 (xmin, ymin, xmax, ymax) —— 像素坐标的紧包围框；失败时返回 None。
    """
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    return xmin, ymin, xmax, ymax



def _read_gaze(frame_dir: Path):
    """
    返回 (hit_class, xy_pix, box_pix)。
    统一约定（与任务一致）：hit_class: 0=obj0, 1=obj1, 2=none。
    xy_pix: (x,y) 像素坐标；若 None 返回 (-1,-1)。
    box_pix: (xmin,ymin,xmax,ymax) 像素坐标；若无则返回 (-1,-1,-1,-1)。
    """
    j = frame_dir / "gaze_contact.json"
    if not j.exists():
        return 2, np.array([-1.0, -1.0], np.float32), np.array([-1,-1,-1,-1], np.float32)
    try:
        d = json.loads(j.read_text())
    except Exception:
        return 2, np.array([-1.0, -1.0], np.float32), np.array([-1,-1,-1,-1], np.float32)

    # rs_id = d.get("rs_object_id", None)  # 0 或 1，或 None
    hit = d.get("hit", False)
    rs_id = d.get("rs_object_id", None)  # 0 或 1，或 None
    cx, cy = d.get("centroid_x", None), d.get("centroid_y", None)

    # 优先用 json 里的质心；缺失时回退到 rs_mask 的质心
    # if rs_id in (0, 1):
    if not hit:
        return 2, np.array([-1.0, -1.0], np.float32), np.array([-1,-1,-1,-1], np.float32)
    if rs_id in (0, 1):
        if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
            mapped = int(rs_id)  # 0=obj0, 1=obj1
            # return mapped, np.array([float(cx), float(cy)], np.float32)
            # box 优先从掩码获得
            mask_name = "rs_mask_obj0.png" if rs_id == 0 else "rs_mask_obj1.png"
            box = _mask_bbox(frame_dir / mask_name)
            if box is not None:
                return mapped, np.array([float(cx), float(cy)], np.float32), np.array(box, np.float32)
            else:
                return mapped, np.array([float(cx), float(cy)], np.float32), np.array([-1,-1,-1,-1], np.float32)
        else:
            mask_name = "rs_mask_obj0.png" if rs_id == 0 else "rs_mask_obj1.png"
            # cen = _mask_centroid(frame_dir / mask_name)
            # if cen is not None:
            #     mapped = int(rs_id)
            #     return mapped, np.array(cen, np.float32)
            cen = _mask_centroid(frame_dir / mask_name)
            box = _mask_bbox(frame_dir / mask_name)
            if cen is not None and box is not None:
                mapped = int(rs_id)
                return mapped, np.array(cen, np.float32), np.array(box, np.float32)
            

    # 无命中或无法求质心
    return 2, np.array([-1.0, -1.0], np.float32), np.array([-1,-1,-1,-1], np.float32)
 

def _normalize_xy(xy_pix, src_w, src_h):
    # (-1,-1) 表示 None，占位直接返回
    if xy_pix[0] < 0 or xy_pix[1] < 0:
        return np.array([-1.0, -1.0], np.float32)
    x = np.clip(xy_pix[0] / max(1.0, src_w - 1), 0.0, 1.0)
    y = np.clip(xy_pix[1] / max(1.0, src_h - 1), 0.0, 1.0)
    return np.array([x, y], np.float32)

def _normalize_box(box_pix, src_w, src_h, fallback_xy=None, min_ratio=0.05):
    """
    规范化到 [0,1]，并做 fallback：
    - 若 box_pix 正常，则规范化并裁剪到 [0,1]；
    - 若 box_pix 缺失且提供 fallback_xy，则基于质心生成一个最小正方框（边长=min_ratio*min(w,h)）。
    - 最终若仍无，返回 (-1,-1,-1,-1)。
    """
    if box_pix is not None and np.all(np.array(box_pix) >= 0):
        x0, y0, x1, y1 = box_pix
        w = max(1.0, src_w - 1); h = max(1.0, src_h - 1)
        bx0 = np.clip(x0 / w, 0.0, 1.0); by0 = np.clip(y0 / h, 0.0, 1.0)
        bx1 = np.clip(x1 / w, 0.0, 1.0); by1 = np.clip(y1 / h, 0.0, 1.0)
        if bx1 > bx0 and by1 > by0:
            return np.array([bx0, by0, bx1, by1], np.float32)
    # fallback: 用质心造一个小正方框
    if fallback_xy is not None and fallback_xy[0] >= 0 and fallback_xy[1] >= 0:
        side = min_ratio * float(min(src_w, src_h))
        cx = float(fallback_xy[0]); cy = float(fallback_xy[1])
        x0 = max(0.0, cx - side/2); y0 = max(0.0, cy - side/2)
        x1 = min(float(src_w-1), cx + side/2); y1 = min(float(src_h-1), cy + side/2)
        w = max(1.0, src_w - 1); h = max(1.0, src_h - 1)
        return np.array([x0/w, y0/h, x1/w, y1/h], np.float32)
    return np.array([-1.0, -1.0, -1.0, -1.0], np.float32)


def _save_shards(samples, out_dir: Path, shard_size: int, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    shard_id = 0
    idx = 0
    while idx < n:
        chunk = samples[idx: idx + shard_size]
        p = out_dir / f"{prefix}_{shard_id:05d}.pkl"
        with open(p, "wb") as f:
            pkl.dump(chunk, f)
        print(f"[dump] {p}  ({len(chunk)} samples)")
        shard_id += 1
        idx += shard_size
    print(f"[done] {out_dir}  total={n}, shards={shard_id}")

def _split_train_val(samples, val_ratio: float, seed: int, stratify: bool):
    rng = random.Random(seed)
    if not stratify:
        idxs = list(range(len(samples)))
        rng.shuffle(idxs)
        k = int(len(samples) * (1.0 - val_ratio))
        return [samples[i] for i in idxs[:k]], [samples[i] for i in idxs[k:]]
    # stratified by gaze_hit_class ∈ {0,1,2}
    buckets = {0: [], 1: [], 2: []}
    for s in samples:
        buckets[int(s["gaze_hit_class"])].append(s)
    train, val = [], []
    for cls in buckets:
        arr = buckets[cls]
        rng.shuffle(arr)
        k = int(len(arr) * (1.0 - val_ratio))
        train.extend(arr[:k]); val.extend(arr[k:])
    return train, val



def main(_):
    root = Path(FLAGS.root_dir)
    out = Path(FLAGS.out_dir)
    # out.mkdir(parents=True, exist_ok=True)

    # shard, shard_id = [], 0
    # total = 0
    out.mkdir(parents=True, exist_ok=True)
    samples = []

    for session in sorted(root.iterdir()):
        if not session.is_dir():
            continue
        frames = _sorted_frames(session)
        for fdir in frames:
            img_bgr = _load_color_bgr(fdir)
            if img_bgr is None:
                continue

            h0, w0 = img_bgr.shape[:2]
            hit_class, xy_pix, box_pix = _read_gaze(fdir)
            xy_norm = _normalize_xy(xy_pix, w0, h0)
            box_norm = _normalize_box(box_pix, w0, h0, fallback_xy=(xy_pix if xy_pix[0]>=0 else None),
                                      min_ratio=FLAGS.bbox_min_size_ratio)
            

            # 统一 resize（RGB，uint8）
            # img_resz = cv2.resize(img_bgr, (FLAGS.resize_w, FLAGS.resize_h),
            #                       interpolation=cv2.INTER_LINEAR)
            # img_rgb = img_resz[..., :-1:-1].copy()  # BGR->RGB 的更快写法
            
            img_resz = cv2.resize(img_bgr, (FLAGS.resize_w, FLAGS.resize_h),
                                  interpolation=cv2.INTER_LINEAR)
            img_rgb = img_resz[..., ::-1].copy()

            sample = {
                "observations": {
                    FLAGS.image_key: img_rgb,         # [H, W, 3] uint8
                },
                "gaze_hit_class": np.int32(hit_class),   # 0=obj0, 1=obj1, 2=none
                "gaze_xy": xy_norm.astype(np.float32),   # [2] in [0,1] or (-1,-1)（保留兼容）
                "gaze_box": box_norm.astype(np.float32), # [4] in [0,1] or (-1,-1,-1,-1)
            }
            samples.append(sample)
    total = len(samples)
    if total == 0:
        print("[warn] 没有采到任何样本，退出。")
        return

    train_samples, val_samples = _split_train_val(
        samples,
        val_ratio=FLAGS.val_ratio,
        seed=FLAGS.seed,
        stratify=FLAGS.stratify,
    )

    def _count(arr):
        c0 = sum(1 for s in arr if int(s["gaze_hit_class"]) == 0)
        c1 = sum(1 for s in arr if int(s["gaze_hit_class"]) == 1)
        c2 = sum(1 for s in arr if int(s["gaze_hit_class"]) == 2)
        return c0, c1, c2
    t0,t1,t2 = _count(train_samples)
    v0,v1,v2 = _count(val_samples)
    print(f"[split] train: {len(train_samples)}  (obj0={t0}, obj1={t1}, none={t2})")
    print(f"[split] val  : {len(val_samples)}  (obj0={v0}, obj1={v1}, none={v2})")
    

    out_train = out / "train"
    out_val   = out / "val"
    _save_shards(train_samples, out_train, FLAGS.shard_size, prefix="gaze_samples")
    _save_shards(val_samples,   out_val,   FLAGS.shard_size, prefix="gaze_samples")
    print(f"[Done] total samples = {total}, train={len(train_samples)}, val={len(val_samples)}")


if __name__ == "__main__":
    app.run(main)
