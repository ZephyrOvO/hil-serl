#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, glob, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import jax
import jax.numpy as jnp
from flax.training import checkpoints

# === 你的工程内路径（按你现有 repo 结构可选）===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
import sys
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 直接从你改过的文件里拿 builder
from serl_launcher.networks.reward_classifier_gaze import create_classifier as create_classifier_extended

# ---------------------- 工具函数 ----------------------

def list_frames(root: Path) -> List[Path]:
    """
    兼容 frame0 / frame_0 / frame_0001 / frame1 等命名。
    """
    cand = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        # 匹配 'frame' 后跟可选的 '_'，再跟数字
        m = re.match(r"^frame_?(\d+)$", d.name)
        if not m:
            # 也接受完全没有下划线的 'frame' + 数字，如 frame23
            m = re.match(r"^frame(\d+)$", d.name)
        if m:
            idx = int(m.group(1))
            cand.append((idx, d))
    cand.sort(key=lambda x: x[0])
    return [p for _, p in cand]

def tight_bbox_from_mask(mask_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    从 mask png 取>0的像素集，返回像素坐标的 (xmin, ymin, xmax, ymax)；无像素则 None。
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
    return (xmin, ymin, xmax, ymax)

def norm_box_xyxy(box_xyxy_pix, w: int, h: int) -> np.ndarray:
    """像素 bbox -> 归一化 [0,1] xyxy。"""
    x0, y0, x1, y1 = box_xyxy_pix
    W = max(1.0, w - 1)
    H = max(1.0, h - 1)
    return np.array([x0 / W, y0 / H, x1 / W, y1 / H], dtype=np.float32)

def denorm_box_xyxy(box_xyxy_norm, w: int, h: int) -> np.ndarray:
    """归一化 xyxy -> 像素 bbox。"""
    x0, y0, x1, y1 = box_xyxy_norm
    W = max(1.0, w - 1)
    H = max(1.0, h - 1)
    return np.array([x0 * W, y0 * H, x1 * W, y1 * H], dtype=np.float32)

def load_gt_class_and_bbox(frame_dir: Path) -> Tuple[int, Optional[np.ndarray]]:
    """
    解析 gaze_contact.json 的 class_id：
      - 0 -> GT class 0；bbox 从 rs_mask_obj0.png
      - 1 -> GT class 1；bbox 从 rs_mask_obj1.png
      - null/缺失 -> GT class 2；bbox = None（不参与 bbox 评估）
    返回： (gt_class, gt_box_norm or None)
    """
    jpath = frame_dir / "gaze_contact.json"
    gt_cls = 2
    gt_box_norm = None

    if jpath.exists():
        try:
            d = json.loads(jpath.read_text())
            cid = d.get("class_id", None)  # 可能为 0/1 或 None
            if cid is None:
                gt_cls = 2
            elif int(cid) in (0, 1):
                gt_cls = int(cid)
            else:
                gt_cls = 2
        except Exception:
            gt_cls = 2

    if gt_cls in (0, 1):
        mask_name = "rs_mask_obj0.png" if gt_cls == 0 else "rs_mask_obj1.png"
        mpath = frame_dir / mask_name
        box_pix = tight_bbox_from_mask(mpath)
        if box_pix is not None:
            img = cv2.imread(str(frame_dir / "color_image.jpg"))
            if img is not None:
                h, w = img.shape[:2]
                gt_box_norm = norm_box_xyxy(box_pix, w, h)

    return gt_cls, gt_box_norm

def draw_bboxes(img_bgr: np.ndarray,
                pred_box_norm: Optional[np.ndarray],
                gt_box_norm: Optional[np.ndarray]) -> np.ndarray:
    """
    在 BGR 图上画预测框(实线)和 GT 框(虚线)；均为归一化输入。
    颜色：pred = 绿色、gt = 蓝色虚线
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    def _rect(img, xyxy_norm, color, thickness=2, dotted=False):
        x0, y0, x1, y1 = denorm_box_xyxy(xyxy_norm, w, h).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w-1, x1), min(h-1, y1)
        if dotted:
            # 画虚线（简单实现：跳点）
            step = 5
            for x in range(x0, x1, step*2):
                cv2.line(img, (x, y0), (min(x+step, x1), y0), color, thickness)
                cv2.line(img, (x, y1), (min(x+step, x1), y1), color, thickness)
            for y in range(y0, y1, step*2):
                cv2.line(img, (x0, y), (x0, min(y+step, y1)), color, thickness)
                cv2.line(img, (x1, y), (x1, min(y+step, y1)), color, thickness)
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

    if pred_box_norm is not None and np.all(pred_box_norm >= 0):
        _rect(out, pred_box_norm, (0,255,0), 2, dotted=False)  # 绿：pred
    if gt_box_norm is not None and np.all(gt_box_norm >= 0):
        _rect(out, gt_box_norm, (255,128,0), 2, dotted=True)   # 橙蓝系虚线更显眼

    return out

def build_model_for_ckpt(rngkey, image_key: str, input_size=(128,128), task="gaze_box"):
    """
    构图并返回一个 apply 函数：输入 dict(front_camera: (B,H,W,3) float32[0..1]) -> (logits, box_xyxy)
    """
    state = create_classifier_extended(
        rngkey,
        {image_key: np.zeros((1, input_size[1], input_size[0], 3), np.float32)},
        image_keys=[image_key],
        image_key_weights=None,
        task=task,
    )
    return state

def restore_state_from_ckpt_dir(state, ckpt_dir: str, step: int, prefix: str):
    """
    从 ckpt_dir 中恢复指定 step（例如 epoch_0050）。要求保存时用了相同的 prefix。
    """
    state = checkpoints.restore_checkpoint(
        ckpt_dir,
        target=state,
        step=step,
        prefix=prefix,
    )
    return state

def softmax_np(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

# ---------------------- 主流程 ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",
                    default="/mnt/data3/hcj/valid_data/recorded_data_training-10-11-10",
                    help="包含 frame* 子目录的验证数据根")
    ap.add_argument("--ckpt_dir",
                    default="/mnt/data3/hcj/gaze_ckpt",
                    help="ckpt 所在目录（目录内应包含 epoch_* 文件）")
    ap.add_argument("--ckpt_prefix", default="checkpoint_", help="保存 ckpt 时用的 prefix")
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--end",   type=int, default=2300)
    ap.add_argument("--step",  type=int, default=50)
    ap.add_argument("--img_key", default="front_camera")
    ap.add_argument("--resize_w", type=int, default=128)
    ap.add_argument("--resize_h", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out_dir",
                    default="/mnt/data3/hcj/gaze_eval_out",
                    help="输出指标、可视化等")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    frames = list_frames(data_root)
    if not frames:
        print(f"[ERR] No frame* folders found under {data_root}")
        return

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    csv_dir = out_dir / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 预读所有样本：原图、GT class、GT box（归一化）
    all_imgs_rgb: List[np.ndarray] = []
    all_sizes: List[Tuple[int,int]] = []
    all_gt_cls: List[int] = []
    all_gt_box: List[Optional[np.ndarray]] = []
    all_names: List[str] = []

    for fdir in frames:
        name = fdir.name
        img_bgr = cv2.imread(str(fdir / "color_image.jpg"))
        if img_bgr is None:
            continue
        h0, w0 = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 归一化输入到网络期望大小
        img_resz = cv2.resize(img_rgb, (args.resize_w, args.resize_h), interpolation=cv2.INTER_LINEAR)
        all_imgs_rgb.append(img_resz.astype(np.float32) / 255.0)
        all_sizes.append((w0, h0))
        gt_cls, gt_box_norm = load_gt_class_and_bbox(fdir)
        all_gt_cls.append(gt_cls)
        all_gt_box.append(gt_box_norm)
        all_names.append(name)

    imgs_np = np.stack(all_imgs_rgb, axis=0)  # (N, H, W, 3)
    gt_cls_np = np.array(all_gt_cls, dtype=np.int32)
    N = imgs_np.shape[0]

    # JAX 模型构建（一次），然后每个 ckpt 恢复参数
    rng = jax.random.PRNGKey(0)
    state = build_model_for_ckpt(rng, args.img_key, (args.resize_w, args.resize_h), task="gaze_box")

    @jax.jit
    def apply_fn(params, batch):
        logits, boxes = state.apply_fn({"params": params}, batch, train=False)
        return logits, boxes

    # 汇总 CSV（所有 ckpt 一行）
    summary_rows = []
    summary_header = ["ckpt_step", "num_frames", "cls_acc", "bbox_mae_l1_mean", "bbox_eval_count"]

    for step in range(args.start, args.end + 1, args.step):
        print(f"\n[CKPT] Evaluating step={step} from {args.ckpt_dir} ...")
        state_i = restore_state_from_ckpt_dir(state, args.ckpt_dir, step=step, prefix=args.ckpt_prefix)

        # 批量前向
        pred_logits_list = []
        pred_boxes_list  = []
        for i in range(0, N, args.batch):
            sl = slice(i, min(i + args.batch, N))
            batch = {args.img_key: imgs_np[sl]}
            logits, boxes = apply_fn(state_i.params, batch)
            pred_logits_list.append(np.array(logits))
            pred_boxes_list.append(np.array(boxes))
        pred_logits = np.concatenate(pred_logits_list, axis=0)  # (N,3)
        pred_boxes  = np.concatenate(pred_boxes_list,  axis=0)  # (N,4) 归一化 xyxy

        # 分类准确率（gt_cls: 0/1/2）
        pred_cls = np.argmax(pred_logits, axis=-1)              # (N,)
        cls_acc = float((pred_cls == gt_cls_np).mean())

        # bbox L1（只对 gt_cls in {0,1} 的样本）
        bbox_mask = (gt_cls_np < 2).astype(np.float32)          # (N,)
        bbox_cnt  = int(bbox_mask.sum())
        bbox_l1_mean = float("nan")
        if bbox_cnt > 0:
            # 汇总 GT
            gt_boxes_stack = []
            valid_idx = []
            for i in range(N):
                if bbox_mask[i] > 0 and (all_gt_box[i] is not None):
                    gt_boxes_stack.append(all_gt_box[i])
                    valid_idx.append(i)
            if gt_boxes_stack:
                gt_boxes_stack = np.stack(gt_boxes_stack, axis=0)     # (M,4)
                pred_boxes_sel = pred_boxes[np.array(valid_idx)]
                l1 = np.abs(pred_boxes_sel - gt_boxes_stack).sum(axis=-1)  # (M,)
                bbox_l1_mean = float(l1.mean())
                bbox_cnt = int(l1.shape[0])
            else:
                bbox_cnt = 0
                bbox_l1_mean = float("nan")

        # 保存每帧预测到 CSV
        csv_path = csv_dir / f"pred_{step:04d}.csv"
        with open(csv_path, "w") as f:
            f.write("frame_name,gt_class,pred_class,prob0,prob1,prob2,pred_x0,pred_y0,pred_x1,pred_y1,gt_x0,gt_y0,gt_x1,gt_y1\n")
            probs = softmax_np(pred_logits)
            for i in range(N):
                gt_box = all_gt_box[i] if all_gt_box[i] is not None else np.array([-1,-1,-1,-1], dtype=np.float32)
                f.write("{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                    all_names[i],
                    gt_cls_np[i],
                    int(pred_cls[i]),
                    probs[i,0], probs[i,1], probs[i,2],
                    pred_boxes[i,0], pred_boxes[i,1], pred_boxes[i,2], pred_boxes[i,3],
                    gt_box[0], gt_box[1], gt_box[2], gt_box[3],
                ))

        # 保存指标 JSON
        metrics_path = out_dir / f"metrics_{step:04d}.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "ckpt_step": step,
                "num_frames": int(N),
                "cls_acc": cls_acc,
                "bbox_mae_l1_mean": bbox_l1_mean,
                "bbox_eval_count": bbox_cnt,
            }, f, indent=2)
        print(f"[CKPT {step}] acc={cls_acc:.4f}, bbox_mae(L1)={bbox_l1_mean} (count={bbox_cnt})")
        print(f"  -> {metrics_path}")
        print(f"  -> {csv_path}")

        # 可视化（每帧一张图）
        ckpt_vis_dir = vis_dir / f"ckpt_{step:04d}"
        ckpt_vis_dir.mkdir(parents=True, exist_ok=True)
        for i in range(N):
            # 读原图（为避免重复 IO，也可复用 earlier 但我们需要 BGR）
            fdir = data_root / all_names[i]
            img_bgr = cv2.imread(str(fdir / "color_image.jpg"))
            if img_bgr is None:
                continue
            pred_box_norm = pred_boxes[i]
            gt_box_norm = all_gt_box[i] if all_gt_box[i] is not None else None
            img_draw = draw_bboxes(img_bgr, pred_box_norm, gt_box_norm)
            # 叠加分类标签
            label_text = f"gt:{gt_cls_np[i]}  pred:{int(pred_cls[i])}"
            cv2.putText(img_draw, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            cv2.imwrite(str(ckpt_vis_dir / f"{all_names[i]}.jpg"), img_draw)

        # 汇总
        summary_rows.append([step, N, cls_acc, bbox_l1_mean, bbox_cnt])

    # 写总汇总
    summary_csv = out_dir / "summary_all_ckpt.csv"
    with open(summary_csv, "w") as f:
        f.write(",".join(summary_header) + "\n")
        for r in summary_rows:
            f.write("{},{},{},{},{}\n".format(r[0], r[1], r[2], r[3], r[4]))
    print(f"\n[Done] Wrote {summary_csv}")

if __name__ == "__main__":
    main()
