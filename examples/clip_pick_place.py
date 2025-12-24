#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np

def natural_key(p: Path):
    m = re.search(r'(\d+)', p.name)
    return (int(m.group(1)) if m else -1, p.name)

def list_frame_dirs(src_root: Path):
    frames = [p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("frame_")]
    frames.sort(key=natural_key)
    return frames

def load_frame_image(frame_dir: Path):
    # 你数据里是 frame_x/color_image.jpg，这里也提供几个兜底名
    for name in ["color_image.jpg", "color_image.png", "rs_image.jpg", "rs_image.png",
                 "color_image2.jpg", "color_image2.png"]:
        p = frame_dir / name
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img, name
    return None, None

def draw_overlay(img, lines, color=(255,255,0)):
    vis = img.copy()
    y = 28
    for t in lines:
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 26
    return vis

def robust_src_path(user_src: Path) -> Path:
    """如果用户给的路径不存在，尝试把最后一段名字里的'_'和'-'互换后再查一次。"""
    if user_src.exists():
        return user_src
    parts = list(user_src.parts)
    if not parts:
        return user_src
    last = parts[-1]
    alternatives = set()
    if "_" in last:
        alternatives.add(last.replace("_", "-"))
    if "-" in last:
        alternatives.add(last.replace("-", "_"))
    for alt in alternatives:
        cand = Path(*parts[:-1], alt)
        if cand.exists():
            print(f"[NOTE] --src 不存在，使用近似路径：{cand}")
            return cand
    return user_src  # 仍然返回原路径，后续抛错

def move_tree(src_dir: Path, dst_dir: Path):
    """把整个 frame_xxxx 目录移动到目标；同盘用 rename，更快；跨盘/已有同名则用 shutil.move。"""
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists():
        # 防止重名覆盖：在目标后缀加一段
        i = 1
        base = dst_dir.name
        while dst_dir.exists():
            dst_dir = dst_dir.parent / f"{base}__{i}"
            i += 1
    try:
        src_dir.rename(dst_dir)  # 同文件系统直接重命名移动
    except Exception:
        shutil.move(str(src_dir), str(dst_dir))  # 跨盘或重命名失败时

def main():
    ap = argparse.ArgumentParser(description="手动将 tennis-ball 任务按 pick/place 切分（s/e 标注），并将 pick 帧目录移动到新位置。")
    ap.add_argument("--src", required=True, type=str, help="源数据根目录（包含 frame_xxxx 子目录）")
    ap.add_argument("--dst", required=True, type=str, help="pick 输出目录（将把选中的 frame_xxxx 目录移动到此处）")
    ap.add_argument("--win", default="Pick Clipper", type=str, help="窗口名")
    ap.add_argument("--resize", default="1280x720", type=str, help="显示尺寸，如 1280x720；留空表示按原图显示")
    args = ap.parse_args()

    src_root = robust_src_path(Path(args.src).expanduser())
    if not src_root.exists():
        raise FileNotFoundError(f"源目录不存在：{src_root}")

    dst_root = Path(args.dst).expanduser()
    dst_root.mkdir(parents=True, exist_ok=True)

    # 枚举帧目录
    frame_dirs = list_frame_dirs(src_root)
    if not frame_dirs:
        raise RuntimeError(f"未找到 frame_* 子目录：{src_root}")

    # 解析显示尺寸
    target_size = None
    if args.resize and "x" in args.resize:
        try:
            w, h = args.resize.lower().split("x")
            target_size = (int(w), int(h))
        except Exception:
            target_size = None

    print(f"[INFO] 共 {len(frame_dirs)} 个帧目录。")
    print("[INFO] 操作：s=起点  e=终点  ←/a=前一帧  →/d=后一帧  space=暂停/继续  q/ESC=结束")
    print("[INFO] 你可以标多个区间，脚本会合并重叠/相邻区间。结束后会将 pick 的 frame_* 目录“移动”至 --dst。")

    idx = 0
    paused = True
    current_start = None
    ranges = []  # (s, e) 都是帧索引，闭区间

    cv2.namedWindow(args.win, cv2.WINDOW_NORMAL)

    while 0 <= idx < len(frame_dirs):
        fdir = frame_dirs[idx]
        img, img_name = load_frame_image(fdir)
        if img is None:
            img = np.zeros((480, 640, 3), np.uint8)
            img_name = "(color_image.* 缺失)"

        lines = [
            f"{src_root.name}  [{idx+1}/{len(frame_dirs)}]  {fdir.name}/{img_name}",
            f"ranges: {ranges}",
            f"current_start: {current_start if current_start is not None else '-'}",
            "keys: s=start, e=end, ←/a, →/d, space, q/ESC"
        ]
        vis = draw_overlay(img, lines)
        if target_size:
            vis = cv2.resize(vis, target_size, interpolation=cv2.INTER_AREA)
        cv2.imshow(args.win, vis)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):  # ESC
            break
        elif key in (ord('a'), 81):  # 左

            idx = max(0, idx - 1); paused = True
        elif key in (ord('d'), 83):  # 右
            idx = min(len(frame_dirs) - 1, idx + 1); paused = True
        elif key == ord(' '):
            idx = min(len(frame_dirs) - 1, idx + 1); paused = True
        elif key == ord('s'):
            current_start = idx; paused = True
        elif key == ord('e'):
            if current_start is None:
                print("[WARN] 请先按 s 再按 e。")
            else:
                s, e = int(current_start), int(idx)
                if e < s: s, e = e, s
                ranges.append((s, e))
                print(f"[INFO] 记录区间: {s}..{e}")
                current_start = None
                paused = True
        else:
            pass


    cv2.destroyAllWindows()

    # 若 start 未闭合，默认闭合到最后一帧
    if current_start is not None:
        s, e = int(current_start), len(frame_dirs) - 1
        ranges.append((s, e))
        print(f"[INFO] 自动闭合最后区间: {s}..{e}")

    # 合并重叠/相邻区间
    ranges.sort()
    merged = []
    for s, e in ranges:
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s <= pe + 1:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e])
    ranges = [(s, e) for s, e in merged]

    # 需要移动的帧索引
    pick_idx = set()
    for s, e in ranges:
        for i in range(s, e + 1):
            pick_idx.add(i)
    pick_idx = sorted(pick_idx)

    # 执行“移动”
    moved = []
    for i in pick_idx:
        src_dir = frame_dirs[i]
        dst_dir = dst_root / src_dir.name
        try:
            move_tree(src_dir, dst_dir)
            moved.append(src_dir.name)
        except Exception as ex:
            print(f"[ERROR] 移动失败：{src_dir} -> {dst_dir}  ({ex})")

    # 写出记录（manifest + pick_frames + place_frames）
    manifest = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "ranges_inclusive": ranges,
        "moved_frames": moved,
        "total_moved": len(moved),
    }
    (dst_root / "pick_ranges.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    (dst_root / "pick_frames.txt").write_text("\n".join(moved))

    # place 清单：源目录里剩下的 frame_*（未移动的）
    try:
        remain = list_frame_dirs(src_root)
        place_names = [p.name for p in remain]
        # 放到源目录，便于你以“剩余即 place”的逻辑继续处理
        (src_root / "place_frames.txt").write_text("\n".join(place_names))
    except Exception:
        pass

    print(f"[DONE] 已移动 {len(moved)} 个 frame_* 到：{dst_root}")
    print(f"[INFO] pick_ranges.json / pick_frames.txt 写在：{dst_root}")
    print(f"[INFO] place 清单写在：{src_root/'place_frames.txt'}")

if __name__ == "__main__":
    main()
