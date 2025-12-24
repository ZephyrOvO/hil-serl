import pickle as pkl
import numpy as np
import cv2
import os

IN_PKL  = "/home/ruiqiang/workspaces/HK_TacExo_HAN/hil-serl/examples/demo_data/tennis_ball_pick_18_demos_2025-10-16_18-44-43.pkl"  
OUT_PKL = IN_PKL.replace(".pkl", "_fixed.pkl")

def to_3ch(img):
    """确保是(H,W,3)，把灰度/单通道扩为3通道。"""
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img

def ensure_uint8(img):
    if img.dtype != np.uint8:
        # 若是0/1 float，放大到0/255
        if img.dtype in (np.float32, np.float64):
            if img.max() <= 1.0:
                img = (img * 255.0).round().astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img

def fix_one_image(img, target_hw=(128,128)):
    """返回形状(T=1,H,W,3)的uint8。"""
    img = np.asarray(img)
    # 去掉可能的时间维
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    # 统一成3通道
    img = to_3ch(img)
    # resize到128x128
    H, W = img.shape[:2]
    if (H, W) != target_hw:
        img = cv2.resize(img, target_hw[::-1], interpolation=cv2.INTER_LINEAR)
    # 转uint8
    img = ensure_uint8(img)
    # 加时间维
    img = img[None, ...]
    return img

def fix_transition(tr):
    obs = tr["observations"]
    nxt = tr["next_observations"]

    for key in ["front_camera", "tactile_data", "gaze_mask"]:
        if key in obs:
            obs[key] = fix_one_image(obs[key])
        if key in nxt:
            nxt[key] = fix_one_image(nxt[key])

    tr["observations"] = obs
    tr["next_observations"] = nxt
    return tr

def main():
    with open(IN_PKL, "rb") as f:
        data = pkl.load(f)
    print("Loaded transitions:", len(data))
    fixed = [fix_transition(t) for t in data]
    with open(OUT_PKL, "wb") as f:
        pkl.dump(fixed, f)
    print("Saved fixed transitions to:", OUT_PKL)

if __name__ == "__main__":
    main()
