#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
record_demos.py — 纯“逐帧在线”版本
依赖 DensoEnv 已内置：
- _infer_gaze_box_from_rgb(): BGR->RGB resize 640x480 -> gaze_apply(obs, n_way=3)
- pred==2 -> 返回全零掩码；否则 _mask_from_box_sam2() 单帧 box prompt 生成 mask
- get_im() 末尾：将 full-res mask NEAREST 下采样到 (128,128) 并 repeat 成 3 通道，
  写入 obs/images['gaze_mask']，同时镜像落盘 frame_{id}/gaze_mask.png 和 rs_images/{id}.jpg
- gaze_ckpt 路径：self.gaze_ckpt = ".../gaze_ckpt/checkpoint_100"（写死在 env 内）

"""

import os
import sys
import copy
import pickle as pkl
import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
from absl import app, flags

# export PYTHONPATH=$(pwd)/../serl_robot_infra:$PYTHONPATH
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
# sys.path.insert(0, project_root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, "Experiment folder not found."
    config = NEW_MAPPING[FLAGS.exp_name]()
    # 关键：你的 env 已经在每帧完成 gaze->SAM2->gaze_mask 的落到 obs
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    obs, info = env.reset()
    # 标记首个 episode 的起点（若你的 env 需要）
    if getattr(env.unwrapped, "_cur_ep_start", None) is None:
        env.unwrapped._cur_ep_start = env.unwrapped.global_frame_id
        print(f"[record_demos][mark] start={env.unwrapped._cur_ep_start} @first reset")

    transitions = []
    trajectory = []
    returns = 0.0
    success_count = 0
    success_needed = int(FLAGS.successes_needed)
    pbar = tqdm(total=success_needed)

    try:
        while success_count < success_needed:
            # 这里用占位动作；你也可以接键盘/策略
            actions = np.zeros(env.action_space.sample().shape, dtype=np.float32)
            actions[1] = -0.1

            next_obs, rew, done, truncated, info = env.step(actions)
            returns += float(rew)

            # 直接记录（obs/next_obs 里已含 images['gaze_mask']）
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)
            pbar.set_description(f"Return: {returns:.2f}")

            obs = next_obs

            if done:
                if info.get("succeed", False):
                    transitions.extend(copy.deepcopy(trajectory))
                    success_count += 1
                    pbar.update(1)

                # reset
                trajectory = []
                returns = 0.0
                input("reset env")
                obs, info = env.reset()
                env.unwrapped._cur_ep_start = env.unwrapped.global_frame_id
                print(f"[record_demos][mark] start={env.unwrapped._cur_ep_start} @per-episode reset"); sys.stdout.flush()

    finally:
        # 由 env 负责把每帧镜像保存到磁盘（rs_images/ 与 frame_k/gaze_mask.png）
        env.save_all_data_on_exit()
        if hasattr(env, "keyboard_process") and getattr(env, "keyboard_process").is_alive():
            print("Shutting down keyboard process...")
            env.keyboard_process.terminate()
            env.keyboard_process.join()

    # 保存 demos
    os.makedirs("./demo_data", exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
    print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)
