import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

# 提前输入export PYTHONPATH=$(pwd)/../serl_robot_infra:$PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
sys.path.insert(0, project_root)

from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

# ===================== NEW (仅添加)：SAM2 v2 的在线一致化开关与参数 =====================
# 与 train_rlpd.py 的 actor 端保持一致的批处理策略
flags.DEFINE_boolean("sam2_enable", True, "Enable inline SAM2 v2 batch labeling during recording (match online flow).")
flags.DEFINE_integer("sam2_batch_episodes", 1, "Trigger one SAM2 v2 run after collecting this many episodes.")
flags.DEFINE_integer("sam2_y_prompts_et", 20, "Total ET keyframes to annotate per batch (approx).")
flags.DEFINE_integer("sam2_y_prompts_rs", 10, "Total RS keyframes to annotate per batch (approx).")

# 依赖
import json
import cv2
import traceback
from pathlib import Path
from collections import defaultdict
# ===========================================================================


def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
    obs, info = env.reset()
    if getattr(env.unwrapped, "_cur_ep_start", None) is None:
        env.unwrapped._cur_ep_start = env.unwrapped.global_frame_id
        print(f"[record_demos][mark] start={env.unwrapped._cur_ep_start} @first reset")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0

    # ===================== NEW：与在线训练一致的 SAM2 批处理状态 =====================
    batch_frame_ranges = []      # 收集的 episode 帧区间列表 [(s,e), ...]
    episodes_in_batch = 0       # 该批次已收集的 episodes 个数

    # 目标掩码尺寸（兼容：可能嵌套在 images 里，也可能是扁平）
    spaces = getattr(env.observation_space, "spaces", {})
    def _shape_of_mask(sp):
        shp = tuple(sp.shape)
        if len(shp) >= 3:
            H, W, C = shp[-3:] 
            return int(H), int(W)
        elif len(shp) == 2:
            H, W = shp
            return int(H), int(W)
        else:
            return 128, 128
        
    if "images" in spaces and hasattr(spaces["images"], "spaces") and "gaze_mask" in spaces["images"].spaces:
        _H, _W = _shape_of_mask(spaces["images"].spaces["gaze_mask"])
    elif "gaze_mask" in spaces:
        _H, _W = _shape_of_mask(spaces["gaze_mask"])
    else:
        _H, _W = 128, 128  # 兜底默认

    def _resolve_frame_idx(info_dict, frame_root: Path):
        """把 info['frame_idx'] 对齐到实际存在的 frame_k 目录。优先 k-1，其次 k（与在线训练一致的容错）。"""
        idx = info_dict.get("frame_idx", None)
        if idx is None:
            return None
        for cand in (int(idx) - 1, int(idx)):
            if (frame_root / f"frame_{cand}").exists():
                return cand
        return int(idx) - 1

    def _load_mask_bgr_for_frame(fid: int, frame_root: Path, out_shape_hw=(128, 128)):
        """根据 gaze_contact.json 的 class_id 选 rs_mask_obj{0/1}.png，返回 BGR 三通道、resize 到 out_shape_hw。"""
        fdir = frame_root / f"frame_{fid}"
        j = fdir / "gaze_contact.json"
        if not j.exists():
            return None
        try:
            d = json.loads(j.read_text())
            cid = d.get("class_id", None)
        except Exception:
            cid = None
        if cid is None or cid not in (0, 1):
            return None
        mpath = fdir / f"rs_mask_obj{int(cid)}.png"
        if not mpath.exists():
            return None
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        H, W = out_shape_hw
        return cv2.resize(m3, (W, H), interpolation=cv2.INTER_NEAREST)

    def _ranges_to_fid_set(ranges):
        s = set()
        for (a, b) in ranges:
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            s.update(range(a, b + 1))
        return s

    def _run_sam2_v2_and_patch(transitions_list, ranges):
        """对 ranges 覆盖的帧跑 SAM2 v2（先 ET 后 RS，关键帧不联动），并将生成的 RS 掩码写回 transitions（与在线一致）。"""
        if not ranges:
            print("[record_demos][SAM2] empty ranges; skip")
            return
        root_env = env.unwrapped  
        print(f"[record_demos][SAM2] run v2 on ranges={ranges}")
        
        
        try:
            print("[record_demos][SAM2] pausing any existing display windows ..."); sys.stdout.flush()
            if hasattr(root_env, "pause_display"):
                root_env.pause_display()
            else:
                cv2.destroyAllWindows()
        except Exception:
            pass
        # 1) 跑 v2
        print("[record_demos][SAM2] calling run_inline_sam2_and_label_v2 ..."); sys.stdout.flush()
        try:
            if hasattr(root_env, "ui_pause_event"):
                print("[record_demos][SAM2] pausing RGB viewer ...")
                root_env.ui_pause_event.set()
            time.sleep(0.05)
            # env.run_inline_sam2_and_label_v2(
            root_env.run_inline_sam2_and_label_v2(
                frame_ranges=ranges,
                y_prompts_et=FLAGS.sam2_y_prompts_et,
                y_prompts_rs=FLAGS.sam2_y_prompts_rs,
                rs_select_uses_same_keyset=False,
                random_seed=42,
            )
            print("[record_demos][SAM2] v2 labeling finished")
        except Exception as e:
            print(f"[record_demos][WARN] SAM2 v2 labeling failed for batch: {e}")
            traceback.print_exc()
            try:
                if hasattr(root_env, "resume_display"):
                    root_env.resume_display()
            except Exception:
                pass
            return
        finally:
            if hasattr(root_env, "ui_pause_event"):
                print("[record_demos][SAM2] resume RGB viewer")
                root_env.ui_pause_event.clear()
            time.sleep(0.05)
            try:
                if hasattr(root_env, "resume_display"):
                    root_env.resume_display()
            except Exception:
                pass
        time.sleep(0.2)

        # 2) 回填 gaze_mask
        
        # frame_root = Path(env.frame_root) if hasattr(env, "frame_root") else Path(getattr(env, "frame_save_path", "./"))
        frame_root = Path(getattr(root_env, "frame_root", getattr(root_env, "frame_save_path", "./")))
        fid_set = _ranges_to_fid_set(ranges)
        patched = 0
        for tr in transitions_list:
            info = tr.get("infos", {})
            fid = _resolve_frame_idx(info, frame_root)
            if fid is None or fid not in fid_set:
                continue
            m_cur = _load_mask_bgr_for_frame(fid, frame_root, (_H, _W))
            if m_cur is None:
                continue
            m_next = _load_mask_bgr_for_frame(fid + 1, frame_root, (_H, _W)) or m_cur

            # 强制把 gaze_mask 写进 observations/next_observations 的 images 子键里
            tr.setdefault("observations", {}).setdefault("images", {})
            tr.setdefault("next_observations", {}).setdefault("images", {})
            tr["observations"]["images"]["gaze_mask"] = m_cur
            tr["next_observations"]["images"]["gaze_mask"] = m_next
            patched += 1

        print(f"[record_demos][SAM2] Patched transitions in batch: {patched}")
    # ===========================================================================

    try:
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape)
            actions[1] = -0.1
            next_obs, rew, done, truncated, info = env.step(actions)
            print("reward = ", rew)
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
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
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done:
                if info["succeed"]:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)

                # === NEW（仅添加）：登记该 episode 的帧范围，并按批次触发 SAM2 v2，与在线训练一致 ===
                # if FLAGS.sam2_enable and hasattr(env, "end_episode_and_collect"):
                # if FLAGS.sam2_enable and hasattr(env.unwrapped, "end_episode_and_collect"):
                if FLAGS.sam2_enable:
                    try:
                        # rng = env.end_episode_and_collect()
                        rng = env.unwrapped.end_episode_and_collect()
                        print(f"[record_demos][SAM2] end_episode_and_collect() -> {rng}")
                        if rng is not None:
                            batch_frame_ranges.append(tuple(rng))
                            episodes_in_batch += 1
                            print(f"[record_demos][SAM2] collected episode range: {rng}  ({episodes_in_batch}/{FLAGS.sam2_batch_episodes})")
                            # 凑满一批就立刻触发与在线训练一致的 v2 流程，并写回当批 transitions
                            if episodes_in_batch >= FLAGS.sam2_batch_episodes:
                                print(f"[record_demos][SAM2] triggering _run_sam2_v2_and_patch on {batch_frame_ranges}")
                                _run_sam2_v2_and_patch(transitions, batch_frame_ranges)
                                batch_frame_ranges.clear()
                                episodes_in_batch = 0
                        else:
                            print("[record_demos][SAM2] rng is None (episode had no frames?)")
                    except Exception as e:
                        print(f"[record_demos][WARN] end_episode_and_collect failed: {e}")

                trajectory = []
                returns = 0
                input("reset env")
                obs, info = env.reset()
                env.unwrapped._cur_ep_start = env.unwrapped.global_frame_id
                print(f"[record_demos][mark] start={env.unwrapped._cur_ep_start} @per-episode reset"); sys.stdout.flush()
    finally:
        env.save_all_data_on_exit()
        if hasattr(env, "keyboard_process") and env.keyboard_process.is_alive():
            print("Shutting down keyboard process...")
            env.keyboard_process.terminate()
            env.keyboard_process.join()

    # === NEW（仅添加）：脚本结束前，处理剩余未满一批的 episodes（与在线训练一致的尾批次）===
    if FLAGS.sam2_enable and len(batch_frame_ranges) > 0:
        _run_sam2_v2_and_patch(transitions, batch_frame_ranges)
        batch_frame_ranges.clear()
        episodes_in_batch = 0

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)
