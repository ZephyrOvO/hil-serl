import copy
import json
import cv2
import os
import numpy as np
import gymnasium as gym
# --- robust import for kinematics_utils (same folder as this file) ---
import os, importlib.util

try:
    from examples.utils import kinematics_utils  # type: ignore
except ModuleNotFoundError:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    KIN_PATH = os.path.join(THIS_DIR, "kinematics_utils.py")
    if not os.path.exists(KIN_PATH):
        raise ModuleNotFoundError(f"Cannot locate kinematics_utils.py at {KIN_PATH}")
    spec = importlib.util.spec_from_file_location("kinematics_utils", KIN_PATH)
    kinematics_utils = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(kinematics_utils)
# --- end robust import ---


import re
from collections import deque
from scipy.spatial.transform import Rotation as R

palm_lower2denso_end_tf = np.array([
    [1.00000000e+00, -3.26589794e-07, 0.00000000e+00, -6.00952496e-02],
    [-3.26589379e-07, -9.99998732e-01, 1.59265292e-03, -3.39726879e-02],
    [-5.20144187e-10, -1.59265292e-03, -9.99998732e-01, -1.69276725e-01],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

gripper_open_joint = [
    2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
    2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
    3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    4.312019824981689453, 3.905515193939208984, 3.374757766723632812, 3.597184896469116211    #thumb
]

gripper_close_joint = [
    3.546563625335693359, 4.127942085266113281, 3.413689804077148438, 3.641670465469360352,
    3.626330614089965820, 3.529689788818359375, 2.931437253952026367, 3.782796621322631836,
    3.838019847869873047, 3.532757759094238281, 3.535825729370117188, 3.413107156753540039,
    4.661767482757568359, 3.366175127029418945, 3.260291767120361328, 3.566796636581420898
]

class ObsHistoryBuffer:
    # def __init__(self, obs_horizon=3, image_keys=("front_camera", "side_camera"), proprio_key="state"):
    def __init__(self, obs_horizon=3, image_keys=("front_camera","gaze_mask"), proprio_key="state"):
        self.obs_horizon = obs_horizon
        self.image_keys = image_keys
        self.proprio_key = proprio_key
        self.buffer = deque(maxlen=obs_horizon)

    def reset(self, first_obs):
        self.buffer.clear()
        for _ in range(self.obs_horizon):
            self.buffer.append(copy.deepcopy(first_obs))

    def append(self, obs):
        self.buffer.append(copy.deepcopy(obs))

    def get_stacked_obs(self):
        stacked_obs = {}
        for key in self.image_keys:
            frames = [o[key] for o in self.buffer]  # list of (H, W, 3)
            stacked_obs[key] = np.concatenate(frames, axis=-1)  # (H, W, 9)

        if self.proprio_key is not None:
            vecs = [o[self.proprio_key] for o in self.buffer]  # list of (23,)
            stacked_obs[self.proprio_key] = np.concatenate(vecs, axis=-1)  # (69,)

        return stacked_obs
    
    def get_success_fail_obs(self):
        stacked_obs = {}
        for key in self.image_keys:
            frames = [o[key] for o in self.buffer]  # list of (H, W, 3)
            stacked_obs[key] = np.stack(frames, axis=0)

        if self.proprio_key is not None:
            vecs = [o[self.proprio_key] for o in self.buffer]
            stacked_obs[self.proprio_key] = np.stack(vecs, axis=0)

        return stacked_obs
    

# 从当前帧目录读取并返回 注视目标的二值掩码图，作为 obs["gaze_mask"]
def _require_gaze_mask(frame_path, target_size=(128, 128)):
    gaze_json = os.path.join(frame_path, "gaze_contact.json")
    if not os.path.exists(gaze_json):
        raise FileNotFoundError(f"Missing gaze_contact.json at {frame_path}")
    with open(gaze_json, "r") as f:
        g = json.load(f)

    rs_id = g.get("rs_object_id", None)
    if rs_id not in (0, 1):  # None 或 其它值，一律输出全零掩码
        m = np.zeros((*target_size, 1), dtype=np.uint8)
    else:
        mask_name = "rs_mask_obj0.png" if rs_id == 0 else "rs_mask_obj1.png"
        mask_path = os.path.join(frame_path, mask_name)
        if not os.path.exists(mask_path):
            # 找不到就用全零
            m = np.zeros((*target_size, 1), dtype=np.uint8)
        else:
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = np.zeros((*target_size, 1), dtype=np.uint8)
            else:
                m = cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST)
                m = m[..., None]  # (H,W,1)

    # 关键：把 1 通道复制成 3 通道 -> (H,W,3)
    m3 = np.repeat(m, 3, axis=-1).astype(np.uint8)
    return m3


# === NEW: gaze 标签提取（命中类别 + 归一化质心 + 可选热图） ============================
def _compute_centroid_from_mask(mask: np.ndarray) -> tuple[float, float] | None:
    """
    mask: (H,W) uint8, 前景>0
    return: (cx, cy) in pixel coords (x->W, y->H), 或 None(没有前景)
    """
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())
    return (cx, cy)

def _maybe_make_gaussian_heatmap(H: int, W: int, cx: float, cy: float, sigma: float = 7.0) -> np.ndarray:
    """
    生成以像素质心(cx,cy)为中心的高斯热图，范围[0,1]，shape=(H,W).
    """
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    return g

def get_gaze_label(frame_path: str, out_size: tuple[int, int] = (128, 128), make_heatmap: bool = False):
    """
    从 frame_* 目录读取 gaze_contact.json 和 rs_mask_obj{0/1}.png，
    返回 (hit_class, gaze_xy, gaze_heatmap or None).
    - hit_class: 0=obj0, 1=obj1, 2=none
    - gaze_xy:   [x_norm,y_norm] ∈ [0,1]^2 （none时返回(0,0)；训练时会被mask忽略）
    - gaze_heatmap: (H,W) float32 in [0,1]；none时全0；仅当 make_heatmap=True 才返回真实热图
    """
    gaze_json = os.path.join(frame_path, "gaze_contact.json")
    Ht, Wt = int(out_size[0]), int(out_size[1])
    # 默认 none 情况
    hit_class = 2
    gaze_xy = np.array([0.0, 0.0], dtype=np.float32)
    gaze_hm = np.zeros((Ht, Wt), dtype=np.float32) if make_heatmap else None

    if not os.path.exists(gaze_json):
        return hit_class, gaze_xy, gaze_hm

    try:
        with open(gaze_json, "r") as f:
            g = json.load(f)
        rs_id = g.get("rs_object_id", None)
    except Exception:
        rs_id = None

    if rs_id not in (0, 1):
        # none → 直接返回默认
        return hit_class, gaze_xy, gaze_hm

    # 命中 obj0/obj1
    hit_class = int(rs_id)
    mname = f"rs_mask_obj{hit_class}.png"
    mpath = os.path.join(frame_path, mname)
    if not os.path.exists(mpath):
        return 2, gaze_xy, gaze_hm  # 没有掩码当 none

    mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 2, gaze_xy, gaze_hm

    # 统一 resize 到 out_size
    mask_r = cv2.resize(mask, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    cen = _compute_centroid_from_mask(mask_r)
    if cen is None:
        return 2, gaze_xy, gaze_hm

    cx, cy = cen  # 像素坐标
    # 归一化到 [0,1]
    gaze_xy = np.array([cx / max(Wt - 1, 1), cy / max(Ht - 1, 1)], dtype=np.float32)

    if make_heatmap:
        gaze_hm = _maybe_make_gaussian_heatmap(Ht, Wt, cx, cy, sigma=7.0).astype(np.float32)

    return hit_class, gaze_xy, gaze_hm
# ====================================================================

# === NEW: 为 gaze 数据集导出准备的 reader =========================================
def read_gaze_example(frame_path: str,
                      front_resize: tuple[int, int] = (128, 128),
                      return_heatmap: bool = False):
    """
    从 frame 目录构造用于 gaze 训练的一条样本：
    returns:
      observations: {"front_camera": (H,W,3) uint8}
      labels:       {"gaze_hit_class": int(0/1/2), "gaze_xy": (2,) float32, ["gaze_heatmap":(H,W)float32]}
    """
    color_image_path = os.path.join(frame_path, "color_image.jpg")
    img = cv2.imread(color_image_path)
    if img is None:
        return None

    # BGR->RGB + resize 到与训练一致
    Ht, Wt = front_resize
    img_r = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_LINEAR)[..., ::-1]  # (H,W,3) uint8

    hit_class, gaze_xy, gaze_hm = get_gaze_label(frame_path, out_size=(Ht, Wt), make_heatmap=return_heatmap)

    observations = {"front_camera": img_r}
    labels = {"gaze_hit_class": int(hit_class), "gaze_xy": gaze_xy.astype(np.float32)}
    if return_heatmap:
        labels["gaze_heatmap"] = gaze_hm  # (H,W) float32, none时全0
    return observations, labels
# ====================================================================





def get_frame_data(frame_path, robot_urdf_path,  next_frame_path=None, enable_tactile=False):
    color_image_path = os.path.join(frame_path, "color_image.jpg")
    index_heat_map_path = os.path.join(frame_path, "index_heat_map.jpg")
    thumb_heat_map_path = os.path.join(frame_path, "thumb_heat_map.jpg")
    middle_heat_map_path = os.path.join(frame_path, "middle_heat_map.jpg")
    # color_image_path2 = os.path.join(frame_path, "color_image2.jpg")
    # depth_image_path = os.path.join(frame_path, "depth_image.png")
    # depth_image_path2 = os.path.join(frame_path, "depth_image2.png")
    color_image = cv2.imread(color_image_path) if os.path.exists(color_image_path) else None
    # color_image2 = cv2.imread(color_image_path2) if os.path.exists(color_image_path) else None
    # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED) if os.path.exists(depth_image_path) else None
    # depth_image2 = cv2.imread(depth_image_path2, cv2.IMREAD_UNCHANGED) if os.path.exists(depth_image_path) else None
    index_heat_map_image = cv2.imread(index_heat_map_path) if os.path.exists(index_heat_map_path) else None
    thumb_heat_map_image = cv2.imread(thumb_heat_map_path) if os.path.exists(thumb_heat_map_path) else None
    middle_heat_map_image = cv2.imread(middle_heat_map_path) if os.path.exists(middle_heat_map_path) else None
    

    index_heat_map_image = cv2.resize(index_heat_map_image, (128, 128), interpolation=cv2.INTER_LINEAR)
    thumb_heat_map_image = cv2.resize(thumb_heat_map_image, (128, 128), interpolation=cv2.INTER_LINEAR)
    middle_heat_map_image = cv2.resize(middle_heat_map_image, (128, 128), interpolation=cv2.INTER_LINEAR)
    heatmap_canvas = cv2.hconcat([thumb_heat_map_image, index_heat_map_image, middle_heat_map_image])

    gaze_mask = _require_gaze_mask(frame_path)  # (128,128,1) uint8

    joint_file_path = os.path.join(frame_path, "right_arm_joint.txt")
    record_success_failed_file = os.path.join(frame_path, "is_record_success.txt")
    hand_joint = None
    is_record_success = np.loadtxt(record_success_failed_file, dtype=int)
    
    if os.path.exists(joint_file_path):
        with open(joint_file_path, "r") as f:
            all_joint_values = np.array([float(x.strip()) for x in f.readlines()])
            hand_joint = all_joint_values[6:]
            
            
    if next_frame_path is not None:
        next_joint_file_path = os.path.join(next_frame_path, "right_arm_joint.txt")
        if os.path.exists(next_joint_file_path):
            with open(next_joint_file_path, "r") as f:
                next_all_joint_values = np.array([float(x.strip()) for x in f.readlines()])
                next_hand_joint = next_all_joint_values[6:]
        
        open_j  = np.array(gripper_open_joint,  dtype=np.float32)
        close_j = np.array(gripper_close_joint, dtype=np.float32)
        gripper_direction = np.sign(close_j - open_j)
        gripper_direction[gripper_direction == 0] = 1.0
        max_gripper_step = np.abs(close_j - open_j) / 10.0
        max_gripper_step = np.clip(max_gripper_step, 1e-6, None)
        
        hand_state = float(np.clip(
            np.dot(next_hand_joint - hand_joint,
                   max_gripper_step * gripper_direction)
            / (np.dot(max_gripper_step * gripper_direction,
                      max_gripper_step * gripper_direction) + 1e-8),
            -1.0, 1.0
        ))
        
    tcp_pos, tcp_ori = kinematics_utils.comupute_forward_kinematics(all_joint_values, robot_urdf_path)
    tcp_pos, tcp_ori = kinematics_utils.apply_transformation(tcp_pos, tcp_ori, palm_lower2denso_end_tf)

        
    #if os.path.exists(joint_file_path):

    #    with open(joint_file_path, "r") as f:
    #        all_joint_values = np.array([float(x.strip()) for x in f.readlines()])
            # Change the order of robot arm joint data

            # print("all_joint_values ori = ", all_joint_values)
            # wrist_joint_index = [2,0,1,3,4,5]
            # all_joint_values[:6] = all_joint_values[wrist_joint_index]

    #        hand_joint = all_joint_values[6:]
            # print("all_joint_values = ", all_joint_values)
            # input("enter")

    #open_j  = np.array(gripper_open_joint,  dtype=np.float32)
    #close_j = np.array(gripper_close_joint, dtype=np.float32)
    #curr    = np.array(hand_joint,          dtype=np.float32)
    #if vals.size != 23:
    #    raise ValueError(f"right_arm_joint.txt must have 23 values, got {vals.size}")



    #tcp_pos  = vals[0:3]
    #tcp_quat = vals[3:7]           # wxyz
    #hand_joint = vals[7:]
    #if hand_joint.size != 16:
    #    raise ValueError(f"Expected 16 hand joints, got {hand_joint.size}")


    #delta = close_j - open_j
    #denominator = np.dot(delta, delta) + 1e-8
    #hand_state = float(np.clip(np.dot(curr - open_j, delta) / denominator, 0.0, 1.0))
    # print("tcp_pos = ", tcp_pos)
    # print("tcp_ori = ", tcp_ori)
    # ori_index = [3, 0, 1, 2]
    # tcp_ori = np.array(tcp_ori)[ori_index]
    # tcp_pos, tcp_ori = kinematics_utils.apply_transformation(tcp_pos, tcp_ori, palm_lower2denso_end_tf)
    # print("tcp_pos 1= ", tcp_pos)
    # print("tcp_ori 1= ", tcp_ori)
    # input("debug")

    # state_flattened = np.concatenate([
    #     np.array(tcp_pos, dtype=np.float32).flatten(),
    #     np.array(tcp_ori, dtype=np.float32).flatten(),
    #     np.array(hand_joint, dtype=np.float32).flatten()
    # ])

    state_flattened = np.concatenate([
        np.array(tcp_pos, dtype=np.float32).flatten(),
        np.array(tcp_ori, dtype=np.float32).flatten(),
        np.array(hand_state, dtype=np.float32).flatten(),
    ], axis=0)

    resized_image = cv2.resize(color_image, (128,128))
    # resized_image2 = cv2.resize(color_image2, (320,240))rea
    front_camera_image = resized_image[..., ::-1]
    # side_camera_image = resized_image2[..., ::-1]


    obs = {
        "front_camera": front_camera_image,
        "gaze_mask": gaze_mask,
        "state": state_flattened
    }
    if enable_tactile:
        obs["tactile_data"] = heatmap_canvas

    return obs, int(is_record_success)

def read_data(robot_urdf_path, is_evaluate_classifier=False, enable_tactile=False):
    data = []
    clip_ranges = []
    global_idx = 0
    # action_space = gym.spaces.Box(
    #     np.ones((23,), dtype=np.float32) * -1,
    #     np.ones((23,), dtype=np.float32),
    # )
    low  = np.concatenate([np.ones(6, dtype=np.float32) * -1, [0]])
    high = np.ones(7, dtype=np.float32)
    action_space = gym.spaces.Box(low, high, dtype=np.float32)
    
    actions = np.zeros(action_space.sample().shape)
    if is_evaluate_classifier:
        data_dir = "/mnt/data3/hcj/recorded_data/test_data/"
    else:
        data_dir = "/home/qiangqiang/workspaces/data/2025-4-3/demo_data"
    for collect_data_dir in sorted(os.listdir(data_dir)):
        collect_data_path = os.path.join(data_dir, collect_data_dir)
        if not os.path.isdir(collect_data_path):
            continue

        # 获取 collect_data_path 目录下所有名为 frame_xxx 的子目录，并按照 xxx 数值大小排序。
        frame_dirs = sorted(
            [os.path.join(collect_data_path, d) for d in os.listdir(collect_data_path) if os.path.isdir(os.path.join(collect_data_path, d))],
            key=lambda folder: int(re.search(r'frame_(\d+)', os.path.basename(folder)).group(1)) if re.search(r'frame_(\d+)', os.path.basename(folder)) else float('inf')
        )
        clip_marks_json = os.path.join(collect_data_path, 'clip_marks.json')
        with open(clip_marks_json, 'r') as f:
            clip_marks = json.load(f)


        for clip in clip_marks:
            start_frame = int(clip['start'].split('_')[-1])
            end_frame = int(clip['end'].split('_')[-1])
            clip_start_idx = global_idx
            
            for i in list(range(start_frame, end_frame+1)):
            # for i in range(len(frame_dirs) - 1):
                current_frame_path = os.path.join(collect_data_path, frame_dirs[i])
                if i == end_frame:
                    next_frame_path = current_frame_path
                else:
                    next_frame_path = os.path.join(collect_data_path, frame_dirs[i + 1])

                if not os.path.isdir(current_frame_path) or not os.path.isdir(next_frame_path):
                    continue


                obs, is_record_success= get_frame_data(current_frame_path, robot_urdf_path, enable_tactile)
                if i == end_frame:
                    next_obs = obs
                else:
                    next_obs, _ = get_frame_data(next_frame_path, robot_urdf_path, enable_tactile)
                # print("next_obs['state'][3:7] = ", next_obs["state"][3:7])

                delta_pos = next_obs["state"][:3] - obs["state"][:3]
                actions[:3] = delta_pos

                current_quat = obs["state"][3:7]  # wxyz
                next_quat = next_obs["state"][3:7]

                current_euler = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]]).as_euler("xyz")
                next_euler = R.from_quat([next_quat[1], next_quat[2], next_quat[3], next_quat[0]]).as_euler("xyz")

                delta_euler = next_euler - current_euler
                actions[3:6] = delta_euler

                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        next_observations=next_obs,
                        actions=actions,
                        is_record_success=is_record_success,
                        rewards=0,
                        masks=1.0,
                        dones=0,
                    )
                )
                data.append(transition)
                global_idx += 1
            clip_end_idx = global_idx - 1 
            clip_ranges.append((clip_start_idx, clip_end_idx))

    return data, clip_ranges
