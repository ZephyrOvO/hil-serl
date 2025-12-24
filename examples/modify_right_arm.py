#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import cv2
import shutil

import pinocchio as pin
from transforms3d.quaternions import mat2quat, quat2mat

from tactexo_finger_ik import FingerIKSolver

# =========================
# 配置区（按需修改）
# =========================
DATASET_FOLDERS = [
    "/mnt/data3/hcj/recorded_data/Ball_Pick/New_Ball_Pick_08_14_00_hcj"
]
DATA_FRAME_PATH = "/mnt/data3/hcj/recorded_data/Ball_Pick/New_Ball_Pick_08_14_00_hcj"

TACEXO_URDF_PATH = "/home/ruiqiang/workspaces/HK_TacExo/ros2_ws/src/TacExo_URDF_if_correct_base/urdf/TacExo_URDF.urdf"
HAND_URDF_PATH   = "/home/ruiqiang/workspaces/HK_TacExo/ros2_ws/src/data_recorder/leap_hand_mesh/robot_pybullet.urdf"
HAND_MESH_DIR    = "/home/ruiqiang/workspaces/HK_TacExo/ros2_ws/src/data_recorder/leap_hand_mesh/"

RS_TYPE = "rs455"

# 顺序: [thumb, index, middle, ring, pinky]， 1=启用, 0=置零
FINGER_ENABLE = [1, 1, 1, 0, 0]

# =========================
# 与原策略一致的几何/外参
# =========================
def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return Rz @ Ry @ Rx

def matrix_to_pos_quat(T):
    pos = T[:3,3]
    q = mat2quat(T[:3,:3])  # [w,x,y,z]
    return np.concatenate([pos, q], axis=0)

LEAP_CENTER_TRANSLATION = [0.03424224387, 0.060095249652862544332, 0.04527759642]
LEAP_CENTER_RPY         = [3.14, 0.0, 1.570796327]
LEAP_CENTER_TF = np.eye(4)
LEAP_CENTER_TF[:3,:3] = rpy_to_matrix(*LEAP_CENTER_RPY)
LEAP_CENTER_TF[:3, 3] = LEAP_CENTER_TRANSLATION

LEAP_TO_VIVE = np.eye(4)
LEAP_TO_VIVE[:3,:3] = rpy_to_matrix(1.57, 0.0, 0.0)
LEAP_TO_VIVE[:3, 3] = [0.02, 0.04, 0.01]

TACTEXO_FINGER_TO_LEAP = np.eye(4)
TACTEXO_FINGER_TO_LEAP[:3,:3] = rpy_to_matrix(-1.57, 1.57, 0.0)

DEFAULT_RIGHT_GLOVE_TO_VIVE = np.eye(4)

T_vive_to_d457 = np.array([
    [-1.00000000e+00,  1.22464680e-16,  1.36346372e-26, -1.52500000e-02],
    [-6.12323400e-17, -5.00000000e-01,  8.66025404e-01,  6.06829947e-02],
    [ 1.06057524e-16,  8.66025404e-01,  5.00000000e-01, -1.63260300e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
])
DEFAULT_CAM_TO_VIVE_RS455 = np.linalg.inv(T_vive_to_d457)

cam_front_translation = [1.2367936975704506, 0.032497565951025945, 0.5359742126690214]
cam_front_quaternion  = [0.012480230443529135, 0.27804828390806924, -0.026321127948753298, -0.960125301139054]  # [w, x, y, z]
CamR = quat2mat(cam_front_quaternion)
CAMFRONT2ROBOT = np.eye(4)
CAMFRONT2ROBOT[:3,:3] = CamR
CAMFRONT2ROBOT[:3, 3] = cam_front_translation
T_axis_swap = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
CAMFRONT2ROBOT = CAMFRONT2ROBOT @ T_axis_swap

def load_txt_T(file_path, default_T):
    return np.loadtxt(file_path) if os.path.exists(file_path) else default_T

def process_hand_joints(tactexo_model, tactexo_data, joint_positions, glove_to_vive):
    q = joint_positions
    pin.forwardKinematics(tactexo_model, tactexo_data, q)
    pin.updateFramePlacements(tactexo_model, tactexo_data)

    base_frame   = 'T265_Link'
    thumb_frame  = 'fingertip_tactile'
    index_frame  = 'i_fingertip_t'
    middle_frame = 'md_fingertip_t'

    b_id = tactexo_model.getFrameId(base_frame)
    t_id = tactexo_model.getFrameId(thumb_frame)
    i_id = tactexo_model.getFrameId(index_frame)
    m_id = tactexo_model.getFrameId(middle_frame)

    if min(b_id, t_id, i_id, m_id) < 0:
        raise RuntimeError("URDF 中找不到某些 frame，请检查命名。")

    oMb = tactexo_data.oMf[b_id]
    oMt = tactexo_data.oMf[t_id]
    oMi = tactexo_data.oMf[i_id]
    oMm = tactexo_data.oMf[m_id]

    bMt = oMb.inverse() * oMt
    bMi = oMb.inverse() * oMi
    bMm = oMb.inverse() * oMm

    thumb_pose  = glove_to_vive @ bMt.homogeneous
    index_pose  = glove_to_vive @ bMi.homogeneous
    middle_pose = glove_to_vive @ bMm.homogeneous
    return [thumb_pose, index_pose, middle_pose]

def main():
    tactexo_model = pin.buildModelFromUrdf(TACEXO_URDF_PATH)
    tactexo_data  = tactexo_model.createData()

    ik_solver = FingerIKSolver(
        urdf_model_path=HAND_URDF_PATH,
        mesh_dir=HAND_MESH_DIR,
        enable_visualization=False,
        visualization_type='mesh'
    )

    cam_to_vive_file_path         = os.path.join(DATA_FRAME_PATH, 'cam_to_vive.txt')
    right_glove_to_vive_file_path = os.path.join(DATA_FRAME_PATH, 'right_glove_to_vive.txt')

    if RS_TYPE == "rs455":
        CAM_TO_VIVE = DEFAULT_CAM_TO_VIVE_RS455
    else:
        CAM_TO_VIVE = load_txt_T(cam_to_vive_file_path, np.eye(4))

    RIGHT_GLOVE_TO_VIVE = load_txt_T(right_glove_to_vive_file_path, DEFAULT_RIGHT_GLOVE_TO_VIVE)

    for ds in DATASET_FOLDERS:
        frame_dirs = sorted(
            [os.path.join(ds, d) for d in os.listdir(ds)
             if os.path.isdir(os.path.join(ds, d)) and re.match(r'^frame_\d+$', d)],
            key=lambda p: int(re.findall(r'\d+', os.path.basename(p))[0])
        )

        for fdir in frame_dirs:
            try:
                right_arm_joint_path   = os.path.join(fdir, "right_arm_joint.txt")
                cam_vive_pose_path     = os.path.join(fdir, "vive_pose0.txt")
                right_glove_vive_path  = os.path.join(fdir, "vive_pose1.txt")

                if not (os.path.exists(right_arm_joint_path)
                        and os.path.exists(cam_vive_pose_path)
                        and os.path.exists(right_glove_vive_path)):
                    continue

                # 先读取原始关节数据（计算用）
                original_joint_data = np.loadtxt(right_arm_joint_path)
                cam_vive_pose       = np.loadtxt(cam_vive_pose_path)
                right_glove_vive    = np.loadtxt(right_glove_vive_path)

                # 若没备份，则备份为 *_1.txt
                backup_path = os.path.join(fdir, "right_arm_joint_1.txt")
                if not os.path.exists(backup_path):
                    shutil.copyfile(right_arm_joint_path, backup_path)

                # === 标定链 ===
                if RS_TYPE == "rs455":
                    VIVE_TO_ROBOT = cam_vive_pose @ CAM_TO_VIVE @ np.linalg.inv(CAMFRONT2ROBOT)
                    ROBOT_TO_VIVE = np.linalg.inv(VIVE_TO_ROBOT)
                else:
                    VIVE_TO_ROBOT = np.eye(4)
                    ROBOT_TO_VIVE = np.eye(4)

                # 1) TacExo URDF → 各指末端（在 glove/base）→ 映射到 vive
                fingerposes_vive = process_hand_joints(
                    tactexo_model, tactexo_data, original_joint_data, RIGHT_GLOVE_TO_VIVE
                )

                # 2) 映射到机器人系
                right_hand_wrist_pose  = ROBOT_TO_VIVE @ right_glove_vive @ RIGHT_GLOVE_TO_VIVE
                right_hand_thumb_pose  = ROBOT_TO_VIVE @ right_glove_vive @ fingerposes_vive[0]
                right_hand_index_pose  = ROBOT_TO_VIVE @ right_glove_vive @ fingerposes_vive[1]
                right_hand_middle_pose = ROBOT_TO_VIVE @ right_glove_vive @ fingerposes_vive[2]

                # 3) LEAP base（机器人系）
                leap_base_pose = right_hand_wrist_pose @ LEAP_TO_VIVE @ LEAP_CENTER_TF

                # 4) 目标在 LEAP 局部
                lcs_thumb  = np.linalg.inv(leap_base_pose) @ (right_hand_thumb_pose  @ TACTEXO_FINGER_TO_LEAP)
                lcs_index  = np.linalg.inv(leap_base_pose) @ (right_hand_index_pose  @ TACTEXO_FINGER_TO_LEAP)
                lcs_middle = np.linalg.inv(leap_base_pose) @ (right_hand_middle_pose @ TACTEXO_FINGER_TO_LEAP)

                # 5) IK
                tgt_pos, tgt_rot = ik_solver.generate_target_poses(ik_solver.model, ik_solver.data)
                tgt_pos['thumb']  = lcs_thumb[:3, 3]
                tgt_pos['index']  = lcs_index[:3, 3]
                tgt_pos['middle'] = lcs_middle[:3, 3]
                tgt_rot['thumb']  = lcs_thumb[:3, :3]
                tgt_rot['index']  = lcs_index[:3, :3]
                tgt_rot['middle'] = lcs_middle[:3, :3]

                q_ik = np.array(ik_solver.compute_ik(tgt_pos, tgt_rot, only_pos=True), dtype=float)

                # 6) finger enable
                idxs = {'index': (0, 4), 'thumb': (4, 8), 'middle': (8, 12), 'ring': (12, 16)}
                enable_map = {'thumb': FINGER_ENABLE[0], 'index': FINGER_ENABLE[1],
                              'middle': FINGER_ENABLE[2], 'ring': FINGER_ENABLE[3]}
                for k, (s, e) in idxs.items():
                    if enable_map[k] == 0:
                        q_ik[s:e] = 0.0

                # 7) 23 维：EE TCP(7) + q_ik(16)
                ee7 = matrix_to_pos_quat(leap_base_pose)  # [x,y,z,qw,qx,qy,qz]
                out23 = np.concatenate([ee7, q_ik], axis=0)

                # 8) 覆盖写回（每行一个数，共 23 行）
                np.savetxt(right_arm_joint_path, out23.reshape(-1, 1), fmt="%.8f")

            except Exception as e:
                print(f"[WARN] 跳过 {fdir}: {e}")

    print("完成：每个 frame_*/right_arm_joint.txt 已先备份为 right_arm_joint_1.txt（若不存在），再覆盖为 23 行新结果。")

if __name__ == "__main__":
    main()
