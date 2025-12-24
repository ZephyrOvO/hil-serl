import os
import jax
import jax.numpy as jnp
import numpy as np
# 新增
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict as GymDict, Box as GymBox

class AddKeyToSpace(gym.ObservationWrapper):
    """确保空间里声明了指定的 image key（否则网络构建时拿不到 shape）。"""
    def __init__(self, env, key, shape, dtype=np.uint8, low=0, high=255):
        super().__init__(env)
        self._key = key
        self._shape = shape
        self._dtype = dtype
        self._low = low
        self._high = high
        spaces = dict(self.observation_space.spaces)
        spaces[self._key] = GymBox(low=self._low, high=self._high, shape=self._shape, dtype=self._dtype)
        self.observation_space = GymDict(spaces)

    def observation(self, obs):
        if self._key not in obs or obs[self._key] is None:
            obs[self._key] = np.zeros(self._shape, dtype=self._dtype)
        return obs




class EnsureChannelMultipleOf3(gym.ObservationWrapper):
    """如果通道数不是 3 的倍数，在通道维重复/填充到最近的 3 的倍数。"""
    def __init__(self, env, image_keys):
        super().__init__(env)
        self.image_keys = tuple(image_keys)
        spaces = dict(self.observation_space.spaces)
        new_spaces = {}
        for k, space in spaces.items():
            if k in self.image_keys:
                shape = list(space.shape)
                if len(shape) == 3:
                    H, W, C = shape
                    newC = ((C + 2) // 3) * 3
                    shape[-1] = newC
                    new_spaces[k] = GymBox(0, 255, shape=tuple(shape), dtype=space.dtype)
                elif len(shape) == 4:
                    T, H, W, C = shape
                    newC = ((C + 2) // 3) * 3
                    shape[-1] = newC
                    new_spaces[k] = GymBox(0, 255, shape=tuple(shape), dtype=space.dtype)
                else:
                    new_spaces[k] = space
            else:
                new_spaces[k] = space
        self.observation_space = GymDict(new_spaces)

    def observation(self, obs):
        for k in self.image_keys:
            if k not in obs:
                continue
            x = obs[k]
            if x is None:
                continue
            if x.ndim == 3:
                H, W, C = x.shape
                if C % 3 != 0:
                    newC = ((C + 2) // 3) * 3
                    reps = (newC + C - 1) // C
                    x = np.repeat(x, reps, axis=-1)[..., :newC]
            elif x.ndim == 4:
                T, H, W, C = x.shape
                if C % 3 != 0:
                    newC = ((C + 2) // 3) * 3
                    reps = (newC + C - 1) // C
                    x = np.repeat(x, reps, axis=-1)[..., :newC]
            obs[k] = x
        return obs



class EnsureStackDim(gym.ObservationWrapper):
    """
    确保所有 image_keys 的张量形状至少是 (1, H, W, C)：
    - 如果原来是 (H,W,C)，就在最前面加一维 -> (1,H,W,C)
    - 如果原来已经是 (T,H,W,C)，保持不变
    """
    def __init__(self, env, image_keys):
        super().__init__(env)
        self.image_keys = tuple(image_keys)
        spaces = dict(self.observation_space.spaces)
        for k in self.image_keys:
            if k not in spaces:
                continue
            shape = spaces[k].shape
            if len(shape) == 3:
                H, W, C = shape
                spaces[k] = GymBox(low=0, high=255, shape=(1, H, W, C), dtype=spaces[k].dtype)
            elif len(shape) == 4:
                pass  # 已经有 T 维
            else:
                raise ValueError(f"Unexpected shape for {k}: {shape}")
        self.observation_space = GymDict(spaces)

    def observation(self, obs):
        for k in self.image_keys:
            if k in obs and obs[k] is not None and obs[k].ndim == 3:
                obs[k] = obs[k][None, ...]
        return obs
    
class EnsureGazeMask(gym.ObservationWrapper):
    """
    确保 obs 中存在 'gaze_mask'；若缺失则补 0 (128,128,3)。
    若为单通道，自动扩成 3 通道；统一 dtype=uint8。
    """
    def __init__(self, env, shape=(128, 128, 3)):
        super().__init__(env)
        self._gaze_shape = shape
        if isinstance(self.observation_space, spaces.Dict):
            spaces_dict = dict(self.observation_space.spaces)
            if "gaze_mask" not in spaces_dict:
                spaces_dict["gaze_mask"] = spaces.Box(
                    low=0, high=255, shape=self._gaze_shape, dtype=np.uint8
                )
            else:
                # 若已有，确保 high/shape 合理（可选）
                pass
            self.observation_space = spaces.Dict(spaces_dict)

    def observation(self, obs):
        m = obs.get("gaze_mask", None)
        if m is None:
            m = np.zeros(self._gaze_shape, dtype=np.uint8)
        else:
            m = np.asarray(m)
            # 灰度 => (H,W,1)
            if m.ndim == 2:
                m = m[..., None]
            # 1 通道 => 3 通道
            if m.shape[-1] == 1:
                m = np.repeat(m, 3, axis=-1)
            # dtype 统一
            if m.dtype != np.uint8:
                # 这里简单截断到 [0,255] 后转 uint8
                m = np.clip(m, 0, 255).astype(np.uint8)
        obs["gaze_mask"] = m
        return obs

import os

from serl_robot_infra.denso_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    KeyboardIntervention,
)

from denso_env.envs.denso_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.tennis_ball_pick.wrapper import RAMEnv

class EnvConfig(DefaultEnvConfig):
    EXP_NAME = "tennis_ball_pick"
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "front_camera": {
            "serial_number": "242422303461",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
        # "side_camera": {
        #     "serial_number": "234222300515",
        #     "dim": (640, 480),
        #     "exposure": 40000,
        # },
    }
    EXTRA_REALSENSE_CAMERAS = {
        "side_camera": {
            "serial_number": "234222300515",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
    }
    IMAGE_CROP = {
        "front_camera": lambda img: img[150:450, 350:1100],
        "side_camera": lambda img: img[100:500, 400:900],
    }
    # TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    TARGET_POSE = np.array([1.55513753, -0.14267503, 0.18153528, -0.03244228, 0.99039508, 0.12396424, -0.05194187])
    # GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    # RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    # ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    # ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    # ACTION_SCALE = (0.01, 0.06, 1)
    ACTION_SCALE = (0.05, 0.05, 0.05)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    REWARD_THRESHOLD = np.array([0.01, 0.005, 0.01, 1, 1, 1])  # [x, y, z, roll, pitch, yaw]
    # COMPLIANCE_PARAM = {
    #     "translational_stiffness": 2000,
    #     "translational_damping": 89,
    #     "rotational_stiffness": 150,
    #     "rotational_damping": 7,
    #     "translational_Ki": 0,
    #     "translational_clip_x": 0.0075,
    #     "translational_clip_y": 0.0016,
    #     "translational_clip_z": 0.0055,
    #     "translational_clip_neg_x": 0.002,
    #     "translational_clip_neg_y": 0.0016,
    #     "translational_clip_neg_z": 0.005,
    #     "rotational_clip_x": 0.01,
    #     "rotational_clip_y": 0.025,
    #     "rotational_clip_z": 0.005,
    #     "rotational_clip_neg_x": 0.01,
    #     "rotational_clip_neg_y": 0.025,
    #     "rotational_clip_neg_z": 0.005,
    #     "rotational_Ki": 0,
    # }
    # PRECISION_PARAM = {
    #     "translational_stiffness": 2000,
    #     "translational_damping": 89,
    #     "rotational_stiffness": 250,
    #     "rotational_damping": 9,
    #     "translational_Ki": 0.0,
    #     "translational_clip_x": 0.1,
    #     "translational_clip_y": 0.1,
    #     "translational_clip_z": 0.1,
    #     "translational_clip_neg_x": 0.1,
    #     "translational_clip_neg_y": 0.1,
    #     "translational_clip_neg_z": 0.1,
    #     "rotational_clip_x": 0.5,
    #     "rotational_clip_y": 0.5,
    #     "rotational_clip_z": 0.5,
    #     "rotational_clip_neg_x": 0.5,
    #     "rotational_clip_neg_y": 0.5,
    #     "rotational_clip_neg_z": 0.5,
    #     "rotational_Ki": 0.0,
    # }
    IS_ARM_ONLY = True
    ENABLE_TACTILE = True
    TACT_BASE_PATH = '/home/ruiqiang/workspaces/HK_TacExo/9DTact/shape_reconstruction/'


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front_camera", "tactile_data", "gaze_mask"]
    # image_keys = ["front_camera", "side_camera"]
    classifier_keys = ["front_camera", "tactile_data"]
    classifier_key_weights = {"front_camera": 1.0, "tactile_data": 0.5}
    state_weights = np.concatenate(
        [
            np.full(6, 1.0, dtype=np.float32),  # arm joints
            np.full(1, 1.0, dtype=np.float32),  # leaphand joints
        ]
    )
    proprio_keys = ["tcp_pos", "tcp_ori", "gripper_pose"]
    # proprio_keys = ["tcp_pos", "tcp_ori"]
    # classifier_keys = ["front_camera", "side_camera"]
    buffer_period = 1000
    checkpoint_period = 1000 
    steps_per_update = 100
    encoder_type = "resnet-pretrained"
    num_stack = 1#
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env_config = EnvConfig()

        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )
        # env = GripperCloseEnv(env)
        if not fake_env:
            env = KeyboardIntervention(env)
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = AddKeyToSpace(env, key="gaze_mask", shape=(1, 128, 128, 1), dtype=np.uint8)
        env = EnsureStackDim(env, self.image_keys)
        env = EnsureChannelMultipleOf3(env, self.image_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = EnsureGazeMask(env, shape=(128, 128, 3))
        if classifier:
            # print("classifier path = ", os.path.abspath("../../classifier_ckpt/"))
            classifier_pick = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("../../classifier_ckpt_pick/"),
            )

            # classifier_place = load_classifier_func(
            #     key=jax.random.PRNGKey(0),
            #     sample=env.observation_space.sample(),
            #     image_keys=self.classifier_keys,
            #     checkpoint_path=os.path.abspath("../../classifier_ckpt/"),
            # )
            classifier_pick = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_pick/"),
            )
            
            classifier_normal = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                image_key_weights=self.classifier_key_weights,
                checkpoint_path=os.path.abspath("/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt"),
            )
            # input("debug")
            def reward_func(obs, is_pick=True):
                # print("classifier obs = ", classifier(obs))
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # if is_pick:
                #     print("classifier = classifier_pick")
                #     classifier = classifier_pick
                # else:
                #     print("classifier = classifier_place")
                #     classifier = classifier_normal
                classifier = classifier_pick
                print("sigmoid(classifier(obs) = ", sigmoid(classifier(obs)))
                # added check for z position to further robustify classifier, but should work without as well
                # return int(sigmoid(classifier(obs)).item() > 0.95)
            
                prob = sigmoid(classifier(obs)).item()
                
                success = prob > 0.3
                # if classifier == classifier_pick:
                #     reward = 0.3 if success else 0
                # else:
                reward = 1 if success else 0
                state = obs["state"]
                ee_pos = state[0, :3] if state.ndim > 1 else state[:3]
                gripper_pose = state[0, -1] if state.ndim > 1 else state[-1]
                # if ee_pos[1] > -0.13 and ee_pos[2] < 0.14:
                #     reward -= 0.01
                # if ee_pos[2] < 0.02:
                #     reward -= 0.05
                # if not is_pick and -0.30 < ee_pos[1] < -0.04 and gripper_pose < 0.8:
                #     reward -= 0.05
                return reward

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env