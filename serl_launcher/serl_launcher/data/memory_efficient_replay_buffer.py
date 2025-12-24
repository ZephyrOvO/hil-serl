import copy
from typing import Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict, _sample
from serl_launcher.data.replay_buffer import ReplayBuffer
from flax.core import frozen_dict
from gymnasium.spaces import Box


class MemoryEfficientReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        include_next_actions: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
    ):
        self.pixel_keys = pixel_keys

        observation_space = copy.deepcopy(observation_space)
        self._num_stack = None
        for pixel_key in self.pixel_keys:
            pixel_obs_space = observation_space.spaces[pixel_key]
            shape = pixel_obs_space.shape

            # ---- 通用解析：把最后 3 维当 (H,W,C)，其余前置维折叠为 T_total ----
            if len(shape) >= 3:
                H, W, C = shape[-3], shape[-2], shape[-1]
                if len(shape) == 3:
                    T_total = 1
                else:
                    T_total = int(np.prod(shape[:-3]))
            else:
                raise ValueError(f"Unexpected pixel obs shape {shape} for key {pixel_key}")

            # 取第一帧的 low/high，得到 (H,W,C) 形状
            if len(shape) == 3:
                low  = pixel_obs_space.low       # (H,W,C)
                high = pixel_obs_space.high      # (H,W,C)
            else:
                # 用 (0,...,0) 索引到第一帧
                idx0 = (0,) * (len(shape) - 3)
                low  = pixel_obs_space.low[idx0]   # (H,W,C)
                high = pixel_obs_space.high[idx0]  # (H,W,C)

            if self._num_stack is None:
                self._num_stack = T_total
            else:
                assert self._num_stack == T_total, (
                    f"Inconsistent T across pixel keys: {self._num_stack} vs {T_total}"
                )

            self._unstacked_dim_size = C  # channels per frame

            # 把 obs space 变成“去掉所有时间维”的 Box（即单帧）
            unstacked_pixel_obs_space = Box(low=low, high=high, dtype=pixel_obs_space.dtype)
            observation_space.spaces[pixel_key] = unstacked_pixel_obs_space

        next_observation_space_dict = copy.deepcopy(observation_space.spaces)
        for pixel_key in self.pixel_keys:
            next_observation_space_dict.pop(pixel_key)
        next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        self._first = True
        self._is_correct_index = np.full(capacity, False, dtype=bool)

        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space=next_observation_space,
            include_next_actions=include_next_actions,
            include_grasp_penalty=include_grasp_penalty,
        )

    def _to_T_HWC(self, arr: np.ndarray) -> np.ndarray:
        """把 (H,W,C) 或 任意形状(...,H,W,C) 统一 reshape 成 (T_total, H, W, C)。"""
        if arr.ndim == 3:
            H, W, C = arr.shape
            return arr.reshape(1, H, W, C)
        elif arr.ndim >= 4:
            H, W, C = arr.shape[-3], arr.shape[-2], arr.shape[-1]
            T_total = int(np.prod(arr.shape[:-3]))
            return arr.reshape(T_total, H, W, C)
        else:
            raise ValueError(f"Unexpected image ndim: {arr.ndim}")

    def insert(self, data_dict: DatasetDict):
        if self._insert_index == 0 and self._capacity == len(self) and not self._first:
            indxs = np.arange(len(self) - self._num_stack, len(self))
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._is_correct_index[self._insert_index] = False
                super().insert(element)

        data_dict = data_dict.copy()
        data_dict["observations"] = data_dict["observations"].copy()
        data_dict["next_observations"] = data_dict["next_observations"].copy()

        obs_pixels = {}
        next_obs_pixels = {}
        for pixel_key in self.pixel_keys:
            obs_pixels[pixel_key] = data_dict["observations"].pop(pixel_key)
            next_obs_pixels[pixel_key] = data_dict["next_observations"].pop(pixel_key)

        if self._first:
            # 把第一步“预热”的 T-1 帧也插入（与原逻辑一致）
            for i in range(self._num_stack):
                for pixel_key in self.pixel_keys:
                    arr = self._to_T_HWC(obs_pixels[pixel_key])  # (T,H,W,C)
                    frame = arr[i if i < arr.shape[0] else -1]
                    data_dict["observations"][pixel_key] = frame

                self._is_correct_index[self._insert_index] = False
                super().insert(data_dict)

        for pixel_key in self.pixel_keys:
            arr = self._to_T_HWC(next_obs_pixels[pixel_key])  # (T,H,W,C)
            frame = arr[-1]  # 下一帧
            data_dict["observations"][pixel_key] = frame

        self._first = data_dict["dones"]

        self._is_correct_index[self._insert_index] = True
        super().insert(data_dict)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._is_correct_index[indx] = False

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> frozen_dict.FrozenDict:
        """Samples from the replay buffer.

        Args:
            batch_size: Minibatch size.
            keys: Keys to sample.
            indx: Take indices instead of sampling.
            pack_obs_and_next_obs: whether to pack img and next_img into one image.
                It's useful when they have overlapping frames.

        Returns:
            A frozen dictionary.
        """

        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

            for i in range(batch_size):
                while not self._is_correct_index[indx[i]]:
                    if hasattr(self.np_random, "integers"):
                        indx[i] = self.np_random.integers(len(self))
                    else:
                        indx[i] = self.np_random.randint(len(self))
        else:
            raise NotImplementedError()

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys

        keys = list(keys)
        keys.remove("observations")
        batch = super().sample(batch_size, keys, indx)
        batch = batch.unfreeze()

        obs_keys = self.dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        for pixel_key in self.pixel_keys:
            obs_keys.remove(pixel_key)

        batch["observations"] = {}
        for k in obs_keys:
            batch["observations"][k] = _sample(
                self.dataset_dict["observations"][k], indx
            )

        for pixel_key in self.pixel_keys:
            # 存储时我们已把像素展成了 (N, H, W, C) 单帧序列
            obs_pixels = self.dataset_dict["observations"][pixel_key]  # (N,H,W,C)
            obs_pixels = np.lib.stride_tricks.sliding_window_view(
                obs_pixels, self._num_stack + 1, axis=0
            )
            obs_pixels = obs_pixels[indx - self._num_stack]  # (B, T, H, W, C)

            if obs_pixels.ndim != 5:
                raise ValueError(f"Expected obs_pixels to be 5D (B,T,H,W,C), got shape {obs_pixels.shape}")

            # 变成 (B, C, T, H, W)
            obs_pixels = obs_pixels.transpose((0, 4, 1, 2, 3))

            if pack_obs_and_next_obs:
                batch["observations"][pixel_key] = obs_pixels
            else:
                batch["observations"][pixel_key] = obs_pixels[:, :, :-1, ...]
                if "next_observations" in keys:
                    batch["next_observations"][pixel_key] = obs_pixels[:, :, 1:, ...]

        return frozen_dict.freeze(batch)
