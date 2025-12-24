"""Gym Interface for Franka"""

import os

# ---- HighGUI 环境兜底（必须在 import cv2 之前）----
if os.environ.get("XDG_SESSION_TYPE","") == "wayland" and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"          # Wayland 下强制用 xcb，避免 HighGUI 不响应
os.environ.setdefault("DISPLAY", ":0")             # 本地会话常见设置；若远程/容器要改成实际 DISPLAY
xr = f"/run/user/{os.getuid()}"
if not os.environ.get("XDG_RUNTIME_DIR"):
    if os.path.isdir(xr):
        os.environ["XDG_RUNTIME_DIR"] = xr
    else:
        tmp = f"/tmp/runtime-{os.getuid()}"
        os.makedirs(tmp, exist_ok=True)
        os.environ["XDG_RUNTIME_DIR"] = tmp
        
import numpy as np
import gymnasium as gym
import cv2

import threading
CV_UI_LOCK = threading.RLock()

import copy

from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
import yaml

from datetime import datetime
from collections import OrderedDict
from typing import Dict

from franka_env.utils.rotations import euler_2_quat, quat_2_euler
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import threading
from denso_env.camera.video_capture import VideoCapture
from denso_env.camera.rs_capture import RSCapture
from leap_hand.srv import LeapPosition, LeapPosVelEff
from shape_reconstruction import Sensor

from examples.utils import read_utils

import pathlib, json, zmq, msgpack
from typing import List, Tuple
from collections import defaultdict
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

# —— 放在 denso_env.py 顶部，所有 OpenCV/GUI 代码之前 ——
try:
    import ctypes
    _x11 = ctypes.CDLL("libX11.so")
    _x11.XInitThreads()  # 关键：允许 X11 多线程 GUI
    print("[X11] XInitThreads OK")
except Exception as _e:
    print("[X11][WARN] XInitThreads failed:", _e)
    
    
def _sam2_prompt_child_main(mirror_dir: str, key_indices: list[int], mode: str, title: str, out_path: str):
    """在独立进程里跑 OpenCV 交互 UI，把标注写到 out_path(JSON)。"""
    win = f"{title} - {mode.upper()}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    prompts = {}
    active_obj = 0
    tmp_box = []
    tmp_box_end = None
    dragging = False
    cur_i = 0

    def _on_mouse(event, x, y, flags, param):
        nonlocal dragging, tmp_box, tmp_box_end, active_obj
        idx = param["cur_idx"]
        d = prompts.setdefault(idx, {})
        if active_obj not in d:
            d[active_obj] = {"clicks": [], "labels": [], "boxes": []}
        ent = d[active_obj]
        if event == cv2.EVENT_LBUTTONDOWN:
            tmp_box = [(x, y)]; tmp_box_end = (x, y); dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging and len(tmp_box) == 1:
            tmp_box_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and len(tmp_box) == 1:
            tmp_box.append((x, y)); ent["boxes"].append(list(tmp_box))
            tmp_box = []; tmp_box_end = None; dragging = False
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            ent["clicks"].append([x, y]); ent["labels"].append(1)
        elif event == cv2.EVENT_MBUTTONDOWN:
            ent["clicks"].append([x, y]); ent["labels"].append(0)

    cv2.setMouseCallback(win, _on_mouse, param={"cur_idx": -1})

    while True:
        kf = int(key_indices[cur_i])
        img_path = os.path.join(mirror_dir, f"{kf}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            if cur_i < len(key_indices) - 1:
                cur_i += 1
                continue
            else:
                break

        vis = img.copy()
        d = prompts.get(kf, {})
        for oid, rec in d.items():
            for (px, py), lbl in zip(rec.get("clicks", []), rec.get("labels", [])):
                cv2.circle(vis, (px, py), 6, (0,255,0) if lbl else (0,0,255), -1)
            for b in rec.get("boxes", []):
                if len(b) == 2:
                    cv2.rectangle(vis, b[0], b[1], (255,0,0), 2)
        if len(tmp_box) == 1 and tmp_box_end is not None:
            cv2.rectangle(vis, tmp_box[0], tmp_box_end, (200,200,0), 1)

        msg = f"[{mode}] key {cur_i+1}/{len(key_indices)} frame={kf} obj={active_obj} (j/k prev/next, n new obj, Enter finish, q quit)"
        cv2.putText(vis, msg, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(10) & 0xFF

        if key in (13, 10):  # Enter
            break
        elif key == ord('q'):
            break
        elif key == ord('n'):
            active_obj += 1
        elif key == ord('j') and cur_i > 0:
            cur_i -= 1
        elif key == ord('k') and cur_i < len(key_indices) - 1:
            cur_i += 1

        cv2.setMouseCallback(win, _on_mouse, param={"cur_idx": kf})

    try: cv2.destroyWindow(win)
    except Exception: pass
    with open(out_path, "w") as f:
        json.dump(prompts, f)



class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)

# === NEW: ===
# 顶部确保有：
# CV_UI_LOCK = threading.RLock()
# import time

class SingleImageDisplayer(threading.Thread):
    def __init__(
        self,
        queue,
        name,
        fullscreen=False,
        pause_event=None,
        target_size=(1920, 1080),
        resize_mode="stretch",  # "stretch" 或 "letterbox"
        assume_rgb=False,       # 如果你往队列里塞的是RGB，就置True；否则按BGR处理
    ):
        super().__init__(daemon=True)
        self.queue = queue
        self.name = name
        self.fullscreen = fullscreen
        self.pause_event = pause_event
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.resize_mode = resize_mode
        self.assume_rgb = assume_rgb

    def _to_bgr(self, img):
        # cv2.imshow 期望BGR；如果外面给的是RGB就翻回去
        if self.assume_rgb and img.ndim == 3 and img.shape[2] == 3:
            return img[..., ::-1]
        return img

    def _ensure_3ch(self, img):
        # 灰度→3通道
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _fit_1920x1080(self, img):
        tw, th = self.target_size
        ih, iw = img.shape[:2]

        if self.resize_mode == "letterbox":
            # 等比缩放 + 黑边
            scale = min(tw / iw, th / ih)
            nw, nh = int(round(iw * scale)), int(round(ih * scale))
            interp = cv2.INTER_AREA if (nw < iw or nh < ih) else cv2.INTER_LINEAR
            resized = cv2.resize(img, (nw, nh), interpolation=interp)
            canvas = np.zeros((th, tw, 3), dtype=resized.dtype)
            y0 = (th - nh) // 2
            x0 = (tw - nw) // 2
            canvas[y0:y0+nh, x0:x0+nw] = resized
            return canvas
        else:
            # 拉伸到精准 1920×1080
            interp = cv2.INTER_AREA if (tw < iw or th < ih) else cv2.INTER_LINEAR
            return cv2.resize(img, (tw, th), interpolation=interp)

    def run(self):
        win = self.name
        with CV_UI_LOCK:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            if self.fullscreen:
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(win, self.target_size[0], self.target_size[1])

        while True:
            if self.pause_event is not None and self.pause_event.is_set():
                with CV_UI_LOCK:
                    cv2.waitKey(1)
                time.sleep(0.02)
                continue

            img = self.queue.get()
            if img is None:
                break

            img = self._ensure_3ch(img)
            img = self._to_bgr(img)
            img_out = self._fit_1920x1080(img)  # ← 这里强制得到 1920×1080

            with CV_UI_LOCK:
                cv2.imshow(win, img_out)
                cv2.waitKey(1)

        with CV_UI_LOCK:
            try:
                cv2.destroyWindow(win)
            except Exception:
                pass



##############################################################################

class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS: Dict = {
        "front_camera": "242422303461",
        # "side_camera": "234222300515",
    }
    EXTRA_REALSENSE_CAMERAS: Dict = {
        # "front_camera": "242422303461",
        "side_camera": "234222300515",
    }
    DISPLAY_RGB_SERIAL: str = "242422303461"
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((7,))
    # GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    # RESET_POSE = np.zeros((6,))
    # RANDOM_RESET = False
    # RANDOM_XY_RANGE = (0.0,)
    # RANDOM_RZ_RANGE = (0.0,)
    # ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    # ABS_POSE_LIMIT_LOW = np.zeros((6,))
    # COMPLIANCE_PARAM: Dict[str, float] = {}
    # RESET_PARAM: Dict[str, float] = {}
    # PRECISION_PARAM: Dict[str, float] = {}
    # LOAD_PARAM: Dict[str, float] = {
    #     "mass": 0.0,
    #     "F_x_center_load": [0.0, 0.0, 0.0],
    #     "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # }
    DISPLAY_IMAGE: bool = True
    # GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 200
    # JOINT_RESET_PERIOD: int = 0


##############################################################################

class ROSNodeInterface(Node):
    def __init__(self):
        super().__init__('denso_env_node')

        # Publishers（发送）
        self.arm_pub = self.create_publisher(
            PoseStamped,
            '/tactexo/robot_control',
            10
        )

        self.publisher_hand = self.create_publisher(
            JointState, 
            '/cmd_leap', 
            10
        )

        # Subscribers（接收）
        self.robot_ee_sub = self.create_subscription(
            PoseStamped,
            '/cartesian_compliance_controller/current_pose',
            self.robot_ee_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',  # 接收机械臂关节角
            self.joint_callback,
            10
        )
        
        self.leap_position_client = self.create_client(LeapPosition, '/leap_position')

        # Wait for the service to be available
        while not self.leap_position_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Waiting for /leap_position service...')

        # 同步事件
        self.robot_ee_event = threading.Event()
        self.joint_event = threading.Event()
        self.hand_joint_event = threading.Event()

        # 数据存储
        self.current_joints = None
        self.current_hand_joints = None

        self.cur_position = np.zeros(3, dtype=np.float32)
        self.cur_oritation = np.zeros(4, dtype=np.float32)


    def robot_ee_callback(self, msg):
        #用ros2 INFO打印接收到的数据 
        position = msg.pose.position
        # self.get_logger().info(f"robot_ee_received:{position}")
        self.cur_position = np.array([position.x, position.y, position.z])

        # 从 msg 中提取四元数方向数据（xyzw）
        orientation = msg.pose.orientation
        self.cur_oritation = np.array([
            orientation.w, orientation.x, orientation.y, orientation.z
        ])
        # 设置事件为已收到数据
        # self.robot_ee_event.set()


    def joint_callback(self, msg):

        # 将 joint name 和对应的位置打包为字典
        joint_dict = {name: pos for name, pos in zip(msg.name, msg.position)}

        # 按照你需要的顺序提取关节角：joint1 ~ joint6
        ordered_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        ordered_joint_positions = [joint_dict.get(joint, 0.0) for joint in ordered_joint_names]

        # 保存为 numpy array
        self.joint_position = np.array(ordered_joint_positions, dtype=np.float32)

        # 设置事件为“数据已接收”
        # self.joint_event.set()


    def publish_arm_action(self, pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        x, y, z = map(float, pose[:3])
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        ori_w, ori_x, ori_y, ori_z = map(float, pose[3:7])
        # msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        msg.pose.orientation.w = ori_w
        msg.pose.orientation.x = ori_x
        msg.pose.orientation.y = ori_y
        msg.pose.orientation.z = ori_z
        # print("msg.pose = ", msg.pose)
        self.arm_pub.publish(msg)


    def publish_hand_action(self, hand_joints):
        # self.get_logger().info('publish_hand_action')
        stater = JointState()
        stater.name = [f"joint_{i}" for i in range(len(hand_joints))]
        stater.position = hand_joints
        self.publisher_hand.publish(stater)
        

    # def get_current_robot_ee(self, timeout=5.0):
    #     # success = self.joint_event.wait(timeout=timeout)
    #     # if not success:
    #     #     raise TimeoutError("等待机械臂数据超时，请检查ROS话题是否正常发布。")
    #     self.robot_ee_event.wait()
    #     self.robot_ee_event.clear()
    #     return self.cur_position, self.cur_oritation
    

    def get_current_robot_ee(self, timeout=5.0):
        # success = self.robot_ee_event.wait(timeout=timeout)
        # if not success:
        #     raise TimeoutError("等待get_current_robot_ee超时，请检查ROS话题是否正常发布。")
        # self.robot_ee_event.wait()
        # self.robot_ee_event.clear()
        # self.get_logger().info(f"robot_ee_received:{self.cur_position}")
        # print("get_current_robot_ee: cur_position = ", self.cur_position)
        # print("get_current_robot_ee: cur_oritation = ", self.cur_oritation)
        return self.cur_position, self.cur_oritation
    

    def get_current_joint(self, timeout=5.0):
        # success = self.joint_event.wait(timeout=timeout)
        # if not success:
        #     raise TimeoutError("等待get_current_joint超时，请检查ROS话题是否正常发布。")
        # self.joint_event.wait()
        # self.joint_event.clear()
        return self.joint_position

    def get_current_leap_position(self):
        # Create a request for the LeapPosition service
        req = LeapPosition.Request()
        future = self.leap_position_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            return list(future.result().position)
        else:
            self.get_logger().info("Failed to get current position, using zeros")
            return [0.0] * 16
        

    def reset_cur_pose(self):
        self.cur_position = np.array([0.55513753, 0.04267503, 0.18153528])
        self.cur_oritation = np.array([-0.03244228, 0.99039508, 0.12396424, -0.05194187])

##############################################################################

class DensoEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        # self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        # self.gripper_sleep = config.GRIPPER_SLEEP
        self.is_arm_only = config.IS_ARM_ONLY
        self.tact_base_path = config.TACT_BASE_PATH
        self.enable_tactile = config.ENABLE_TACTILE
        self.rgb_display_serial = getattr(config, "DISPLAY_RGB_SERIAL", None)
        self.fake_env = fake_env
        



        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        # self.resetpos = np.concatenate(
        #     [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        # )
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        # self.randomreset = config.RANDOM_RESET
        # self.random_xy_range = config.RANDOM_XY_RANGE
        # self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        # self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        # self.xyz_bounding_box = gym.spaces.Box(
        #     config.ABS_POSE_LIMIT_LOW[:3],
        #     config.ABS_POSE_LIMIT_HIGH[:3],
        #     dtype=np.float64,
        # )
        # self.rpy_bounding_box = gym.spaces.Box(
        #     config.ABS_POSE_LIMIT_LOW[3:],
        #     config.ABS_POSE_LIMIT_HIGH[3:],
        #     dtype=np.float64,
        # )
        # Action/Observation Space
        if not self.is_arm_only:
            print("init arm with hand")
            self.action_space = gym.spaces.Box(
                np.ones((22,), dtype=np.float32) * -1,
                np.ones((22,), dtype=np.float32),
            )

            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pos": gym.spaces.Box(
                                -np.inf, np.inf, shape=(3,)
                            ),
                            "tcp_ori": gym.spaces.Box(
                                -np.inf, np.inf, shape=(4,)
                            ),
                            "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(16,)),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {key: gym.spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS}
                    ),
                }
            )
        elif self.enable_tactile:
            print("init arm with tactile")
            # low  = np.concatenate([np.ones(6, dtype=np.float32) * -1, [0]])
            # high = np.ones(7, dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))
            # self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pos": gym.spaces.Box(
                                -np.inf, np.inf, shape=(3,)
                            ),
                            "tcp_ori": gym.spaces.Box(
                                -np.inf, np.inf, shape=(4,)
                            ),
                            "gripper_pose": gym.spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            )
                        }
                    ),
                    # "images": gym.spaces.Dict(
                    #     {
                    #         **{key: gym.spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8) 
                    #                 for key in config.REALSENSE_CAMERAS},
                    #         "tactile_data": gym.spaces.Box(0, 255, shape=(240, 960, 3), dtype=np.uint8),
                    #     }
                    # ),
                    "images": gym.spaces.Dict(
                        {
                            **{key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS},
                            "tactile_data": gym.spaces.Box(0, 255, shape=(128, 384, 3), dtype=np.uint8),
                            "gaze_mask": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8), 
                        }
                    ),
                }
            )
        else:
            print("init arm only")
            self.action_space = gym.spaces.Box(
                np.ones((6,), dtype=np.float32) * -1,
                np.ones((6,), dtype=np.float32),
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pos": gym.spaces.Box(
                                -np.inf, np.inf, shape=(3,)
                            ),
                            "tcp_ori": gym.spaces.Box(
                                -np.inf, np.inf, shape=(4,)
                            ),
                            # "gripper_pose": gym.spaces.Box(
                            #     -np.inf, np.inf, shape=(1,), dtype=np.int32
                            # )
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {key: gym.spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS}
                    ),
                }
            )


        if self.enable_tactile and not self.fake_env:
            # tactile configuration loading and init
            thumb_cfg_path = os.path.join(self.tact_base_path, "shape_config_thumb.yaml")
            # assert the path exists
            if not os.path.exists(thumb_cfg_path):
                raise FileNotFoundError(f"Configuration file not found: {thumb_cfg_path}")
            thumb_f = open(thumb_cfg_path, 'r+', encoding='utf-8')
            thumb_cfg = yaml.load(thumb_f, Loader=yaml.FullLoader)
            self.thumb_tactile_sensor = Sensor(thumb_cfg)
            # self.thumb_tactile_vis  = Visualizer(self.thumb_tactile_sensor.points)

            index_cfg_path = os.path.join(self.tact_base_path, "shape_config_index.yaml")
            index_f = open(index_cfg_path, 'r+', encoding='utf-8')
            index_cfg = yaml.load(index_f, Loader=yaml.FullLoader)
            self.index_tactile_sensor = Sensor(index_cfg)
            # self.index_tactile_vis  = Visualizer(self.index_tactile_sensor.points)

            middle_cfg_path = os.path.join(self.tact_base_path, "shape_config_middle.yaml")
            middle_f = open(middle_cfg_path, 'r+', encoding='utf-8')
            middle_cfg = yaml.load(middle_f, Loader=yaml.FullLoader)
            self.middle_tactile_sensor = Sensor(middle_cfg)
            # self.middle_tactile_vis  = Visualizer(self.middle_tactile_sensor.points)

            self.thumb_raw_img = []
            self.index_raw_img = []
            self.middle_raw_img = []

            self.thumb_points = []
            self.index_points = []
            self.middle_points = []

            self.thumb_heat_map = []
            self.index_heat_map = []
            self.middle_heat_map = []

            self.rthumb_raw_buffer = []  # Right thumb raw tactile image
            self.rindex_raw_buffer = []  # Right index raw tactile image
            self.rmiddle_raw_buffer = []

            self.rthumb_heatmap_buffer = []  # Right thumb heatmap tactile image
            self.rindex_heatmap_buffer = []  # Right index heatmap tactile image
            self.rmiddle_heatmap_buffer = []

            self.tac_thumb_lock = threading.Lock()
            self.tac_index_lock = threading.Lock()
            self.tac_middle_lock = threading.Lock()
            self.tac_main_lock = threading.Lock()

            # Start threads for each tactile sensor
            self.start_tac_processing()
        
        self.cycle_count = 0

        self.front_color_buffer = []            #  D435i color 
        self.front_depth_buffer = []            #  D435i depth 
        self.side_color_buffer = []           #  D435i 2 color
        self.side_depth_buffer = []           #  D435i 2 depth
        self.joint_buffer = []

        # robot_urdf_path = "/home/qiangqiang/workspaces/HK_TACTEXO_DATA/denso_robot_with_ati_4.urdf"
        # self.data_count = 0
        # self.data = read_utils.read_data(robot_urdf_path,True)
        # self._update_cur_position()

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS, config.EXTRA_REALSENSE_CAMERAS)
        if self.display_image:
            #self.img_queue = queue.Queue()
            #self.displayer = ImageDisplayer(self.img_queue, self.url)
            #self.displayer.start()
            # === NEW: front_camera 独立显示 ===
            self.ui_pause_event = threading.Event()
            self.rgb_queue = queue.Queue(maxsize=2)
            self.rgb_viewer = SingleImageDisplayer(
                self.rgb_queue,
                name=f"{self.url}-front-rgb",
                fullscreen=False,
                pause_event=self.ui_pause_event,# 置为 False 则仅最大化窗口可拖拽
                target_size=(1920, 1080),
                resize_mode="stretch",
                assume_rgb=False, 
            )
            self.rgb_viewer.start()

        self.position_data = []
        self.oritation_data = []
        self.next_hand_pos = np.zeros(16)

        rclpy.init(args=None)

        self.ros_interface = ROSNodeInterface()

        # executor = rclpy.executors.MultiThreadedExecutor()
        # executor.add_node(self.ros_interface)
        # executor_thread = threading.Thread(target=executor.spin, daemon=True)
        # executor_thread.start()

        # Spin ROS callbacks in a background thread and keep references so we can
        # shut everything down cleanly when the environment closes.
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.ros_interface)
        self.executor_thread = threading.Thread(
            target=self.executor.spin, daemon=True
        )
        self.executor_thread.start()

        self.interpolation_thread = None
        self.thread_lock = threading.Lock()

        self.print_action = True
        self._last_step_time = None

        self.frame_save_path = "/mnt/data3/hcj/recorded_data_robot/recorded_data_training-10-11-10"  # 可自行修改
        os.makedirs(self.frame_save_path, exist_ok=True)
        self.frame_count = 0

        self.cur_position = np.zeros(3, dtype=np.float32)
        self.cur_oritation = np.zeros(4, dtype=np.float32)

        # grip with index
        self.gripper_close_joint = [
            3.546563625335693359, 4.127942085266113281, 3.413689804077148438, 3.641670465469360352,
            3.626330614089965820, 3.529689788818359375, 2.931437253952026367, 3.782796621322631836,
            3.838019847869873047, 3.532757759094238281, 3.535825729370117188, 3.413107156753540039,
            4.661767482757568359, 3.366175127029418945, 3.260291767120361328, 3.566796636581420898
        ]

        # self.gripper_open_joint = [
        #     2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
        #     2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
        #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        #     4.312019824981689453, 3.905515193939208984, 3.374757766723632812, 3.597184896469116211    #thumb
        # ]
        
        self.gripper_open_joint = [
            2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
            2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
            3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
            4.512019824981689453, 3.3605515193939208984, 3.374757766723632812, 3.397184896469116211    #thumb
        ]

        #grip with middle
        # self.gripper_close_joint = [
        #     3.132388830184936523, 3.186078071594238281, 3.153864383697509766, 3.147728681564331055,
        #     3.201417922973632812, 4.543651103973388672, 2.943709135055541992, 3.740427684783935547,
        #     3.144660711288452148, 3.181476116180419922, 3.144660711288452148, 3.140058755874633789,
        #     4.825903415679931641, 3.525670242309570312, 3.230563640594482422, 3.240349960327148438
        # ]

        # self.gripper_open_joint = [
        #     2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
        #     2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
        #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        #     4.312019824981689453, 3.905515193939208984, 3.374757766723632812, 3.597184896469116211  #thumb
        # ]

        # self.last_hand_pos = [
        #     3.546563625335693359, 4.127942085266113281, 3.413689804077148438, 3.641670465469360352,
        #     3.626330614089965820, 3.529689788818359375, 2.931437253952026367, 3.782796621322631836,
        #     3.144660711288452148, 3.181476116180419922, 3.144660711288452148, 3.140058755874633789,
        #     4.661767482757568359, 3.366175127029418945, 3.260291767120361328, 3.566796636581420898
        # ]

        self.curr_leap_hand_pos = list(self.gripper_open_joint)

        # self.changed_hand_pos = [
        #     2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
        #     2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
        #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        #     4.512019824981689453, 3.3605515193939208984, 3.374757766723632812, 3.397184896469116211    #thumb
        # ]

        self.hand_state = 0.0 #hand opened lable
        
        self.gripper_close_joint_np = np.asarray(self.gripper_close_joint, dtype=np.float32)
        self.gripper_open_joint_np = np.asarray(self.gripper_open_joint, dtype=np.float32)
        self.gripper_direction = np.sign(self.gripper_close_joint_np - self.gripper_open_joint_np)
        self.max_gripper_step = np.abs(self.gripper_close_joint_np - self.gripper_open_joint_np) / 10.0
        # Ensure direction is either -1 or 1 where movement is required
        self.gripper_direction[self.gripper_direction == 0] = 1.0



        print("Initialized Denso")

        # === NEW: inline 采集 & SAM2 所需的状态 ===
        self.global_frame_id = 0
        self.frame_root = self.frame_save_path  # 与保存根目录保持一致
        self.et_mirror_dir = os.path.join(self.frame_root, "et_images")
        self.rs_mirror_dir = os.path.join(self.frame_root, "rs_images")
        os.makedirs(self.et_mirror_dir, exist_ok=True)
        os.makedirs(self.rs_mirror_dir, exist_ok=True)

        # 批 episode 记账
        self.episode_frame_ranges: List[Tuple[int, int]] = []
        self._cur_ep_start: int | None = None
        self._episode_counter = 0

        # Pupil Core 订阅
        self._pupil_host = "127.0.0.1"
        self._pupil_port = 50020
        self._pupil_ctx = zmq.Context()
        self._pupil_sub = None
        self._pupil_last_world_payload = None
        self._pupil_last_gaze = None
        try:
            ctrl = self._pupil_ctx.socket(zmq.REQ)
            ctrl.connect(f"tcp://{self._pupil_host}:{self._pupil_port}")
            ctrl.send_string("SUB_PORT")
            sub_port = ctrl.recv_string()
            ctrl.close()

            self._pupil_sub = self._pupil_ctx.socket(zmq.SUB)
            self._pupil_sub.connect(f"tcp://{self._pupil_host}:{sub_port}")
            self._pupil_sub.setsockopt(zmq.RCVTIMEO, 500)
            self._pupil_sub.setsockopt_string(zmq.SUBSCRIBE, "gaze.")
            self._pupil_sub.setsockopt_string(zmq.SUBSCRIBE, "frame.world")

            threading.Thread(target=self._pupil_listen, daemon=True).start()
            print("[DensoEnv] Pupil subscriber started.")
        except Exception as e:
            print(f"[DensoEnv][WARN] Pupil init failed: {e}")
            self._pupil_sub = None


    def start_tac_processing(self):
        # Start threads for each tactile sensor
        self.thumb_thread = threading.Thread(target=self.process_thumb_tactile, daemon=True)
        self.thumb_thread.start()
        self.index_thread = threading.Thread(target=self.process_index_tactile, daemon=True)
        self.index_thread.start()
        self.middle_thread = threading.Thread(target=self.process_middle_tactile, daemon=True)
        self.middle_thread.start()


    def close_tac_processing(self):
        # Close threads for each tactile sensor
        self.thumb_thread.join()
        self.index_thread.join()
        self.middle_thread.join()


    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        
        action = np.clip(action, self.action_space.low, self.action_space.high)

        print("action = ", action)

        xyz_delta = action[:3]

        self.nextpos = np.concatenate((self.cur_position, self.cur_oritation), axis=0)
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
        if self.nextpos[2] < 0.03:
            self.nextpos[2] = 0.03

        # GET ORIENTATION FROM ACTION
        rpy_delta = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # self.nextpos[3:] = (
        #     Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
        #     * Rotation.from_quat(self.cur_oritation)
        # ).as_quat()
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", rpy_delta * self.action_scale[1])
            * Rotation.from_quat(self.cur_oritation)
        ).as_quat()


        grip_action = float(np.clip(action[6], -1.0, 1.0))
        current_hand_pos = np.asarray(self.curr_leap_hand_pos, dtype=np.float32)
        proposed_hand_pos = current_hand_pos + grip_action * self.max_gripper_step * self.gripper_direction
        close_ge_open_mask = self.gripper_close_joint_np >= self.gripper_open_joint_np
        # print("current_hand_pos = ", current_hand_pos)
        if grip_action > 0:
            target_hand_pos = np.where(
                close_ge_open_mask,
                np.minimum(proposed_hand_pos, self.gripper_close_joint_np),
                np.maximum(proposed_hand_pos, self.gripper_close_joint_np),
            )
        elif grip_action < 0:
            target_hand_pos = np.where(
                close_ge_open_mask,
                np.maximum(proposed_hand_pos, self.gripper_open_joint_np),
                np.minimum(proposed_hand_pos, self.gripper_open_joint_np),
            )
        else:
            target_hand_pos = current_hand_pos.copy()
            
        # print("target_hand_pos = ", target_hand_pos)

        self.ros_interface.publish_arm_action(self.nextpos)

        diff = target_hand_pos - current_hand_pos
        # # print("np.max(np.abs(diff)) = ", np.max(np.abs(diff)))
        # if np.max(np.abs(diff)) > 0.15:
                
        np.asarray(target_hand_pos, dtype=np.float32)
        
        # print("target_hand_pos 1= ", target_hand_pos)
        # print("diff = ", diff)
        if np.max(np.abs(diff)) > 0.01:
        # if not grip_action == 0.0:
            self._send_leap_hand_command(target_hand_pos.copy())


        # time.sleep(1.5)
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        t_end = time.time()
        print(f"[publish End] {t_end:.6f}, Step总耗时(含sleep): {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")

        self.curr_path_length += 1
        self._update_cur_position(self.nextpos)
        self.hand_state = grip_action

        # t_end = time.time()
        # print(f"[update_position End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # print("after publish arm action cur_position = ", self.cur_position)
        self.frame_count += 1
        self.save_training_frame()

        ob = self._get_obs()
        reward = self.compute_reward(ob)
        # print(f"reward in denso_env = {reward}")
        # done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        # here!!!!!!!!!! done = reward or self.terminate
        # t_end = time.time()
        # print(f"[Step End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # print("curr hand pos = ", self.curr_leap_hand_pos)
        # input("debug for hand pos")
        done = False
        done_reason = None
        if reward:
            done = True
            done_reason = "success"
        elif self.terminate:
            done = True
            done_reason = "manual_terminate"
        elif self.curr_path_length >= self.max_episode_length:
            done = True
            done_reason = "max_length"
            

        # New
        info = {"succeed": reward}
        info["done_reason"] = done_reason # 
        info["path_length"] = int(self.curr_path_length) # 
        info["frame_idx"] = self.global_frame_id  # 将在 save_training_frame() 中自增
        info["episode_id"] = self._episode_counter
        return ob, int(reward), done, False, info

    
    def move_up(self):
        print("move up to avoid collision")
        pos = self.cur_position.copy()
        pos[2] += 0.1
        ori = self.cur_oritation.copy()
        nextpos = np.concatenate((pos, ori), axis=0)
        self.ros_interface.publish_arm_action(nextpos)
        time.sleep(2.0)
    

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]
        # convert from quat to euler first
        # current_rot = Rotation.from_quat(current_pose[3:7]).as_matrix()
        # target_rot = Rotation.from_quat(self._TARGET_POSE[3:7]).as_matrix()
        # diff_rot = current_rot.T  @ target_rot
        # diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        # delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], [0, 0, 0]]))
        # if np.all(delta < self._REWARD_THRESHOLD):
        #     return True
        # else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
        #     return False
        pos_err = np.abs(current_pose[:3] - self._TARGET_POSE[:3])
        thr = self._REWARD_THRESHOLD
        if thr is None or len(thr) == 0:
            return False
        thr3 = np.asarray(thr[:3], dtype=np.float64)
        return bool(np.all(pos_err <= thr3))
    

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            if key == "side_camera":
                continue
            try:
                frame = cap.read()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    rgb = frame[..., :3]   # BGR 彩色
                else:
                    rgb = frame
                rgb = rgb.astype(np.uint8)
                # cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                cropped_rgb = rgb #当前不需要裁剪
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS, self.config.EXTRA_REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            #with self.tac_index_lock:
                # index_heat_map_resized = cv2.resize(self.index_heat_map, (128, 128))  # 如果想缩放
            #    heat_map = cv2.hconcat([self.thumb_heat_map, self.index_heat_map])
            #    display_image = {"heat_map": heat_map}
            #self.img_queue.put(display_image)
            # === NEW: 把 front_camera 的原始分辨率图像送到独立 RGB 窗口 ===
            # 优先使用 full-res（cropped_rgb），否则退化为缩放后的
            # try:
            #     if hasattr(self, "rgb_queue"):
            #         if "front_camera_full" in display_images:
            #             self.rgb_queue.put(display_images["front_camera_full"])
            #         elif "front_camera" in display_images:
            #             self.rgb_queue.put(display_images["front_camera"])

            # except Exception as _e:
            #     pass
            try:
                if hasattr(self, "rgb_queue"):
                    chosen_img = None
                    target_sn = getattr(self, "rgb_display_serial", None)
                    if target_sn and hasattr(self, "cap_serial"):
                        for cam_name, sn in self.cap_serial.items():
                            if sn == target_sn:
                                if f"{cam_name}_full" in display_images:
                                    chosen_img = display_images[f"{cam_name}_full"]
                                elif cam_name in display_images:
                                    chosen_img = display_images[cam_name]
                                break
                    if chosen_img is None:
                        for cand in ("front_camera_full", "front_camera"):
                            if cand in display_images:
                                chosen_img = display_images[cand]
                                break
                    if chosen_img is not None:
                        self._safe_put(self.rgb_queue, chosen_img)
            except Exception as _e:
                pass


        if "gaze_mask" in self.observation_space["images"].spaces:
            H, W, C = self.observation_space["images"]["gaze_mask"].shape
            images["gaze_mask"] = np.zeros((H, W, C), dtype=np.uint8)

        return images


    def get_rgb_and_dpth_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        depth_images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                frame = cap.read()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    rgb = frame[..., :3]   # BGR 彩色
                    depth = frame[..., 3]    # 深度
                else:
                    rgb = frame
                    depth = None
                # cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                cropped_rgb = rgb #当前不需要裁剪
                if key in self.observation_space["images"]:
                    resized = cv2.resize(
                        cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                    )
                else:
                    resized = cv2.resize(
                        cropped_rgb, (640, 480)
                    )
                    
                images[key] = resized[..., ::-1]
                depth_images[key] = depth
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS, self.config.EXTRA_REALSENSE_CAMERAS)
                return self.get_rgb_and_dpth_im()

        return images, depth_images


    def process_tactile_data(self, sensor, img_size):
        heat_map = []
        raw_img = []
        points = []
       
        raw_img = sensor.get_rectify_crop_image()
        img_GRAY = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        height_map = sensor.raw_image_2_height_map(img_GRAY)
        height_map = sensor.expand_image(height_map)
        heat_map_input = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX)
        heat_map_input = np.uint8(heat_map_input)
        heat_map = cv2.applyColorMap(heat_map_input, cv2.COLORMAP_JET)
        # Add subtitles to each image
        # cv2.putText(heat_map, "Thumb", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Resize images for display
        target_size = img_size
        heat_map = cv2.resize(heat_map, target_size, interpolation=cv2.INTER_LINEAR)
        points, gradients = sensor.height_map_2_point_cloud_gradients(height_map)

        return raw_img, points, heat_map
    

    def process_thumb_tactile(self):
        while True:
            # thumb_raw_img, thumb_points, thumb_heat_map = self.process_tactile_data(self.thumb_tactile_sensor, (320, 240))
            thumb_raw_img, thumb_points, thumb_heat_map = self.process_tactile_data(self.thumb_tactile_sensor, (128, 128))

            with self.tac_thumb_lock:
                self.thumb_raw_img = thumb_raw_img
                self.thumb_points = thumb_points
                self.thumb_heat_map = thumb_heat_map
            time.sleep(0.01)


    def process_index_tactile(self):
        # Process index tactile data
        while True:
            index_raw_img, index_points, index_heat_map = self.process_tactile_data(self.index_tactile_sensor, (128, 128))

            with self.tac_index_lock:
                self.index_raw_img = index_raw_img
                self.index_points = index_points
                self.index_heat_map = index_heat_map
            time.sleep(0.01)
    

    def process_middle_tactile(self):
        # Process middle tactile data
        while True:
            middle_raw_img, middle_points, middle_heat_map = self.process_tactile_data(self.middle_tactile_sensor, (128, 128))

            with self.tac_middle_lock:
                self.middle_raw_img = middle_raw_img
                self.middle_points = middle_points
                self.middle_heat_map = middle_heat_map
            time.sleep(0.01)
    

    def reset(self, joint_reset=False, **kwargs):
        print("densoenv reset")


        self.data_count = 0
        self.last_gripper_act = time.time()
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        # self._recover()
        # self.go_to_reset(joint_reset=joint_reset)
        # self._recover()

        self.curr_path_length = 0

        self._update_cur_position()
        obs = self._get_obs()
        self.terminate = False

        # === NEW: 记录 episode 起点 ===
        #if self._cur_ep_start is None:
        #    self._cur_ep_start = self.global_frame_id
        self._cur_ep_start = int(self.global_frame_id)
        self._episode_counter += 1
        print(f"[DensoEnv][reset] _episode_counter={self._episode_counter} _cur_ep_start={self._cur_ep_start} global={self.global_frame_id}")
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None, extra_cameras_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        self.cap_serial = {}
        # for cam_name, kwargs in name_serial_dict.items():
        #     cap = VideoCapture(
        #         RSCapture(name=cam_name, **kwargs)
        #     )
        #     self.cap[cam_name] = cap

        # for cam_name, kwargs in extra_cameras_dict.items():
        #     cap = VideoCapture(
        #         RSCapture(name=cam_name, **kwargs)
        #     )
        #    self.cap[cam_name] = cap
        for cam_name, v in (name_serial_dict or {}).items():
            if isinstance(v, str):
                kwargs = {"serial_number": v}
            else:
                kwargs = dict(v) 
            cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
            self.cap[cam_name] = cap
            self.cap_serial[cam_name] = kwargs.get("serial_number")
        
        for cam_name, v in (extra_cameras_dict or {}).items():
            if isinstance(v, str):
                kwargs = {"serial_number": v}
            else:
                kwargs = dict(v)
            cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
            self.cap[cam_name] = cap
            self.cap_serial[cam_name] = kwargs.get("serial_number")
        
        print("[DensoEnv] Camera map:", {k: self.cap_serial.get(k) for k in self.cap})

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")
    
    def _safe_put(self, q, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                _ = q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except Exception:
                pass
    
    def _probe_highgui(self, title="__sam2_probe__", ms=600):
        import os, time, cv2, numpy as np
        try:
            if os.environ.get("XDG_SESSION_TYPE","") == "wayland" and not os.environ.get("QT_QPA_PLATFORM",""):
                os.environ["QT_QPA_PLATFORM"] = "xcb"
            try:
                cv2.startWindowThread()
            except Exception:
                pass
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 320, 200)
            img = np.zeros((200,320,3), np.uint8)
            cv2.putText(img, "SAM2 UI probe", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow(title, img)
            t0 = time.time()
            ok_once = False
            while (time.time()-t0) < (ms/1000.0):
                k = cv2.waitKey(20)
                # 只要窗口曾经可见过一次，就认为 OK
                try:
                    vis = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
                    if vis >= 1:
                        ok_once = True
                        break
                except Exception:
                    pass
            try:
                cv2.destroyWindow(title)
            except Exception:
                pass
            return ok_once
        except Exception as e:
            print(f"[SAM2 v2][probe] HighGUI probe failed: {e}")
            return False


    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")


    def _send_leap_hand_command(self, leap_hand_action: np.ndarray, steps=10, step_time=0.01):
        """Internal function to send leap hand command to the robot."""
        hand_action = leap_hand_action
        # step_time = 0.01  # Example step time
        # steps = 10    # Example number of steps

        if self.interpolation_thread and self.interpolation_thread.is_alive():
            return
        
        current_pos = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(current_pos, dtype=np.float32).copy()
        # print("in send_leap_hand_command curr_leap_hand_pos = ", self.curr_leap_hand_pos)

        with self.thread_lock:
            self.interpolation_thread = threading.Thread(
                target=self.leap_interpolate_and_publish,
                args=(self.curr_leap_hand_pos, hand_action, step_time, steps),
                daemon=True
            )
            self.interpolation_thread.start()


    def leap_interpolate_and_publish(self, start_position, end_position, step_time, steps):
        for i in range(steps + 1):
            # Interpolate between start and end positions
            interpolated_position = [
                start + (end - start) * (i / steps)
                for start, end in zip(start_position, end_position)
            ]
            # Publish the interpolated position
            self.curr_leap_hand_pos = np.asarray(interpolated_position, dtype=np.float32).copy()
            self.ros_interface.publish_hand_action(interpolated_position)
            # print("in leap_interpolate_and_publish curr_leap_hand_pos = ", self.curr_leap_hand_pos)
            time.sleep(step_time)
            

    def _update_cur_position(self, arm_action, timeout=10.0, wait_threshold=0.05):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        start = time.time()
        self.cur_position, self.cur_oritation = self.ros_interface.get_current_robot_ee()
        joint_position = self.ros_interface.get_current_joint()
        self.joint_position = np.asarray(joint_position, dtype=np.float32).copy()

        hand_joint_msg = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
        # print("in _update_cur_position curr_leap_hand_pos = ", self.curr_leap_hand_pos)

        diff = np.asarray(arm_action[:3], dtype=np.float32) - np.asarray(self.cur_position, dtype=np.float32)
        while np.max(np.abs(diff)) > wait_threshold:
            if time.time() - start > timeout:
                print("[WARN] 等待机械臂到位超时")
                break
            time.sleep(0.02)
            self.cur_position, self.cur_oritation = self.ros_interface.get_current_robot_ee()
            joint_position = self.ros_interface.get_current_joint()
            self.joint_position = np.asarray(joint_position, dtype=np.float32).copy()
            hand_joint_msg = self.ros_interface.get_current_leap_position()
            self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
            diff = np.linalg.norm(arm_action[:3] - self.cur_position)


    def _get_obs(self) -> dict:
        images = self.get_im()
        front_camera_image = images["front_camera"]
        gaze_mask_image = images.get("gaze_mask", None)
        # side_camera_image = images["side_camera"]

        if not self.is_arm_only:
            state_flattened = np.concatenate([
                np.array(self.cur_position, dtype=np.float32).flatten(),  # TCP 位置 (3,)
                np.array(self.cur_oritation, dtype=np.float32).flatten(),  # TCP 旋转 (4,)
                np.array(self.curr_leap_hand_pos, dtype=np.float32).flatten()  # 夹爪 (n,)
            ])
        else :
            state_flattened = np.concatenate([
                np.array(self.cur_position, dtype=np.float32).flatten(),  # TCP 位置 (3,)
                np.array(self.cur_oritation, dtype=np.float32).flatten(),  # TCP 旋转 (4,)
                np.array(self.hand_state, dtype=np.float32).flatten(),  # TCP 旋转 (4,)
            ])
        if self.enable_tactile:
            heatmap_canvas = cv2.hconcat([self.thumb_heat_map, self.index_heat_map, self.middle_heat_map])
            obs = copy.deepcopy({
            "front_camera": front_camera_image,
            # "side_camera": side_camera_image,
            "tactile_data":heatmap_canvas,
            "gaze_mask": gaze_mask_image,
            "state": state_flattened
        })
        else:
            obs = copy.deepcopy({
            "front_camera": front_camera_image,
            # "side_camera": side_camera_image,
            "gaze_mask": gaze_mask_image,
            "state": state_flattened
        })
            
        return obs


    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            # self.img_queue.put(None)
            # cv2.destroyAllWindows()
            # self.displayer.join()
            if hasattr(self, "rgb_queue"):
                try:
                    self.rgb_queue.put(None)
                except Exception:
                    pass
            if hasattr(self, "rgb_viewer"):
                try:
                    self.rgb_viewer.join()
                except Exception:
                    pass

        # Ensure ROS executor and node are cleaned up to avoid threading errors
        if hasattr(self, "executor"):
            if hasattr(self, "ros_interface"):
                # Remove the node from the executor and destroy it before shutting down ROS
                self.executor.remove_node(self.ros_interface)
                self.ros_interface.destroy_node()
            try:
                if getattr(rclpy, "is_initialized", lambda: True)():
                    # Signal the spinning thread to exit
                    rclpy.shutdown()
            except Exception:
                pass
            if hasattr(self, "executor_thread"):
                # Wait for the executor thread to finish before closing it
                self.executor_thread.join()
            self.executor.shutdown()
    def pause_display(self):
        try:
            if hasattr(self, "rgb_queue"):
                try:
                    self.rgb_queue.put(None, timeout=0.1)
                except Exception:
                    pass
            if hasattr(self, "rgb_viewer"):
                try:
                    self.rgb_viewer.join(timeout=1.0)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    
    def resume_display(self):
        if not getattr(self, "display_image", False):
            return
        try:
            import queue as _q
            self.rgb_queue = _q.Queue(maxsize=2)
            self.rgb_viewer = SingleImageDisplayer(
                self.rgb_queue,
                name=f"{self.url}-front-rgb",
                fullscreen=False,
                target_size=(1920, 1080),
                resize_mode="stretch",
                assume_rgb=False, 
            )
            self.rgb_viewer.start()
        except Exception as e:
            print(f"[DensoEnv][resume_display] failed: {e}")
            


    def set_data_count(self, data_count):
        self.data_count = data_count + 1

    def pose_callback(self, msg):
        self.arm_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.arm_orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # self.arm_pose = np.concatenate((position, orientation), axis=0)


    def save_training_frame(self):
        try:
            joint_pose = np.concatenate([
                    self.joint_position, self.curr_leap_hand_pos], dtype=np.float32)
            
            self.joint_buffer.append(
                            copy.deepcopy(joint_pose))
            
            # 保存图像
            images, depth_img = self.get_rgb_and_dpth_im()
            for cam_name, img in images.items():
                depth = depth_img[cam_name]
                if cam_name == "front_camera":
                    self.front_color_buffer.append(copy.deepcopy(img[..., ::-1]))
                    self.front_depth_buffer.append(copy.deepcopy(depth))
                elif cam_name == "side_camera":
                    self.side_color_buffer.append(copy.deepcopy(img[..., ::-1]))
                    self.side_depth_buffer.append(copy.deepcopy(depth))

            if self.enable_tactile:
                self.rthumb_raw_buffer.append(copy.deepcopy(self.thumb_raw_img))
                self.rindex_raw_buffer.append(copy.deepcopy(self.index_raw_img))
                self.rmiddle_raw_buffer.append(copy.deepcopy(self.middle_raw_img))

                self.rthumb_heatmap_buffer.append(copy.deepcopy(self.thumb_heat_map))
                self.rindex_heatmap_buffer.append(copy.deepcopy(self.index_heat_map))
                self.rmiddle_heatmap_buffer.append(copy.deepcopy(self.middle_heat_map))

        except Exception as e:
            print("An error occurred while processing frames")
            print(e)

        # === NEW: 镜像 + pupil_gaze.json ===
        try:
            # rs_images
            if "front_camera" in images:
                rs_bgr = images["front_camera"][..., ::-1]  
                cv2.imwrite(os.path.join(self.rs_mirror_dir, f"{self.global_frame_id}.jpg"), rs_bgr)
            # et_images
            if self._pupil_last_world_payload is not None:
                et_img = cv2.imdecode(np.frombuffer(self._pupil_last_world_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if et_img is not None:
                    cv2.imwrite(os.path.join(self.et_mirror_dir, f"{self.global_frame_id}.jpg"), et_img)
            # 每帧目录：pupil_gaze.json
            fdir = os.path.join(self.frame_root, f"frame_{self.global_frame_id}")
            os.makedirs(fdir, exist_ok=True)
            if self._pupil_last_gaze is not None:
                with open(os.path.join(fdir, "pupil_gaze.json"), "w") as f:
                    json.dump(self._pupil_last_gaze, f, indent=2)
        except Exception as e:
            print(f"[save_training_frame][mirror/gaze] {e}")

        # === 全局帧号自增 ===
        self.global_frame_id = int(self.global_frame_id) + 1


    def save_all_data_on_exit(self):
        for frame_id in range(self.frame_count):
            print("save frame ", frame_id)
            frame_dir = os.path.join(self.frame_save_path, f"frame_{frame_id}")
            os.makedirs(frame_dir, exist_ok=True)

            cv2.imwrite(os.path.join(frame_dir, "color_image.jpg"), self.front_color_buffer[frame_id])
            cv2.imwrite(os.path.join(frame_dir, "depth_image.png"), self.front_depth_buffer[frame_id])
            cv2.imwrite(os.path.join(frame_dir, "color_image2.jpg"), self.side_color_buffer[frame_id])
            cv2.imwrite(os.path.join(frame_dir, "depth_image2.png"), self.side_depth_buffer[frame_id])

            if self.enable_tactile:
                cv2.imwrite(os.path.join(frame_dir, "thumb_raw_image.jpg"), self.rthumb_raw_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "thumb_heat_map.jpg"), self.rthumb_heatmap_buffer[frame_id])
            
                cv2.imwrite(os.path.join(frame_dir, "index_raw_image.jpg"), self.rindex_raw_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "index_heat_map.jpg"), self.rindex_heatmap_buffer[frame_id])
            
                cv2.imwrite(os.path.join(frame_dir, "middle_raw_image.jpg"), self.rmiddle_raw_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "middle_heat_map.jpg"), self.rmiddle_heatmap_buffer[frame_id])
            # 保存 state（TCP + orientation + hand joints）
            # if not self.is_arm_only:
            np.savetxt(os.path.join(frame_dir, "right_arm_joint.txt"), self.joint_buffer[frame_id])

    # === NEW: Pupil 监听线程 ===
    def _pupil_listen(self):
        if self._pupil_sub is None:
            return
        while True:
            try:
                parts = self._pupil_sub.recv_multipart()
                if not parts:
                    continue
                topic = parts[0].decode("utf-8")
                if topic.startswith("gaze."):
                    header = parts[1]
                    data = msgpack.loads(header, raw=False)
                    self._pupil_last_gaze = {
                        "topic": topic,
                        "ts": data.get("timestamp", None),
                        "data": data
                    }
                elif topic == "frame.world":
                    if len(parts) >= 3:
                        self._pupil_last_world_payload = parts[2]
                    else:
                        self._pupil_last_world_payload = parts[1]
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[PupilListen] {e}")
                continue

    # === NEW: episode 结束 ===
    def end_episode_and_collect(self):
        import numpy as np

        def _as_int_scalar(x, name):
            # 把任何“看起来像整数”的东西，强行变成 Python int；否则抛清晰的错误
            if isinstance(x, (int, np.integer)):
                return int(x)
            if isinstance(x, np.ndarray):
                if x.shape == ():          # 0-D 标量
                    return int(x.item())
                if x.size == 1:            # 形如 [123]
                    return int(x.reshape(()).item())
                # 不是标量，直接抛错（上层会打印出来）
                raise ValueError(f"{name} expected scalar, got array with shape {x.shape} and dtype {x.dtype}")
            # 其它可转类型
            try:
                return int(x)
            except Exception as e:
                raise ValueError(f"{name} cannot be cast to int: {type(x)} / {x!r}") from e

        try:
            # 读取起点/终点并强制为纯 Python int
            s_raw = self._cur_ep_start
            e_raw = self.global_frame_id

            # 调试：看看类型（出现问题时一眼就知道是哪里变成了 array）
            print(f"[end_ep][debug] _cur_ep_start type={type(s_raw)}, val={repr(s_raw)}")
            print(f"[end_ep][debug] global_frame_id type={type(e_raw)}, val={repr(e_raw)}")

            if s_raw is None:
                return None

            s = _as_int_scalar(s_raw, "start")
            e = _as_int_scalar(e_raw, "end") - 1

            if e >= s:
                rng = (s, e)
                self.episode_frame_ranges.append(rng)
            else:
                rng = None
            return rng
        finally:
            # 不管成功失败，都清空起点，避免后续污染
            self._cur_ep_start = None

        
    
    # === NEW: 关键帧选择（y_total 均匀分配 + 首尾） ===
    def _select_keyframes(
        self,
        frame_ranges: List[Tuple[int,int]],
        y_total: int | None = 20,
        min_per_ep: int = 2,
        max_per_ep: int = 5,
        include_ends: bool = True,
        prefer_gaze: bool = False,
        random_extra: bool = False,
        random_seed: int = 0,
    ) -> list[int]:
        import math, time
        t0 = time.time()
        print("[FAST_SEL] ENTER ranges=", frame_ranges, "y_total=", y_total, flush=True)

        # 标准化各 episode 的 [s,e,l]
        eps = []
        total_len = 0
        for s, e in frame_ranges:
            s, e = int(s), int(e)
            if e < s:
                s, e = e, s
            l = max(1, e - s + 1)
            eps.append((s, e, l))
            total_len += l

        # 预算分配
        if y_total is None:
            # 没给总预算：按每个 ep 至少 min、至多 max 粗分
            per_ep = [max(min_per_ep, min(max_per_ep, l // max(1, l // 3))) for _, _, l in eps]
        else:
            y_budget = max(len(eps) * min_per_ep, int(y_total))
            # 按长度比例分配，再微调到 [min,max]
            per_raw = [l / max(1, total_len) * y_budget for _, _, l in eps]
            per_ep = []
            for (s, e, l), pr in zip(eps, per_raw):
                q = int(round(pr))
                q = max(min_per_ep, min(max_per_ep, q))
                per_ep.append(q)

            # 把四舍五入带来的误差回收一下
            delta = y_budget - sum(per_ep)
            if delta != 0 and len(per_ep) > 0:
                attempts = 0
                changed = True
                while delta != 0 and attempts < len(per_ep) * 2 and changed:
                    changed = False
                    for i in range(len(per_ep)):
                        if delta > 0 and per_ep[i] < max_per_ep:
                            per_ep[i] += 1
                            delta -= 1
                            changed = True
                            if delta == 0:
                                break
                        elif delta < 0 and per_ep[i] > min_per_ep:
                            per_ep[i] -= 1
                            delta += 1
                            changed = True
                            if delta == 0:
                                break
                    attempts += 1
                if delta != 0:
                    print(f"[FAST_SEL] WARN delta leftover={delta}, ignore it.", flush=True)
                
        # 逐 episode 均匀取点
        selected = set()
        for (s, e, l), q in zip(eps, per_ep):
            if q <= 0:
                continue
            want = q

            # 先把首尾放进去（可选）
            chosen = []
            if include_ends:
                chosen.append(s)
                if e != s:
                    chosen.append(e)
            chosen = list(dict.fromkeys(chosen))  # 去重
            need = max(0, want - len(chosen))

            if need > 0:
                if l <= len(chosen) + need:
                    # 很短，就全取了（除了已经取过的）
                    mids = [i for i in range(s, e + 1) if i not in chosen]
                else:
                    # 纯数值均匀采样（不触盘）
                    stride = (l - 1) / float(need + 1)
                    mids = []
                    for k in range(1, need + 1):
                        idx = int(round(s + k * stride))
                        idx = max(s, min(e, idx))
                        if idx not in chosen:
                            mids.append(idx)

                # 补齐
                for m in mids:
                    if len(chosen) >= want:
                        break
                    if m not in chosen:
                        chosen.append(m)

            selected.update(chosen)

        out = sorted(selected)
        print(f"[FAST_SEL] EXIT n={len(out)} took={time.time()-t0:.3f}s", flush=True)
        return out
        """         import random
        rng = random.Random(random_seed)
            

        def has_gaze(idx: int) -> bool:
            p = pathlib.Path(self.frame_root) / f"frame_{idx}" / "pupil_gaze.json"
            if not p.exists(): return False
            try:
                d = json.loads(p.read_text())
                dd = d.get("data", d)
                return "norm_pos" in dd
            except Exception:
                return False

        eps = []
        total_len = 0
        for s,e in frame_ranges:
            s,e = int(s), int(e)
            if e < s: s,e = e,s
            l = max(1, e - s + 1)
            eps.append((s,e,l))
            total_len += l

        if y_total is None:
            y_budget = sum(min(max_per_ep, max(min_per_ep, l // max(1, l // 3))) for _,_,l in eps)
        else:
            y_budget = max(len(eps)*min_per_ep, y_total)

        per_ep=[]
        for s,e,l in eps:
            q = int(round(y_budget * (l/max(1,total_len))))
            q = max(min_per_ep, min(max_per_ep, q))
            per_ep.append(q)
        delta = y_budget - sum(per_ep)
        i=0
        while delta!=0 and len(per_ep)>0:
            can = (per_ep[i] < max_per_ep) if delta>0 else (per_ep[i] > min_per_ep)
            if can:
                per_ep[i] += 1 if delta>0 else -1
                delta += (-1 if delta>0 else 1)
            i=(i+1)%len(per_ep)

        selected=set()
        for (s,e,l), q in zip(eps, per_ep):
            chosen=set()
            if include_ends:
                chosen.add(s)
                if e!=s: chosen.add(e)
            need = max(0, q - len(chosen))
            if need>0:
                if need >= (l - len(chosen)):
                    base = [i for i in range(s,e+1) if i not in chosen]
                else:
                    stride = max(1, (l-1)//max(1, need+1))
                    base=[]; cur=s+stride
                    while cur<e and len(base)<need*2:
                        if cur not in chosen: base.append(cur)
                        cur+=stride
                if prefer_gaze:
                    gf=[i for i in base if has_gaze(i)]
                    ng=[i for i in base if i not in gf]
                else:
                    gf=[]; ng=base
                take=[]
                take.extend(gf[:need])
                if len(take)<need: take.extend(ng[:(need-len(take))])
                for i in take: chosen.add(i)
            selected.update(chosen)

        if len(selected)>y_budget:
            sel=sorted(selected)
            gf=[i for i in sel if has_gaze(i)]
            ng=[i for i in sel if i not in gf]
            keep=[]
            keep.extend(gf[:y_budget])
            need=y_budget-len(keep)
            if need>0:
                stride=max(1, len(ng)//max(1,need))
                keep.extend(ng[::stride][:need])
            return sorted(set(keep))
        return sorted(selected) """
    
        

    # === NEW (SAM2): 交互标注关键帧（ET） ===
    def _sam2_annotate_keyframes(self, et_file_list: list[str], key_indices: list[int]):
        self._sam2_prompts_per_frame = {}
        self._sam2_active_obj = 0
        self._sam2_temp_box = []
        self._sam2_temp_box_end = None
        self._sam2_is_dragging = False

        def _on_mouse(event, x, y, flags, param):
            idx = param["cur_idx"]
            d = self._sam2_prompts_per_frame.setdefault(idx, {})
            if self._sam2_active_obj not in d:
                d[self._sam2_active_obj] = {"clicks":[], "labels":[], "boxes":[]}
            entry = d[self._sam2_active_obj]
            if event == cv2.EVENT_LBUTTONDOWN:
                self._sam2_temp_box = [(x,y)]
                self._sam2_temp_box_end = (x,y)
                self._sam2_is_dragging = True
            elif event == cv2.EVENT_MOUSEMOVE and self._sam2_is_dragging and len(self._sam2_temp_box)==1:
                self._sam2_temp_box_end=(x,y)
            elif event == cv2.EVENT_LBUTTONUP and len(self._sam2_temp_box)==1:
                self._sam2_temp_box.append((x,y))
                entry["boxes"].append(list(self._sam2_temp_box))
                self._sam2_temp_box=[]; self._sam2_temp_box_end=None; self._sam2_is_dragging=False
            elif event == cv2.EVENT_LBUTTONDBLCLK:
                entry["clicks"].append([x,y]); entry["labels"].append(1)
            elif event == cv2.EVENT_MBUTTONDOWN:
                entry["clicks"].append([x,y]); entry["labels"].append(0)

        win="SAM2-ET Prompt"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        cur_i = 0
        while True:
            kf = key_indices[cur_i]
            img_path = os.path.join(self.et_mirror_dir, f"{kf}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                # 如关键帧没有 pupil_world.jpg，就跳过这帧
                if cur_i < len(key_indices)-1:
                    cur_i += 1
                    continue
                else:
                    break
            vis = img.copy()
            d = self._sam2_prompts_per_frame.get(kf, {})
            for oid, rec in d.items():
                for (px,py), lbl in zip(rec.get("clicks",[]), rec.get("labels",[])):
                    color = (0,255,0) if lbl else (0,0,255)
                    cv2.circle(vis, (px,py), 6, color, -1)
                for b in rec.get("boxes",[]):
                    if len(b)==2:
                        cv2.rectangle(vis, b[0], b[1], (255,0,0), 2)
            if len(self._sam2_temp_box)==1 and self._sam2_temp_box_end is not None:
                cv2.rectangle(vis, self._sam2_temp_box[0], self._sam2_temp_box_end, (200,200,0), 1)

            msg=f"key {cur_i+1}/{len(key_indices)} frame={kf} obj={self._sam2_active_obj}  (j/k prev/next, n new obj, Enter finish)"
            cv2.putText(vis, msg, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
            cv2.setMouseCallback(win, _on_mouse, param={"cur_idx":kf})
            cv2.imshow(win, vis)
            key=cv2.waitKey(20)&0xFF
            if key in (13,10): break
            elif key==ord('n'): self._sam2_active_obj += 1
            elif key==ord('j') and cur_i>0: cur_i -= 1
            elif key==ord('k') and cur_i<len(key_indices)-1: cur_i += 1
        cv2.destroyWindow(win)
        
    def _load_sam2_predictor(self, default_repo: str):
        ckpt = os.environ.get("SAM2_CKPT", "").strip()
        use_id = ckpt if ckpt else default_repo
        print(f"[SAM2] loading from: {use_id}")
        return SAM2VideoPredictor.from_pretrained(
            use_id,
            config_overrides={"inference": {"multimask_output": False}})
    
    def _force_model_fp32(self, predictor):
        """把 SAM2 模型的所有参数 & buffers 都转成 float32，避免 bf16/float 冲突。"""
        import torch
        # 常见：predictor.model 或 predictor.sam_model
        model = getattr(predictor, "model", None)
        if model is None:
            model = getattr(predictor, "sam_model", None)
        if model is None:
            return predictor  # 找不到就跳过

        model.to(dtype=torch.float32)
        for p in model.parameters(recurse=True):
            if p.dtype != torch.float32:
                p.data = p.data.float()
        for b in model.buffers(recurse=True):
            if hasattr(b, "dtype") and b.dtype != torch.float32:
                b.data = b.data.float()
        return predictor

    def _force_tensor_images_fp32(self, state):
        """把 init_state 里的 images 统一成列表/字典的 Tensor(C,H,W) float32."""
        import torch
        def _fix_one(x):
            if isinstance(x, torch.Tensor):
                if x.ndim == 3 and x.shape[0] in (1,3,4):
                    return x.to(dtype=torch.float32)
                if x.ndim == 4 and x.shape[0] in (1,3,4):  # 万一是 (C,H,W,1) 误进来
                    return x.squeeze(-1).to(dtype=torch.float32)
            return x

        imgs = state.get("images", None)
        if imgs is None:
            return state
        if isinstance(imgs, list):
            state["images"] = [_fix_one(t) for t in imgs]
        elif isinstance(imgs, dict):
            state["images"] = {k: _fix_one(v) for k, v in imgs.items()}
        return state
    
    
    def _deep_to_fp32(self, obj):
        """递归把任意 nn.Module / Tensor / dict / list / tuple 里的 bf16/half 全部转成 float32。"""
        import torch, types
        from torch import nn

        if isinstance(obj, torch.nn.Module):
            obj.to(dtype=torch.float32)
            for p in obj.parameters(recurse=True):
                if p.dtype != torch.float32:
                    p.data = p.data.float()
            for b in obj.buffers(recurse=True):
                if hasattr(b, "dtype") and b.dtype != torch.float32:
                    b.data = b.data.float()
            # 继续扫子属性里未注册的 tensor
            for k, v in obj.__dict__.items():
                if k.startswith("_") or k in ("_parameters","_buffers","_modules"):
                    continue
                self._deep_to_fp32(v)
            return obj

        elif isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = self._deep_to_fp32(obj[i])
            return obj
        elif isinstance(obj, tuple):
            return tuple(self._deep_to_fp32(x) for x in obj)
        elif isinstance(obj, dict):
            for k in list(obj.keys()):
                obj[k] = self._deep_to_fp32(obj[k])
            return obj
        else:
            try:
                import torch
                if isinstance(obj, torch.Tensor) and obj.dtype != torch.float32:
                    return obj.float()
            except Exception:
                pass
            return obj

    def _force_predictor_all_fp32(self, predictor):
        """
        彻底把 predictor 及其所有子模块/张量转成 float32。
        有些实现不仅仅挂在 predictor.model，还会挂 buffer 在其它属性里，这里全量递归。
        """
        # 先处理常见主模型
        model = getattr(predictor, "model", None) or getattr(predictor, "sam_model", None)
        if model is not None:
            self._deep_to_fp32(model)
        # 再把 predictor 其它挂载的对象也递归转
        self._deep_to_fp32(predictor)
        return predictor



    # === NEW (SAM2): 传播并保存（et/rs 两侧） ===
    def _sam2_propagate_and_save(self, mode: str, mirror_dir: str, frame_root: str, selected_indices: list[int], save_union: bool=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = self._load_sam2_predictor("facebook/sam2-hiera-small")
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device=="cuda" else torch.float32):
            state = predictor.init_state(mirror_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)
        if mode == "et":
            for frame_idx, per_obj in getattr(self, "_sam2_prompts_per_frame", {}).items():
                for obj_id, d in per_obj.items():
                    if d.get("clicks"):
                        predictor.add_new_points_or_box(
                            state, frame_idx=frame_idx, obj_id=obj_id,
                            points=np.array(d["clicks"], dtype=np.float32),
                            labels=np.array(d["labels"], dtype=np.int64),
                            box=None
                        )
                    for box in d.get("boxes", []):
                        if len(box)==2:
                            (x0,y0),(x1,y1)=box
                            predictor.add_new_points_or_box(
                                state, frame_idx=frame_idx, obj_id=obj_id,
                                points=None, labels=None, box=np.array([x0,y0,x1,y1], dtype=np.float32)
                            )
        prefix="et_" if mode=="et" else "rs_"
        keep=set(selected_indices)
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            if frame_idx not in keep: 
                continue
            fdir = pathlib.Path(frame_root)/f"frame_{frame_idx}"
            fdir.mkdir(parents=True, exist_ok=True)
            union = None
            for i, oid in enumerate(obj_ids):
                mk = (masks[i].cpu().numpy()>0).astype(np.uint8).squeeze()
                cv2.imwrite(str(fdir / f"{prefix}mask_obj{oid}.png"), mk*255)
                if save_union:
                    if union is None: union=mk.copy()
                    else: union |= mk
            if save_union and union is not None:
                cv2.imwrite(str(fdir / f"{prefix}mask_union.png"), union*255)

    # === NEW: 构建 gaze_contact.json（flip-y），与离线一致 ===
    def _build_gaze_contacts_subset(self, frame_root: str, selected_indices: list[int],
                                    et_to_rs_map: dict[int,int], flip_y=True, eye_wh=(1280,720),
                                    gaze_hit_radius_px: int = 40):
        ET_MASK_PREFIXES = ["et_", ""]
        def _read_mask(frame_dir: pathlib.Path, oid: int):
            for pref in ET_MASK_PREFIXES:
                p = frame_dir / f"{pref}mask_obj{oid}.png"
                if p.exists():
                    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if m is not None: 
                        return (m>0).astype(np.uint8)
            return None

        def _parse_gaze(frame_dir: pathlib.Path):
            f = frame_dir / "pupil_gaze.json"
            if not f.exists(): return None
            try:
                d = json.loads(f.read_text())
            except Exception:
                return None
            W,H = eye_wh
            if isinstance(d, dict):
                dd = d.get("data", {})
                ts = dd.get("timestamp", d.get("ts", d.get("timestamp", None)))
                if "norm_pos" in dd:
                    xn,yn = dd["norm_pos"]
                    u = int(round(xn*W))
                    v = int(round((1.0-yn)*H)) if flip_y else int(round(yn*H))
                    return u,v,ts
                if "norm_pos" in d:
                    xn,yn = d["norm_pos"]
                    u = int(round(xn*W))
                    v = int(round((1.0-yn)*H)) if flip_y else int(round(yn*H))
                    ts = d.get("timestamp", ts)
                    return u,v,ts
            return None

        def _choose_hit(et_masks: dict[int, np.ndarray], u: int, v: int, radius_px: int):
            """
            命中规则：
            1) 先对每个对象掩码做半径=radius_px的椭圆膨胀，若(u,v)落在膨胀后的掩码中，则视为命中（等价“gaze点变大”）。
            2) 若命中多个对象，用“到原始前景的距离”最小者（离哪个物体最近）来打破平局。
            """
            if not et_masks: 
                return None

            # 结构元素：半径 r 的椭圆
            ksz = 2*radius_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

            # 预计算：膨胀掩码 + 到原始前景的距离(用对 ~mask 的 distanceTransform 得最近前景距离)
            dilated = {}
            dist_to_fg = {}
            for oid, m in et_masks.items():
                dilated[oid] = cv2.dilate(m, kernel, iterations=1)

                inv = (1 - m).astype(np.uint8)  # 前景为1的“距离”不好算，转而对背景=1做DT得到到前景的距离
                # OpenCV的 distanceTransform 要求非零像素为“可走区域”，所以这里用背景做DT，得到每个点到前景的欧氏距离
                dist_to_fg[oid] = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

            H, W = next(iter(et_masks.values())).shape[:2]
            if not (0 <= u < W and 0 <= v < H):
                return None

            # 第一步：检查膨胀后的命中
            cands = [oid for oid, dm in dilated.items() if dm[v, u] > 0]
            if not cands:
                return None
            if len(cands) == 1:
                return cands[0]

            # 第二步：若多个命中，选“到原始前景距离最小”的那个（谁最近选谁）
            best_oid = None
            best_d = float("inf")
            for oid in cands:
                d = float(dist_to_fg[oid][v, u])
                if d < best_d:
                    best_d = d
                    best_oid = oid
            return best_oid

        for idx in selected_indices:
            fdir = pathlib.Path(frame_root)/f"frame_{idx}"
            if not fdir.exists(): 
                continue

            parsed = _parse_gaze(fdir)
            u=v=ts=None
            if parsed is not None:
                u,v,ts = parsed

            et_masks={}
            for et_oid in et_to_rs_map.keys():
                m = _read_mask(fdir, et_oid)
                if m is not None: et_masks[et_oid]=m

            et_oid=None
            if u is not None and v is not None and et_masks:
                et_oid=_choose_hit(et_masks, u, v, gaze_hit_radius_px)

            rs_oid = et_to_rs_map.get(et_oid, None) if et_oid is not None else None
            label = {
                "timestamp": ts,
                "gaze_uv_in_eye": None if u is None else [int(u), int(v)],
                "et_object_id": None if et_oid is None else int(et_oid),
                "rs_object_id": None if rs_oid is None else int(rs_oid),
                "class_id": None if rs_oid is None else int(rs_oid),
                "class_name": None if rs_oid is None else f"obj{int(rs_oid)}",
                "hit": bool(et_oid is not None)
            }
            (fdir / "gaze_contact.json").write_text(json.dumps(label, indent=2, ensure_ascii=False))


    def load_gaze_contact_labels(self, frame_ranges: List[Tuple[int,int]]) -> dict[int, int | None]:
        out={}
        for s,e in frame_ranges:
            s,e=int(s),int(e)
            if e<s: s,e=e,s
            for i in range(s,e+1):
                p=pathlib.Path(self.frame_root)/f"frame_{i}"/"gaze_contact.json"
                if not p.exists():
                    out[i]=None
                    continue
                try:
                    d=json.loads(p.read_text())
                    cid=d.get("class_id", None)
                    out[i]=cid
                except Exception:
                    out[i]=None
        return out
    
        # === NEW v2: 通用标注器（可用于 ET 或 RS），独立于旧版 ===
    def _sam2_annotate_keyframes_generic(self, mirror_dir: str, key_indices: list[int], mode: str = "et"):
        """
        与旧版类似，但：
        - 仅显示一个视角（mirror_dir 指向 et_images 或 rs_images）
        - prompts 独立存放：ET → self._sam2_prompts_et，RS → self._sam2_prompts_rs
        - 所有 OpenCV GUI 调用均使用全局 CV_UI_LOCK 进行串行化，避免与其它窗口冲突
        """
        # 选择 prompts 容器
        if mode.lower() == "et":
            if not hasattr(self, "_sam2_prompts_et"):
                self._sam2_prompts_et = {}
            prompts_dict = self._sam2_prompts_et
        else:
            if not hasattr(self, "_sam2_prompts_rs"):
                self._sam2_prompts_rs = {}
            prompts_dict = self._sam2_prompts_rs

        # 交互状态
        self._sam2_active_obj_v2 = 0
        self._sam2_temp_box_v2 = []
        self._sam2_temp_box_end_v2 = None
        self._sam2_is_dragging_v2 = False

        def _on_mouse(event, x, y, flags, param):
            idx = param["cur_idx"]
            d = prompts_dict.setdefault(idx, {})
            if self._sam2_active_obj_v2 not in d:
                d[self._sam2_active_obj_v2] = {"clicks": [], "labels": [], "boxes": []}
            entry = d[self._sam2_active_obj_v2]

            if event == cv2.EVENT_LBUTTONDOWN:
                self._sam2_temp_box_v2 = [(x, y)]
                self._sam2_temp_box_end_v2 = (x, y)
                self._sam2_is_dragging_v2 = True

            elif event == cv2.EVENT_MOUSEMOVE and self._sam2_is_dragging_v2 and len(self._sam2_temp_box_v2) == 1:
                self._sam2_temp_box_end_v2 = (x, y)

            elif event == cv2.EVENT_LBUTTONUP and len(self._sam2_temp_box_v2) == 1:
                self._sam2_temp_box_v2.append((x, y))
                entry["boxes"].append(list(self._sam2_temp_box_v2))
                self._sam2_temp_box_v2.clear()
                self._sam2_temp_box_end_v2 = None
                self._sam2_is_dragging_v2 = False

            elif event == cv2.EVENT_LBUTTONDBLCLK:
                # 正样本点
                entry["clicks"].append([x, y])
                entry["labels"].append(1)

            elif event == cv2.EVENT_MBUTTONDOWN:
                # 负样本点
                entry["clicks"].append([x, y])
                entry["labels"].append(0)

        title = f"SAM2 Prompt - {mode.upper()}"

        # 创建窗口（加锁）
        with CV_UI_LOCK:
            try:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            except Exception:
                # 防守式处理：已存在/被占用等情况
                try:
                    cv2.destroyWindow(title)
                except Exception:
                    pass
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)

        cur_i = 0
        n_keys = len(key_indices)
        # 主循环
        while True:
            kf = key_indices[cur_i]
            img_path = os.path.join(mirror_dir, f"{kf}.jpg")
            img = cv2.imread(img_path)

            if img is None:
                # 此关键帧缺图则跳到下一帧；若已到最后则退出
                if cur_i < n_keys - 1:
                    cur_i += 1
                    continue
                else:
                    break

            vis = img.copy()

            # 已有标注可视化
            d = prompts_dict.get(kf, {})
            for oid, rec in d.items():
                for (px, py), lbl in zip(rec.get("clicks", []), rec.get("labels", [])):
                    color = (0, 255, 0) if lbl else (0, 0, 255)
                    cv2.circle(vis, (px, py), 6, color, -1)
                for b in rec.get("boxes", []):
                    if len(b) == 2:
                        cv2.rectangle(vis, b[0], b[1], (255, 0, 0), 2)

            # 拖拽中的临时框
            if len(self._sam2_temp_box_v2) == 1 and self._sam2_temp_box_end_v2 is not None:
                cv2.rectangle(vis, self._sam2_temp_box_v2[0], self._sam2_temp_box_end_v2, (200, 200, 0), 1)

            msg = (
                f"[{mode}] key {cur_i + 1}/{n_keys} frame={kf} obj={self._sam2_active_obj_v2}  "
                f"(j/k:prev/next, n:new obj, Enter:finish)"
            )
            cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 串行化 GUI 调用（包含 setMouseCallback / imshow / waitKey）
            with CV_UI_LOCK:
                cv2.setMouseCallback(title, _on_mouse, param={"cur_idx": kf})
                cv2.imshow(title, vis)
                key = cv2.waitKey(20) & 0xFF  # 20ms 轮询一遍键盘和鼠标事件

            # 键盘控制（不加锁，只有读 key 值）
            if key in (13, 10):  # Enter
                break
            elif key == ord('n'):
                self._sam2_active_obj_v2 += 1
            elif key == ord('j') and cur_i > 0:
                cur_i -= 1
            elif key == ord('k') and cur_i < n_keys - 1:
                cur_i += 1

        # 销毁窗口（加锁）
        with CV_UI_LOCK:
            try:
                cv2.destroyWindow(title)
            except Exception:
                pass


    def _force_tensor_images(self, state):
        """
        将 state['images'] 统一成 list[Tensor(C,H,W), uint8]，并在 CPU 上常驻。
        SAM-2 内部会再 to(device)。
        支持以下原始格式：
        - list[str] (文件名) / list[np.ndarray] / list[Tensor] / dict[int, ...]
        - Tensor(N,C,H,W) 也接受，拆成 list
        """
        import torch, numpy as np, cv2
        imgs = state.get("images", None)
        if imgs is None:
            return state

        def _to_tensorCHW_uint8(x):
            if isinstance(x, torch.Tensor):
                t = x.detach().cpu()
                if t.ndim == 3 and t.shape[0] in (1,3):  # C,H,W
                    if t.dtype != torch.uint8:
                        t = t.clamp(0,255).to(torch.uint8)
                    return t
                if t.ndim == 3 and t.shape[2] in (1,3):  # H,W,C
                    t = t.permute(2,0,1).contiguous()
                    if t.dtype != torch.uint8:
                        t = t.clamp(0,255).to(torch.uint8)
                    return t
                if t.ndim == 2:  # H,W
                    t = t.unsqueeze(0)  # 1,H,W
                    if t.dtype != torch.uint8:
                        t = t.clamp(0,255).to(torch.uint8)
                    return t
                raise TypeError(f"unsupported tensor shape: {tuple(t.shape)}")
            elif isinstance(x, np.ndarray):
                arr = x
                if arr.ndim == 3 and arr.shape[2] in (1,3):   # H,W,C
                    if arr.dtype != np.uint8:
                        arr = arr.clip(0,255).astype(np.uint8)
                    return torch.from_numpy(arr).permute(2,0,1).contiguous()
                if arr.ndim == 2:                             # H,W
                    if arr.dtype != np.uint8:
                        arr = arr.clip(0,255).astype(np.uint8)
                    return torch.from_numpy(arr).unsqueeze(0)
                raise TypeError(f"unsupported ndarray shape: {arr.shape}")
            elif isinstance(x, str):
                img = cv2.imread(x, cv2.IMREAD_COLOR)  # BGR
                if img is None:
                    raise FileNotFoundError(f"image not found: {x}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(img).permute(2,0,1).contiguous().to(torch.uint8)
            else:
                raise TypeError(f"unsupported image type: {type(x)}")

        # 统一展开
        if isinstance(imgs, torch.Tensor):
            if imgs.ndim != 4 or imgs.shape[1] not in (1,3):
                raise TypeError(f"sam2 unsupported tensor images shape: {tuple(imgs.shape)}")
            imgs_list = [imgs[i].detach().cpu().to(torch.uint8) for i in range(imgs.shape[0])]
        elif isinstance(imgs, dict):
            # dict[int -> any]
            # 需要按帧序排序
            keys = sorted(int(k) for k in imgs.keys())
            imgs_list = [_to_tensorCHW_uint8(imgs[k]) for k in keys]
        elif isinstance(imgs, (list, tuple)):
            imgs_list = [_to_tensorCHW_uint8(x) for x in imgs]
        else:
            raise TypeError(f"unsupported images container type: {type(imgs)}")

        state["images"] = imgs_list
        # 额外日志：确认一张示例
        sample = imgs_list[0]
        print(f"[SAM2] images fixed to list/tensor; sample: {type(sample)} {tuple(sample.shape)} {sample.dtype}")
        return state

        
    def _sam2_propagate_and_save_with_prompts(
        self, mode: str, mirror_dir: str, frame_root: str,
        selected_indices: list[int], prompts_dict: dict,
        save_union: bool = True
    ):
        import numpy as np, cv2, pathlib, torch
        from contextlib import nullcontext
        from torch.amp import autocast

        predictor = self._load_sam2_predictor("facebook/sam2-hiera-small")
        # predictor = self._force_predictor_all_fp32(predictor)  # 你已有的函数

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 统一禁用半精，避免 dtype 冲突
        amp_ctx = autocast('cuda', dtype=torch.float32, enabled=(device == "cuda")) if device == "cuda" else nullcontext()

        with torch.inference_mode(), amp_ctx:
            state = predictor.init_state(
                mirror_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )

        # 统一 images 类型（你已有）
        # state = self._force_tensor_images(state)
        # state = self._force_tensor_images_fp32(state)

        # -------- 规范 prompts：key 转 int，并构造“白名单” --------
        if not isinstance(prompts_dict, dict):
            prompts_dict = {}
        else:
            prompts_dict = {int(k): v for k, v in prompts_dict.items() if isinstance(k, (int, str))}

        allowed_oids = set()   # 这批实际播种过的对象（只要有点或有框就算）
        # 先检查 & 记录 allowed_oids，再真正 add，避免漏数
        for frame_idx, per_obj in prompts_dict.items():
            if not isinstance(per_obj, dict):
                continue
            for obj_id, d in per_obj.items():
                try:
                    oid = int(obj_id)
                except Exception:
                    continue
                clicks = d.get("clicks", []) or []
                boxes  = d.get("boxes", []) or []
                if len(clicks) > 0 or any(isinstance(b, (list,tuple)) and len(b) == 2 for b in boxes):
                    allowed_oids.add(oid)

        # 如果本批没有任何有效对象，直接返回
        if len(allowed_oids) == 0:
            print("[SAM2] no valid prompts in this batch; skip propagation.")
            return

        # 播种（只对 allowed_oids 播）
        with torch.inference_mode(), amp_ctx:
            for frame_idx, per_obj in prompts_dict.items():
                if not isinstance(per_obj, dict):
                    continue
                fidx = int(frame_idx)
                # 边界保护（若你用自读镜像）
                # if fidx < 0 or fidx >= len(state["images"]): continue

                for obj_id, d in per_obj.items():
                    try:
                        oid = int(obj_id)
                    except Exception:
                        continue
                    if oid not in allowed_oids:
                        continue

                    clicks = d.get("clicks", []) or []
                    labels = d.get("labels", []) or []
                    if len(clicks) > 0:
                        pts = np.asarray(clicks, dtype=np.float32)
                        lbs = np.asarray(labels if len(labels) == len(clicks) else [1]*len(clicks), dtype=np.int64)
                        predictor.add_new_points_or_box(
                            state, frame_idx=fidx, obj_id=oid,
                            points=pts, labels=lbs, box=None
                        )

                    for box in d.get("boxes", []) or []:
                        if isinstance(box, (list, tuple)) and len(box) == 2:
                            (x0, y0), (x1, y1) = box
                            # 规范化 & 裁剪到图像范围
                            if x0 > x1: x0, x1 = x1, x0
                            if y0 > y1: y0, y1 = y1, y0
                            H, W = state["images"][fidx].shape[-2:]
                            x0 = float(max(0, min(x0, W-1)))
                            x1 = float(max(0, min(x1, W-1)))
                            y0 = float(max(0, min(y0, H-1)))
                            y1 = float(max(0, min(y1, H-1)))
                            b = np.array([x0, y0, x1, y1], dtype=np.float32)
                            predictor.add_new_points_or_box(
                                state, frame_idx=fidx, obj_id=oid,
                                points=None, labels=None, box=b
                            )

        # -------- 传播 & 只保存白名单对象，并重映射到 0..K-1 --------
        keep_frames = set(int(i) for i in selected_indices)
        prefix = "et_" if mode == "et" else "rs_"
        # 连续化映射：比如 {3,7} -> {0,1}
        oid_remap = {old: new for new, old in enumerate(sorted(allowed_oids))}

        with torch.inference_mode(), amp_ctx:
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                fidx = int(frame_idx)
                if fidx not in keep_frames:
                    continue

                fdir = pathlib.Path(frame_root) / f"frame_{fidx}"
                fdir.mkdir(parents=True, exist_ok=True)

                union = None
                # 只导出允许的对象
                for i, oid in enumerate(obj_ids):
                    oid_int = int(oid)
                    if oid_int not in allowed_oids:
                        continue

                    new_oid = oid_remap[oid_int]  # 重映射到 0..K-1
                    mk = (masks[i].detach().cpu().numpy() > 0).astype(np.uint8).squeeze()

                    cv2.imwrite(str(fdir / f"{prefix}mask_obj{new_oid}.png"), mk * 255)
                    union = mk if union is None else (union | mk)

                if save_union and union is not None:
                    cv2.imwrite(str(fdir / f"{prefix}mask_union.png"), union * 255)


    
    def _sam2_prompt_via_subprocess(self, mirror_dir: str, key_indices: list[int], mode: str = "et", timeout_sec: int = 900):
        """
        用 subprocess 启一个新的 Python 解释器，
        在其中 import 本模块并调用顶层的 _sam2_prompt_child_main。
        """
        import subprocess, json, tempfile, sys, os, time

        workdir = tempfile.TemporaryDirectory(prefix="sam2_ui_")
        args_path = os.path.join(workdir.name, "ui_args.json")
        out_path  = os.path.join(workdir.name, "ui_out.json")

        with open(args_path, "w") as f:
            json.dump({
                "mirror_dir": mirror_dir,
                "keys": list(map(int, key_indices)),
                "mode": mode,
                "out": out_path
            }, f)

        module_name = __name__  # 本文件的模块名
        pycode = (
            "import os, json; "
            f"import {module_name} as M; "
            "args=json.load(open(os.environ['SAM2_ARGS'],'r')); "
            "M._sam2_prompt_child_main(args['mirror_dir'], args['keys'], args['mode'], 'SAM2 Prompt', args['out'])"
        )

        env = os.environ.copy()
        env["SAM2_ARGS"] = args_path

        print(f"[SAM2 v2][subprocess] launch child UI ({mode}) ...")
        proc = subprocess.Popen([sys.executable, "-c", pycode], env=env)

        start = time.time()
        rc = None
        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                if time.time() - start > timeout_sec:
                    print("[SAM2 v2][subprocess] UI timeout, terminating ...")
                    proc.terminate()
                    try:
                        proc.wait(2.0)
                    except Exception:
                        proc.kill()
                    break
                time.sleep(0.1)
        finally:
            prompts = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, "r") as f:
                        prompts = json.load(f)
                except Exception:
                    pass
            if mode == "et":
                self._sam2_prompts_et = prompts
            else:
                self._sam2_prompts_rs = prompts
            workdir.cleanup()
            print(f"[SAM2 v2][subprocess] collected prompts for {mode}: frames={len(prompts)}")
            return prompts



    # === NEW: （先 ET 标注/扩散，再 RS 标注/扩散，最后构建 gaze_contact）===
    def run_inline_sam2_and_label_v2(
        self,
        frame_ranges: List[tuple[int,int]],
        y_prompts_et: int = 20,
        y_prompts_rs: int = 10,
        rs_select_uses_same_keyset: bool = False,
        random_seed: int = 0,
    ):
        """
        - 先按 frame_ranges 为 ET 选关键帧 -> 人工交互 -> 在 et_images 上扩散；
        - 再为 RS 选关键帧（可复用 ET 集合，或重新选择）-> 人工交互 -> 在 rs_images 上扩散；
        - 之后遍历 frame_ranges 构建 gaze_contact.json（flip-y 与离线一致）。
        """
        # 组装这批所有帧（用于扩散 & 生成 label）
        print("[SAM2 v2] enter run_inline_sam2_and_label_v2")
        selected_all = []
        for s,e in frame_ranges:
            s,e = int(s), int(e)
            if e < s: s,e = e,s
            selected_all.extend(range(s, e+1))
        selected_all = sorted(set(selected_all))
        print(f"[SAM2 v2] total frames in batch: {len(selected_all)}, ranges={frame_ranges}")

        # ---------- Step 1: ET 标注 ----------
        print("[SAM2 v2] selecting ET keyframes ...")
        print("[SAM2 v2] calling _select_keyframes now", flush=True)
        try:
            key_et = self._select_keyframes(
                frame_ranges,
                y_total=y_prompts_et,
                min_per_ep=2, max_per_ep=5,
                include_ends=True,
                prefer_gaze=False,
                random_seed=random_seed,
            )
            print(f"[SAM2 v2] _select_keyframes returned {len(key_et)}", flush=True)
        except Exception as e:
            print("[SAM2 v2] _select_keyframes crashed:", repr(e), flush=True)
            key_et = []
        print(f"[SAM2 v2] ET keyframes = {len(key_et)} -> {key_et[:10]}{'...' if len(key_et)>10 else ''}")
        if key_et:
            print("[SAM2 v2] opening ET prompt (subprocess) ...")
            # self._sam2_annotate_keyframes_generic(self.et_mirror_dir, key_et, mode="et")
            self._sam2_prompt_via_subprocess(self.et_mirror_dir, key_et, mode="et", timeout_sec=900)
            print("[SAM2 v2] ET prompt finished")
        else:
            print("[SAM2 v2] No ET keyframes selected; will still try propagation (in case previous seeds exist).")

        # ET 传播
        print("[SAM2 v2] propagating ET masks ...")
        et_prompts = getattr(self, "_sam2_prompts_et", {})
        if selected_all:
            self._sam2_propagate_and_save_with_prompts(
                "et", self.et_mirror_dir, self.frame_root,
                selected_indices=selected_all,
                prompts_dict=et_prompts,
                save_union=True,
            )
        print("[SAM2 v2] ET propagation done.")

        # ---------- Step 2: RS 标注（独立） ----------
        print("[SAM2 v2] selecting RS keyframes ...")
        if rs_select_uses_same_keyset:
            key_rs = list(key_et)
        else:
            key_rs = self._select_keyframes(
                frame_ranges,
                y_total=y_prompts_rs,
                min_per_ep=1, max_per_ep=4,
                include_ends=True,
                prefer_gaze=False,   # RS 不依赖 gaze，均匀抽样即可
                random_seed=random_seed + 1,
            )

        if key_rs:
            print("[SAM2 v2] opening RS prompt (subprocess) ...")
            # self._sam2_annotate_keyframes_generic(self.rs_mirror_dir, key_rs, mode="rs")
            self._sam2_prompt_via_subprocess(self.rs_mirror_dir, key_rs, mode="rs", timeout_sec=900)
            print("[SAM2 v2] RS prompt window closed.")
        else:
            print("[SAM2 v2] No RS keyframes selected; will still try propagation (in case previous seeds exist).")
            
        print("[SAM2 v2] propagating RS masks ...")
        # RS 传播
        rs_prompts = getattr(self, "_sam2_prompts_rs", {})
        if selected_all:
            self._sam2_propagate_and_save_with_prompts(
                "rs", self.rs_mirror_dir, self.frame_root,
                selected_indices=selected_all,
                prompts_dict=rs_prompts,
                save_union=True,
            )

        # ---------- Step 3: 生成 gaze_contact.json ----------
        print("[SAM2 v2] building gaze_contact.json ...")
        self._build_gaze_contacts_subset(
            self.frame_root,
            selected_indices=selected_all,
            et_to_rs_map={0:0, 1:1},
            flip_y=True,
            eye_wh=(1280, 720),
        )
        print(f"[SAM2 v2] done. episodes={len(frame_ranges)} frames={len(selected_all)} "
              f"prompts_et={len(key_et)} prompts_rs={len(key_rs)}")

