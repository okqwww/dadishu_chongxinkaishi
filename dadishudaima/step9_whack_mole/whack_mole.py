#!/usr/bin/env python3
"""
Step 9-2: 完整打地鼠链路
相机检测地鼠 → 坐标转换 → IK求解 → 机械臂点击
"""

import sys
import numpy as np
import json
import time

import sys
import numpy as np
import json
import time
import queue
import threading
sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/lerobot/src")
sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step7_pixel_to_base")
sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step8_mole_detection")

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics
from pixel_to_base import PixelToBaseConverter, load_params as load_coord_params
from mole_detector import MoleDetector

# Apriltag库兼容
try:
    import pupil_apriltags as apriltag_lib
    _APRILTAG_LIB = "pupil_apriltags"
except ImportError:
    import apriltag as apriltag_lib
    _APRILTAG_LIB = "apriltag"

# 路径配置
URDF_PATH = "/home/zyj/dadishu_chongxinkaishi/SO-ARM100/Simulation/SO101/so101_5dof_stylus_2.urdf"
ROBOT_PORT = "/dev/ttyACM0"
HOME_FILE = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step9_whack_mole/home_position.json"
IK_INIT_FILE = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step9_whack_mole/ik_init_joints.json"
MOLE_CONFIG = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step8_mole_detection/mole_calibration.json"

# Apriltag参数
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.040
HALF_TAG = TAG_SIZE_M / 2.0
TAG_POSITIONS_MM = {
    0: np.array([0.0, 0.0, 0.0]),
    1: np.array([130.0, 0.0, 0.0]),
    3: np.array([0.0, 130.0, 0.0]),
    2: np.array([130.0, 130.0, 0.0]),
}

# FK_OFFSET（与Step3相同）
FK_OFFSET = np.array([0.0305, -0.0118, 0.0])
# FK_OFFSET = np.array([0.0, 0.0, 0.0])

# 固定旋转（末端竖直向下）
FIXED_ROTATION = np.eye(3)

# 目标Z值（毫米）- 用read_z.py脚本读取后填入
# 如果填了非0值，则使用此值覆盖计算出的Zbase
TARGET_Z_MM = 18.85

TASK_QUEUE = queue.Queue(maxsize=1)
EXIT_FLAG = threading.Event()
COOL_DOWN_SEC = 0.0


class WhackMole:
    def __init__(self, camera_idx=1):
        # 加载初始位置（实际初始位置，点击后回到这里）
        with open(HOME_FILE, 'r') as f:
            self.home_data = json.load(f)
        self.home_joints = np.array(self.home_data["joint_positions"])
        print(f"✅ 初始位置已加载: {self.home_joints.round(2).tolist()}")

        # 加载IK初始角度（用于IK求解）
        with open(IK_INIT_FILE, 'r') as f:
            self.ik_init_data = json.load(f)
        self.ik_init_joints = np.array(self.ik_init_data["joint_positions"])
        print(f"✅ IK初始角度已加载: {self.ik_init_joints.round(2).tolist()}")

        # 连接机械臂
        print(f"\n连接机械臂 (port={ROBOT_PORT})...")
        robot_config = SOFollowerRobotConfig(port=ROBOT_PORT, id="my_awesome_follower_arm")
        self.robot = SOFollower(robot_config)
        # print("\n连接机器人 ...")
        # config = SOFollower.config_class(
        #     id="my_awesome_follower_arm",
        #     port="/dev/ttyACM0",
        # )
        # robot = SOFollower(config)
        self.robot.connect(calibrate=False)
        print("✅ 机械臂连接成功")

        # 初始化IK求解器
        print(f"\n初始化IK求解器...")
        self.kin = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name="stylus_tcp_link",
            joint_names=self.home_data["joint_names"]
        )
        print("✅ IK求解器初始化成功")

        # 加载坐标转换参数
        K, T_cam_to_base = load_coord_params()
        self.converter = PixelToBaseConverter(K, T_cam_to_base)
        self.K = K
        print("✅ 坐标转换模块加载成功")

        # 加载地鼠检测器
        self.mole_detector = MoleDetector.from_config(MOLE_CONFIG)
        print("✅ 地鼠检测器加载成功")

        # Apriltag检测器
        if _APRILTAG_LIB == "pupil_apriltags":
            self.apriltag_detector = apriltag_lib.Detector(
                families=TAG_FAMILY,
                nthreads=2,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
            )
        else:
            options = apriltag_lib.DetectorOptions(families=TAG_FAMILY)
            self.apriltag_detector = apriltag_lib.Detector(options)

        # 连接相机
        self.cap = None
        self.T_tray_to_cam = None
        self._init_camera(camera_idx)

    def _init_camera(self, camera_idx):
        print(f"\n连接相机 (index={camera_idx})...")
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(10):
            self.cap.read()
        print("✅ 相机连接成功")

    def detect_apriltag(self, frame):
        """检测Apriltag，返回T_tray_to_cam"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if _APRILTAG_LIB == "pupil_apriltags":
            detections = self.apriltag_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]),
                tag_size=TAG_SIZE_M,
            )
        else:
            detections = self.apriltag_detector.detect(gray)

        if len(detections) == 0:
            return None, []

        obj_points = []
        img_points = []

        for det in detections:
            if det.tag_id not in TAG_POSITIONS_MM:
                continue
            tag_center_m = TAG_POSITIONS_MM[det.tag_id] / 1000.0
            corners_in_tag = np.array([
                [-HALF_TAG,  HALF_TAG, 0.0],
                [ HALF_TAG,  HALF_TAG, 0.0],
                [ HALF_TAG, -HALF_TAG, 0.0],
                [-HALF_TAG, -HALF_TAG, 0.0],
            ], dtype=np.float64)
            corners_in_tray = corners_in_tag + tag_center_m
            obj_points.extend(corners_in_tray)
            img_points.extend(det.corners.astype(np.float32))

        if len(obj_points) < 4:
            return None, []

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        try:
            _, rvecs, tvecs = cv2.solvePnPGeneric(
                obj_points, img_points, self.K, np.zeros(5),
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            min_error = float('inf')
            best_idx = 0
            for i in range(len(rvecs)):
                proj, _ = cv2.projectPoints(obj_points, rvecs[i], tvecs[i], self.K, np.zeros(5))
                error = np.linalg.norm(img_points - proj.reshape(-1, 2), axis=1).mean()
                if error < min_error:
                    min_error = error
                    best_idx = i
            rvec = rvecs[best_idx]
            tvec = tvecs[best_idx]
        except Exception:
            _, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, self.K, np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        R, _ = cv2.Rodrigues(rvec)
        T_tray_to_cam = np.eye(4)
        T_tray_to_cam[:3, :3] = R
        T_tray_to_cam[:3, 3] = tvec.flatten()

        used_tags = [det.tag_id for det in detections if det.tag_id in TAG_POSITIONS_MM]
        return T_tray_to_cam, used_tags

    def move_to_home(self):
        """移动到初始位置"""
        self.robot.send_action({
            f"{j}.pos": float(a)
            for j, a in zip(self.home_data["joint_names"], self.home_joints)
        })
        time.sleep(1.5)

    def whack_one(self, u, v):
        """点击一个地鼠"""
        # 坐标转换
        Xbase, Ybase, Zbase = self.converter.convert(u, v, self.T_tray_to_cam)
        if Xbase is None:
            print(f"  坐标转换失败")
            return False

        # 如果TARGET_Z_MM>0，则用此值覆盖Zbase
        if TARGET_Z_MM > 0:
            Zbase = TARGET_Z_MM / 1000.0
            Zbase_stay = Zbase+0.03
            print(f"  Z值已覆盖: {Zbase*1000:.1f}mm (原计算值被忽略)")
        else:
            print(f"  补偿之前目标位置: ({Xbase*1000:.1f}, {Ybase*1000:.1f}, {Zbase*1000:.1f})mm")

        # FK_OFFSET补偿
        target = np.array([Xbase, Ybase, Zbase]) - FK_OFFSET
        target_stay = np.array([Xbase, Ybase, Zbase_stay]) - FK_OFFSET
        print(f"  补偿之后目标位置: ({target[0]*1000:.1f}, {target[1]*1000:.1f}, {target[2]*1000:.1f})mm")
        # print(f"  目标位置: ({target[0]*1000:.1f}, {target[1]*1000:.1f}, {target[2]*1000:.1f})mm")

        # IK求解（使用IK初始角度）
        joint_pos = self.ik_init_joints.copy()
        desired_pose = np.eye(4)
        desired_pose_stay = np.eye(4)
        desired_pose[:3, :3] = FIXED_ROTATION
        desired_pose[:3, 3] = target
        desired_pose_stay[:3, :3] = FIXED_ROTATION
        desired_pose_stay[:3, 3] = target_stay

        result_joints = self.kin.inverse_kinematics(joint_pos, desired_pose)
        result_joints_stay = self.kin.inverse_kinematics(joint_pos, desired_pose_stay)
        if result_joints is None:
            print(f"  IK求解失败")
            return False

        print(f"  IK求解成功: {result_joints.round(2).tolist()}")

        # 移动到目标
        self.robot.send_action({
            f"{j}.pos": float(a)
            for j, a in zip(self.home_data["joint_names"], result_joints)
        })
        print("等待1s")
        time.sleep(0.5)  # 等待点击稳定
        self.robot.send_action({
            f"{j}.pos": float(a)
            for j, a in zip(self.home_data["joint_names"], result_joints_stay)
        })
        print("等待1s")
        time.sleep(0.5)  # 等待点击稳定
        # 回到初始位置
        self.move_to_home()

        return True
        
    def _arm_worker(self):
            """机械臂工作子线程：执行击打+冷却，不占用视觉主线程"""
            print("🤖 机械臂工作线程已启动")
            while not EXIT_FLAG.is_set():
                try:
                    # 等待待击打坐标，超时0.2s防止卡死
                    u, v = TASK_QUEUE.get(timeout=0.2)
                    print(f"\n[子线程] 执行击打，像素=({u},{v})")
                    self.whack_one(u, v)
                    print(f"[子线程] 击打完成，进入{COOL_DOWN_SEC}秒冷却...")
                    # 冷却延时放在子线程，主线程正常跑视觉
                    time.sleep(COOL_DOWN_SEC)
                    print(f"[子线程] 冷却结束，等待新目标")
                except queue.Empty:
                    continue
            print("🤖 机械臂工作线程退出")


    def run(self):
        print("=" * 60)
        print("Step 9: 完整打地鼠链路")
        print("=" * 60)
        print("\n操作说明：")
        print("  检测到地鼠后自动执行点击")
        print("  按 'q' 或 ESC 退出")
        print("  按 'h' 回到初始位置")
        print()

        cv2.namedWindow("Whack Mole", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Whack Mole", 960, 540)
        # 启动机械臂子线程
        worker_thread = threading.Thread(target=self._arm_worker, daemon=True)
        worker_thread.start()
        try:
            while True:
                print("进入循环")
                ret, frame = self.cap.read()
                if not ret:
                    continue
                # print(f"Frame id: {id(frame)}, shape: {frame.shape}")
                # print(frame)
                # Apriltag检测
                T_tray_to_cam, tags = self.detect_apriltag(frame)
                self.T_tray_to_cam = T_tray_to_cam

                # 地鼠检测
                mole_centers = self.mole_detector.detect(frame)
                # ========== 核心队列逻辑：每帧必清空旧任务 ==========
                # 1. 强制清空队列所有过期坐标
                while not TASK_QUEUE.empty():
                    TASK_QUEUE.get()
                # 2. 当前帧有效才放入最新地鼠
                if mole_centers and T_tray_to_cam is not None:
                    uv = mole_centers[0]
                    TASK_QUEUE.put(uv)
                # ==================================================
                # 绘制
                vis = frame.copy()

                # 画tag边框
                if T_tray_to_cam is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if _APRILTAG_LIB == "pupil_apriltags":
                        detections = self.apriltag_detector.detect(
                            gray,
                            estimate_tag_pose=True,
                            camera_params=(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]),
                            tag_size=TAG_SIZE_M,
                        )
                    else:
                        detections = self.apriltag_detector.detect(gray)

                    for det in detections:
                        corners = det.corners.astype(int)
                        cv2.polylines(vis, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

                    rvec, _ = cv2.Rodrigues(T_tray_to_cam[:3, :3])
                    cv2.drawFrameAxes(vis, self.K, np.zeros(5), rvec, T_tray_to_cam[:3, 3], 0.05)

                # 画地鼠
                for i, (u, v) in enumerate(mole_centers):
                    cv2.circle(vis, (u, v), 10, (0, 0, 255), -1)
                    cv2.putText(vis, f"#{i+1}", (u+12, v-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 状态栏
                if T_tray_to_cam is not None:
                    status = f"Tags: {tags} | Apriltag OK | 检测到 {len(mole_centers)} 个地鼠"
                else:
                    status = "未检测到有效Tag"
                cv2.putText(vis, status, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1)
                cv2.putText(vis, "[q] quit [h] home", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                cv2.imshow("Whack Mole", vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('h'):
                    self.move_to_home()
                    print("\n已回到初始位置")

                # 检测到地鼠时自动点击
                # if mole_centers and T_tray_to_cam is not None:
                #     u, v = mole_centers[0]  # 取第一个
                #     print(f"\n检测到地鼠: 像素=({u},{v})")
                #     self.whack_one(u, v)
                #     # 等待一下避免重复点击
                #     print("等待5s")
                #     time.sleep(5.0)
                #     print("5s等待完成")

        finally:
            EXIT_FLAG.set()
            self.cap.release()
            cv2.destroyAllWindows()
            self.robot.disconnect()
            print("\n已断开连接")


if __name__ == "__main__":
    import argparse
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=int, default=1)
    args = parser.parse_args()

    app = WhackMole(camera_idx=args.camera)
    app.run()
