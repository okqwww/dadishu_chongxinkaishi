#!/usr/bin/env python3
"""
Step 5-1: 手眼标定数据采集
棋盘格贴在机械臂末端，移动机械臂到不同位姿，采集数据
"""

import sys
import numpy as np
import cv2
import json
import time
import os
from pathlib import Path
from lerobot.robots.so_follower import SOFollower
# 添加lerobot路径
sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/lerobot/src")

# from lerobot.robots.so_follower.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.model.kinematics import RobotKinematics

# 路径配置
INTRINSIC_FILE = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step4_camera_intrinsic_calibration/intrinsic_params.yaml"
URDF_PATH = "/home/zyj/dadishu_chongxinkaishi/SO-ARM100/Simulation/SO101/so101_5dof_stylus_2.urdf"
OUTPUT_DIR = Path(__file__).parent
CALIB_IMAGE_DIR = OUTPUT_DIR / "calibration_images"
CALIB_DATA_FILE = OUTPUT_DIR / "hand_eye_data.json"

# 棋盘格参数
CHESSBOARD_SIZE = (11, 8)  # 内角点数
SQUARE_SIZE = 0.005  # 方格尺寸（米）

# 关节名称
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# FK_OFFSET：补偿 FK 模型的系统性平移误差
# FK_OFFSET = np.array([0.0305, -0.0118, 0.0])
FK_OFFSET = np.array([0.0, 0.0, 0.0])


def load_intrinsics(intrinsic_file):
    """加载相机内参"""
    fs = cv2.FileStorage(intrinsic_file, cv2.FileStorage_READ)
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    img_width = int(fs.getNode("img_width").real())
    img_height = int(fs.getNode("img_height").real())
    fs.release()
    return K, dist, img_width, img_height


def detect_checkerboard_pnp(color_image, K, dist, chessboard_size, square_size):
    """PnP检测棋盘格位姿

    Returns:
        success: bool
        T_cam_to_board: 4x4 齐次变换矩阵（相机→棋盘格）
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

    # 检测角点
    ret, corners = cv2.findChessboardCorners(
        gray, chessboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        return False, None

    # 亚像素精化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 棋盘格3D坐标
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # PnP求解
    ret, rvec, tvec = cv2.solvePnP(
        objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ret:
        return False, None

    # 构造 T_cam_to_board
    R, _ = cv2.Rodrigues(rvec)
    T_cam_to_board = np.eye(4)
    T_cam_to_board[:3, :3] = R
    T_cam_to_board[:3, 3] = tvec.flatten()

    return True, T_cam_to_board


def main():
    import argparse
    parser = argparse.ArgumentParser(description="手眼标定数据采集")
    parser.add_argument("--camera", "-c", type=int, default=0, help="相机设备号")
    parser.add_argument("--port", "-p", type=str, default="/dev/ttyACM0", help="机械臂串口")
    parser.add_argument("--num", "-n", type=int, default=15, help="采集位姿数量（默认15）")
    args = parser.parse_args()

    # 创建目录
    CALIB_IMAGE_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("手眼标定 - 数据采集")
    print("=" * 60)

    # 加载内参
    print(f"\n加载内参: {INTRINSIC_FILE}")
    K, dist, img_width, img_height = load_intrinsics(INTRINSIC_FILE)
    print(f"  内参矩阵:\n{K}")
    print(f"  图像分辨率: {img_width} x {img_height}")

    # 初始化运动学
    print(f"\n加载 URDF: {URDF_PATH}")
    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="stylus_tcp_link",
        joint_names=JOINT_NAMES
    )
    print("✅ URDF 加载成功")

    # 连接机械臂
    print(f"\n连接机械臂 (port={args.port})...")
    print("\n连接机器人 ...")
    config = SOFollower.config_class(
        id="my_awesome_follower_arm",
        port="/dev/ttyACM0",
    )
    robot = SOFollower(config)
    robot.connect()
    print("✅ 机械臂连接成功")

    # 连接相机
    print(f"\n连接相机 (index={args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
    if not cap.isOpened():
        print(f"❌ 无法打开相机 {args.camera}")
        return
    # 消耗几帧让自动曝光生效
    for _ in range(10):
        cap.read()
    print("✅ 相机连接成功")

    print("\n" + "=" * 60)
    print("操作说明：")
    print("  1. 确认棋盘格已贴在机械臂末端，且在相机视野内")
    print("  2. 按 'd' 禁用机械臂力矩（可手动移动）")
    print("  3. 手动移动机械臂到新位姿")
    print("  4. 按 'e' 启动力矩并进入采集模式")
    print("  5. 程序将自动采集数据并保存")
    print("  6. 重复步骤2-5，采集至少10-15个不同位姿")
    print("  7. 按 'q' 退出并求解标定")
    print("=" * 60)

    calibration_data = []

    try:
        while True:
            print("\n请输入命令 (d=禁用力矩, e=启动力矩并采集, q=退出): ", end="", flush=True)
            cmd = input().strip().lower()

            if cmd == 'q':
                break

            elif cmd == 'd':
                # 禁用力矩
                robot.bus.disable_torque()
                print("  → 力矩已禁用，可以手动移动机械臂")

            elif cmd == 'e':
                # 启动力矩
                robot.bus.enable_torque()
                time.sleep(0.5)  # 等待稳定

                # 读取关节角度，FK计算末端位姿（加FK_OFFSET补偿）
                obs = robot.get_observation()
                joint_pos = np.array([float(obs.get(f"{j}.pos", 0)) for j in JOINT_NAMES])
                T_base_to_end_raw = kin.forward_kinematics(joint_pos)
                T_base_to_end = T_base_to_end_raw.copy()
                T_base_to_end[:3, 3] += FK_OFFSET  # 补偿FK系统偏差

                print(f"  末端位置（补偿后）: {T_base_to_end[:3, 3].round(4)}")

                # 清空相机缓冲区，读取最新帧
                for _ in range(5):
                    cap.read()
                time.sleep(0.05)

                ret, frame = cap.read()
                if not ret:
                    print("  ❌ 相机读取失败")
                    continue

                # PnP检测
                success, T_cam_to_board = detect_checkerboard_pnp(
                    frame, K, dist, CHESSBOARD_SIZE, SQUARE_SIZE
                )

                if not success:
                    print("  ❌ 未检测到棋盘格，请调整位姿后重试")
                    continue

                print(f"  棋盘格位置（相机坐标系下）: {T_cam_to_board[:3, 3].round(4)}")

                # 保存数据
                pose_id = len(calibration_data)
                data = {
                    "pose_id": pose_id,
                    "T_base_to_end": T_base_to_end.tolist(),
                    "T_cam_to_board": T_cam_to_board.tolist(),
                }
                calibration_data.append(data)

                # 保存原图和带标注的图
                raw_path = CALIB_IMAGE_DIR / f"pose_{pose_id:02d}_raw.jpg"
                ann_path = CALIB_IMAGE_DIR / f"pose_{pose_id:02d}_ann.jpg"
                cv2.imwrite(str(raw_path), frame)

                # 标注
                display = frame.copy()
                cv2.drawFrameAxes(display, K, dist,
                                   cv2.Rodrigues(T_cam_to_board[:3, :3])[0],
                                   T_cam_to_board[:3, 3], SQUARE_SIZE * 3)
                cv2.imwrite(str(ann_path), display)

                print(f"  ✅ 已采集第 {pose_id + 1} 个位姿")
                print(f"     原图: {raw_path}")
                print(f"     标注图: {ann_path}")

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        robot.disconnect()
        cap.release()
        print("\n已断开连接")

    # 保存标定数据
    if calibration_data:
        with open(CALIB_DATA_FILE, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"\n✅ 标定数据已保存: {CALIB_DATA_FILE}")
        print(f"   共 {len(calibration_data)} 个位姿")
        print(f"\n请检查 {CALIB_IMAGE_DIR}/ 目录中的标注图，删除检测不准确的图片")
        print(f"确认后运行 step5_solve.py 求解标定参数")
    else:
        print("\n⚠️  未采集到有效数据")


if __name__ == "__main__":
    main()