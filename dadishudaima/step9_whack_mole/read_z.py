#!/usr/bin/env python3
"""
读取当前机械臂末端的Z值
禁用力矩后手动移动到目标位置，按's'读取FK计算的末端Z值
"""

import sys
import numpy as np

sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/lerobot/src")

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics

URDF_PATH = "/home/zyj/dadishu_chongxinkaishi/SO-ARM100/Simulation/SO101/so101_5dof_stylus_2.urdf"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def main():
    print("=" * 60)
    print("读取机械臂末端Z值")
    print("=" * 60)

    # 连接机械臂
    port = "/dev/ttyACM0"
    print(f"\n连接机械臂 (port={port})...")
    robot_config = SOFollowerRobotConfig(port=port, id="my_awesome_follower_arm")
    robot = SOFollower(robot_config)
    robot.connect(calibrate=False)
    print("✅ 机械臂连接成功")

    # 初始化FK求解器
    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="stylus_tcp_link",
        joint_names=JOINT_NAMES
    )

    try:
        # 禁用力矩
        print("\n禁用力矩，可以手动移动机械臂")
        robot.bus.disable_torque()

        print("\n操作说明：")
        print("  手动移动机械臂到目标位置（触控笔悬在屏幕上方的点击位置）")
        print("  按 's' 读取当前末端位置")
        print("  按 'q' 退出")
        print()

        while True:
            cmd = input("请输入命令 (s=读取, q=退出): ").strip().lower()

            if cmd == 'q':
                break

            elif cmd == 's':
                # 读取关节角度
                obs = robot.get_observation()
                joint_pos = np.array([float(obs.get(f"{j}.pos", 0)) for j in JOINT_NAMES])

                # FK计算末端位置
                T = kin.forward_kinematics(joint_pos)
                pos = T[:3, 3]  # XYZ位置

                print(f"\n当前关节角度: {joint_pos.round(3).tolist()}")
                print(f"FK末端位置:")
                print(f"  X = {pos[0]*1000:.2f} mm")
                print(f"  Y = {pos[1]*1000:.2f} mm")
                print(f"  Z = {pos[2]*1000:.2f} mm")
                print()
                print("将这个Z值填入whack_mole.py中的 TARGET_Z_MM")

    finally:
        robot.disconnect()
        print("\n已断开连接")


if __name__ == "__main__":
    main()
