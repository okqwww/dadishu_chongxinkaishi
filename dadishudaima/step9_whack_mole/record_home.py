#!/usr/bin/env python3
"""
Step 9-1: 记录初始位置
禁用力矩，用户移动机械臂到初始位置，按's'保存关节角度
"""

import sys
import numpy as np
import json

sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/lerobot/src")

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

OUTPUT_FILE = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step9_whack_mole/home_position.json"


def main():
    print("=" * 60)
    print("Step 9: 记录初始位置")
    print("=" * 60)

    # 连接机械臂
    port = "/dev/ttyACM0"
    print(f"\n连接机械臂 (port={port})...")
    robot_config = SOFollowerRobotConfig(port=port, id="my_awesome_follower_arm")
    robot = SOFollower(robot_config)
    robot.connect(calibrate=False)
    print("✅ 机械臂连接成功")

    try:
        # 禁用力矩
        print("\n禁用力矩，可以手动移动机械臂到初始位置")
        robot.bus.disable_torque()

        print("\n操作说明：")
        print("  移动机械臂到初始位置（点击完成后机械臂会回到这里）")
        print("  按 's' 保存当前关节角度")
        print("  按 'q' 退出（不保存）")
        print()

        while True:
            cmd = input("请输入命令 (s=保存, q=退出): ").strip().lower()

            if cmd == 'q':
                print("\n退出，不保存")
                break

            elif cmd == 's':
                # 读取当前关节角度
                obs = robot.get_observation()
                joint_pos = np.array([float(obs.get(f"{j}.pos", 0)) for j in JOINT_NAMES])

                # 启动力矩保持位置
                robot.bus.enable_torque()

                # 保存
                data = {
                    "joint_names": JOINT_NAMES,
                    "joint_positions": joint_pos.tolist(),
                }
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"\n✅ 初始位置已保存: {OUTPUT_FILE}")
                print(f"  关节角度: {joint_pos.round(2).tolist()}")
                break

            else:
                print("未知命令")

    finally:
        robot.disconnect()
        print("\n已断开连接")


if __name__ == "__main__":
    main()
