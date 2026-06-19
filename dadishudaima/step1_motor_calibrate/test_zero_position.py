#!/usr/bin/env python3
"""
Step 1 测试脚本: 发送全部零位，验证机械臂能否到达零位

使用方法:
1. 确保机械臂已连接
2. 运行脚本，观察机械臂是否移动到零位
3. 观察完成后手动检查机械臂姿态是否确实是零位
"""

import time
from lerobot.robots.so_follower import SOFollower

def main():
    # 创建机器人配置 - 根据实际情况修改port
    config = SOFollower.config_class(
        id="my_awesome_follower_arm",
        port="/dev/ttyACM0",  # 根据实际串口修改
    )

    # 创建机器人实例
    robot = SOFollower(config)

    print("=" * 50)
    print("Step 1 测试: 发送全部零位")
    print("=" * 50)

    try:
        # 连接机器人（不校准，因为我们只是测试）
        print("\n连接机器人...")
        robot.connect(calibrate=False)
        print("✅ 机器人已连接")

        # 读取当前关节角度
        print("\n读取当前关节角度...")
        obs = robot.get_observation()
        current_angles = {
            "shoulder_pan": obs.get("shoulder_pan.pos", "N/A"),
            "shoulder_lift": obs.get("shoulder_lift.pos", "N/A"),
            "elbow_flex": obs.get("elbow_flex.pos", "N/A"),
            "wrist_flex": obs.get("wrist_flex.pos", "N/A"),
            "wrist_roll": obs.get("wrist_roll.pos", "N/A"),
        }
        print(f"当前角度: {current_angles}")

        # 发送全部零位
        print("\n发送全部零位命令...")
        action = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
        }
        robot.send_action(action)
        print("✅ 零位命令已发送")

        # 等待机械臂移动
        print("\n等待3秒让机械臂移动到目标位置...")
        time.sleep(3)

        # 再次读取关节角度
        print("\n读取移动后的关节角度...")
        obs = robot.get_observation()
        after_angles = {
            "shoulder_pan": obs.get("shoulder_pan.pos", "N/A"),
            "shoulder_lift": obs.get("shoulder_lift.pos", "N/A"),
            "elbow_flex": obs.get("elbow_flex.pos", "N/A"),
            "wrist_flex": obs.get("wrist_flex.pos", "N/A"),
            "wrist_roll": obs.get("wrist_roll.pos", "N/A"),
        }
        print(f"移动后角度: {after_angles}")

        # 计算与零位的误差
        print("\n与零位的误差检查:")
        for joint, angle in after_angles.items():
            if isinstance(angle, (int, float)):
                error = abs(angle)
                status = "✅" if error < 1.0 else "⚠️"
                print(f"  {joint}: {error:.2f}° {status}")
            else:
                print(f"  {joint}: 无法计算（N/A）")

        print("\n" + "=" * 50)
        print("测试完成")
        print("请手动检查机械臂是否到达零位姿态")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 断开连接
        if robot.is_connected:
            robot.disconnect()
            print("\n机器人已断开连接")

if __name__ == "__main__":
    main()
