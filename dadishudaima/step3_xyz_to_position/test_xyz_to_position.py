#!/usr/bin/env python3
"""
Step 3 测试脚本: 输入 xyz 坐标，机械臂移动到目标位置 (含 FK 补偿，姿态固定)

流程:
1. 用户将机械臂移动到参考位置 A，程序记录此时的角度
2. 循环:
   - 用户输入目标 xyz
   - 程序用固定姿态 (R=I) + (目标xyz - FK偏移) 组合目标位姿
   - 用位置A的角度作为IK初值，求解目标关节角度
   - 用 IK 结果做 FK，对比末端位置与目标位置（补偿后）
   - 机械臂移动到目标位置
   - 等待用户确认
   - 机械臂返回位置A
   - 继续下一轮
"""
import time
import numpy as np
from lerobot.model import RobotKinematics
from lerobot.robots.so_follower import SOFollower

# URDF路径
URDF_PATH = "/home/zyj/dadishu_chongxinkaishi/SO-ARM100/Simulation/SO101/so101_5dof_stylus_2.urdf"
TARGET_FRAME = "stylus_tcp_link"
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# FK 模型系统性偏移补偿（单位：米）
# 实测平均误差：Δx = +0.0305 m, Δy = -0.0118 m
FK_OFFSET = np.array([0.0305, -0.0118, 0.0])
# FK_OFFSET = np.array([0, 0, 0.0])
# 固定姿态：旋转矩阵设为单位矩阵（末端坐标系与基座坐标系平行）
FIXED_ROTATION = np.eye(3)


def get_joint_angles(robot):
    """读取当前关节角度（度）"""
    obs = robot.get_observation()
    return np.array([float(obs.get(f"{j}.pos", 0)) for j in JOINT_NAMES])


def send_joint_positions(robot, joints):
    """发送关节角度目标到机械臂"""
    action = {f"{j}.pos": float(angle) for j, angle in zip(JOINT_NAMES, joints)}
    robot.send_action(action)


def main():
    print("=" * 60)
    print("Step 3: xyz 坐标 → 机械臂位置 (含 FK 补偿，姿态固定)")
    print("=" * 60)
    print(f"URDF: {URDF_PATH}")
    print(f"末端 frame: {TARGET_FRAME}")
    print(f"固定旋转矩阵:\n{FIXED_ROTATION}")

    # 加载运动学
    print("\n加载 URDF ...")
    try:
        kin = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name=TARGET_FRAME,
            joint_names=JOINT_NAMES
        )
        print("✅ URDF 加载成功")
    except Exception as e:
        print(f"❌ URDF 加载失败: {e}")
        return

    # 连接机器人
    print("\n连接机器人 ...")
    config = SOFollower.config_class(
        id="my_awesome_follower_arm",
        port="/dev/ttyACM0",
    )
    robot = SOFollower(config)

    try:
        robot.connect(calibrate=False)
        print("✅ 机器人已连接")

        # ============================================================
        # 第一步：记录参考位置 A
        # ============================================================
        print("\n" + "=" * 60)
        print("设置参考位置 A")
        print("=" * 60)
        print("\n本测试将自动控制电机力矩，允许你手动拖拽机械臂。")
        print("请务必在听到提示后再移动，注意安全。")

        print("\n>>> 即将卸力，请准备移动机械臂到【参考位置A】...")
        robot.bus.disable_torque()
        input("现在可以自由移动机械臂到【参考位置A】，移动完成后按 Enter 继续...")
        joints_a = get_joint_angles(robot)
        print(f"\n参考位置A的关节角度alpha1: {joints_a}")
        robot.bus.enable_torque()
        print("已恢复力矩。")

        # ============================================================
        # 主循环：输入 xyz → IK → 验证 → 移动 → 返回 A
        # ============================================================
        print("\n" + "=" * 60)
        print("主循环开始")
        print("=" * 60)

        while True:
            # 用户输入目标位置
            print("\n请输入目标 xyz 坐标 (单位: 米)")
            print("输入 'q' 退出程序")

            try:
                x_input = input("x = ")
                if x_input.lower() == 'q':
                    print("退出程序")
                    break
                x = float(x_input)
                y = float(input("y = "))
                z = float(input("z = "))
            except ValueError:
                print("输入无效，请重新输入")
                continue

            target_xyz = np.array([x, y, z])
            print(f"目标位置 (物理坐标): {target_xyz}")

            # 为补偿 FK 模型误差，计算送给 IK 的修正目标位置
            target_xyz_corrected = target_xyz - FK_OFFSET
            print(f"修正后的 IK 目标位置: {target_xyz_corrected}")

            # 组合目标位姿：旋转矩阵固定，平移向量用修正后的位置
            desired_pose = np.eye(4)
            desired_pose[:3, :3] = FIXED_ROTATION
            desired_pose[:3, 3] = target_xyz_corrected
            print("\n目标位姿 (给 IK):")
            print(desired_pose)
            # print("IK之前先用FK预热")
            # _ = kin.forward_kinematics(joints_a)
            # print("预热完成")
            # 调用 IK
            print("\n调用 IK 求解 ...")
            try:
                ik_result = kin.inverse_kinematics(
                    current_joint_pos=joints_a,
                    desired_ee_pose=desired_pose,
                    position_weight=1.0,
                    orientation_weight=0.01,
                )
                print(f"IK 求解结果alpha2 (deg): {ik_result}")
            except Exception as e:
                print(f"❌ IK 求解失败: {e}")
                continue

            # ---------- IK 验证（考虑补偿）----------
            print("\n>>> IK 验证：用 IK 结果做 FK，对比目标位置 ...")
            fk_pose_raw = kin.forward_kinematics(ik_result)
            fk_pos_raw = fk_pose_raw[:3, 3]
            # 加上补偿后才是预估的物理位置
            fk_pos_physical = fk_pos_raw + FK_OFFSET
            pos_error_mm = np.linalg.norm(fk_pos_physical - target_xyz) * 1000.0
            print(f"FK 计算原始位置:    {fk_pos_raw}")
            print(f"加补偿后的物理位置: {fk_pos_physical}")
            print(f"与原始目标的距离误差: {pos_error_mm:.2f} mm")
            if pos_error_mm < 2.0:
                print("✅ 位置误差很小，IK 收敛良好")
            elif pos_error_mm < 10.0:
                print("⚠️ 位置误差较大，IK 可能未精确收敛")
            else:
                print("❌ 位置误差过大，IK 求解异常！")
            # -----------------------------------------

            # 移动到目标位置
            print("\n移动到目标位置 ...")
            send_joint_positions(robot, ik_result)
            print("✅ 已发送目标角度，等待 3 秒让机械臂稳定...")
            import time
            time.sleep(3) # 等待 3 秒让机械臂稳定
            joints_b = get_joint_angles(robot)
            print(f"实际执行舵机角度alpha3: {joints_b}")
            # 等待用户确认
            print("\n" + "-" * 40)
            print("机械臂已移动到目标位置")
            print("输入 'q' 退出，输入任意其他内容返回位置A并继续下一轮:")
            user_input = input()

            if user_input.lower() == 'q':
                print("退出程序")
                break

            # 返回参考位置 A
            print("\n返回参考位置 A ...")
            send_joint_positions(robot, joints_a)
            print("✅ 已返回位置A")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if robot.is_connected:
            robot.disconnect()
            print("\n机器人已断开连接")

    print("\n" + "=" * 60)
    print("程序结束")
    print("=" * 60)


if __name__ == "__main__":
    main()