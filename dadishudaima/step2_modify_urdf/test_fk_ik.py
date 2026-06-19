#!/usr/bin/env python3
"""
Step 2 测试脚本: 验证 URDF 和 RobotKinematics 的 FK/IK 准确性

FK测试:
  1. 手动移动机械臂到某位置，读取关节角
  2. 手动测量末端在基坐标系中的坐标 (x, y, z)
  3. FK 解算末端位置，对比手动测量值 (x 坐标已补偿 +0.04m)

IK测试:
  1. 程序自动卸力，用户移动机械臂到位置 A，读取关节角 a
  2. 程序自动卸力，用户移动机械臂到位置 B，读取关节角 b
  3. 用 b 做 FK 并补偿得到目标位姿 Y
  4. 以 a 为初值，Y 为目标调用 IK，得到关节角 c
  5. 对 c 做 FK 并补偿，与 Y 对比末端位置误差
"""

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

# FK x方向补偿量（米），根据实测 FK 值比手动测量值小约 0.04m
FK_X_COMPENSATION = 0.04


def get_joint_angles(robot):
    """读取当前关节角度（度）"""
    obs = robot.get_observation()
    return np.array([float(obs.get(f"{j}.pos", 0)) for j in JOINT_NAMES])


def apply_fk_compensation(pose):
    """对 FK 结果的 x 坐标进行补偿，返回补偿后的 4x4 位姿矩阵（不修改原矩阵）"""
    compensated = pose.copy()
    compensated[0, 3] += FK_X_COMPENSATION
    return compensated


def test_forward_kinematics(kin, robot):
    """FK测试：手动测量 vs FK解算（x已补偿）"""
    print("\n" + "=" * 60)
    print("FK 测试 - 手动测量 vs FK解算 (x补偿 +{:.3f}m)".format(FK_X_COMPENSATION))
    print("=" * 60)
    print("\n请手动移动机械臂到任意位置，然后")
    print("用刻度垫子测量末端在基坐标系中的坐标 (单位: 米)")

    try:
        joints = get_joint_angles(robot)
        print(f"\n当前关节角度(度): {joints}")

        # 输入手动测量值
        try:
            x = float(input("x = "))
            y = float(input("y = "))
            z = float(input("z = "))
        except ValueError:
            print("输入无效，使用示例值 (0.2, 0.1, 0.15)")
            x, y, z = 0.2, 0.1, 0.15

        measured = np.array([x, y, z])
        print(f"手动测量坐标: {measured}")

        # FK解算
        pose_raw = kin.forward_kinematics(joints)
        pose = apply_fk_compensation(pose_raw)
        fk_pos = pose[:3, 3]
        print(f"\nFK 解算坐标 (补偿后): {fk_pos}")
        print("\n完整位姿矩阵 T (4x4，已补偿):")
        print(pose)
        print("\n旋转矩阵 R (3x3):")
        print(pose[:3, :3])

        # 误差
        error_mm = np.linalg.norm(fk_pos - measured) * 1000
        print(f"\n位置误差: {error_mm:.2f} mm")
        if error_mm < 2.0:
            print("✅ FK测试通过 (< 2 mm)")
        elif error_mm < 5.0:
            print("⚠️  FK误差较大 (2-5 mm)，检查URDF或测量")
        else:
            print("❌ FK误差过大 (> 5 mm)，请检查URDF和测量")

    except Exception as e:
        print(f"❌ FK测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_inverse_kinematics(kin, robot):
    """IK测试：从位置A初值求解用户输入目标，对比关节角"""
    print("\n" + "=" * 60)
    print("IK 测试 - 不同初始值收敛测试")
    print("=" * 60)
    print("\n本测试将自动控制电机力矩，允许你手动拖拽机械臂。")
    print("程序会依次卸力让你移动机械臂，移动完成后自动上力。")
    print("请务必在听到提示后再移动，注意安全。")

    try:
        # 位置 A
        print("\n>>> 即将卸力，请准备移动机械臂到【位置A】...")
        robot.bus.disable_torque()
        input("现在可以自由移动机械臂到【位置A】，移动完成后按 Enter 继续...")
        joints_a = get_joint_angles(robot)
        print(f"位置A关节角度: {joints_a}")
        robot.bus.enable_torque()
        print("已恢复力矩。")

        # 用 joints_a 做 FK 得到旋转矩阵
        pose_a = kin.forward_kinematics(joints_a)
        R_a = pose_a[:3, :3]  # 旋转矩阵部分沿用位置A的FK结果
        print("\n位置A的FK旋转矩阵 R_a (3x3):")
        print(R_a)

        # 位置 B
        print("\n>>> 即将再次卸力，请准备移动机械臂到【位置B】...")
        robot.bus.disable_torque()
        input("移动到【位置B】后，按 Enter 继续...")
        joints_b = get_joint_angles(robot)
        print(f"位置B关节角度: {joints_b}")
        robot.bus.enable_torque()
        print("已恢复力矩。")

        # 用户手动输入目标位置
        print("\n请输入目标末端坐标 (单位: 米)，姿态将沿用位置A的旋转矩阵:")
        try:
            x = float(input("x = "))
            y = float(input("y = "))
            z = float(input("z = "))
        except ValueError:
            print("输入无效，使用位置B的坐标作为目标")
            pose_b = kin.forward_kinematics(joints_b)
            x, y, z = pose_b[0, 3], pose_b[1, 3], pose_b[2, 3]

        # 组合目标位姿：旋转矩阵用 R_a，平移向量用用户输入
        desired_pose = np.eye(4)
        desired_pose[:3, :3] = R_a
        desired_pose[:3, 3] = [x, y, z]
        print(f"\n目标位姿 Y (旋转来自FK_a，位置为用户输入):")
        print(desired_pose)

        # 以 joints_a 为初值调用 IK
        ik_result = kin.inverse_kinematics(
            current_joint_pos=joints_a,
            desired_ee_pose=desired_pose,
            position_weight=1.0,
            orientation_weight=0.00,
        )
        print(f"\nIK 输出角度 c: {ik_result}")
        print(f"位置B角度 b:    {joints_b}")
        print(f"角度差异:       {ik_result - joints_b}")

        max_angle_err = np.max(np.abs(ik_result - joints_b))
        print(f"最大角度误差: {max_angle_err:.2f}°")

        # 验证：用 IK 结果做 FK，看末端位置是否回到目标
        verify_pose = kin.forward_kinematics(ik_result)
        pos_err_mm = np.linalg.norm(verify_pose[:3, 3] - [x, y, z]) * 1000
        print(f"末端位置误差 (IK结果FK vs 用户输入目标): {pos_err_mm:.2f} mm")

        # 判断
        if pos_err_mm < 2.0:
            print("✅ IK测试通过 (末端位置误差 < 2 mm)")
        elif pos_err_mm < 5.0:
            print("⚠️  IK末端位置误差较大 (2-5 mm)")
        else:
            print("❌ IK末端位置误差过大 (> 5 mm)")

    except Exception as e:
        print(f"❌ IK测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("Step 2: URDF 与 RobotKinematics 验证")
    print("=" * 60)
    print(f"URDF: {URDF_PATH}")
    print(f"末端 frame: {TARGET_FRAME}")

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

        # 执行测试
        test_forward_kinematics(kin, robot)
        test_inverse_kinematics(kin, robot)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if robot.is_connected:
            robot.disconnect()
            print("\n机器人已断开连接")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()