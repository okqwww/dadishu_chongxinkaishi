#!/usr/bin/env python3
"""
Step 5-2: 手眼标定求解
根据采集的数据求解 AX=XB，得到 T_cam_to_base
"""

import numpy as np
import cv2
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

# 输出路径
OUTPUT_DIR = Path(__file__).parent
CALIB_DATA_FILE = OUTPUT_DIR / "hand_eye_data.json"
RESULT_FILE = OUTPUT_DIR / "hand_eye_params.yaml"


def load_calibration_data(data_file):
    """加载标定数据"""
    with open(data_file, 'r') as f:
        return json.load(f)


def solve_hand_eye(calibration_data, method=cv2.CALIB_HAND_EYE_TSAI):
    """求解手眼标定 AX=XB

    Args:
        calibration_data: 标定数据列表
        method: 求解算法

    Returns:
        R_cam_to_base, t_cam_to_base
    """
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for data in calibration_data:
        T_base_to_end = np.array(data["T_base_to_end"])
        T_cam_to_board = np.array(data["T_cam_to_board"])

        # calibrateHandEye 要求：
        # R_gripper2base: 末端相对于基座的旋转（这里用 inv(T_base_to_end) 得到末端在基座下的位姿的逆，即基座在末端下的位姿？不，参考代码用的是 inv(T_fk)）
        # 实际上根据 hand_eye_solver.py 的约定 A：用 inv(T_FK)
        T_gripper2base = np.linalg.inv(T_base_to_end)

        R_gripper2base.append(T_gripper2base[:3, :3])
        t_gripper2base.append(T_gripper2base[:3, 3:4])

        # T_cam_to_board 就是 target-in-camera
        R_target2cam.append(T_cam_to_board[:3, :3])
        t_target2cam.append(T_cam_to_board[:3, 3:4])

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method,
    )

    return R_cam2base, t_cam2base


def compute_consistency_error(R_cam2base, t_cam2base, calibration_data):
    """计算一致性误差

    对每个pose计算 T_end_to_board = inv(T_base_to_end) @ inv(T_cam2base) @ T_cam_board
    理想情况下 T_end_to_board 应该是常量（棋盘格与末端的刚性偏移）
    """
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3:4] = t_cam2base
    T_base2cam = np.linalg.inv(T_cam2base)

    end_to_board_list = []
    for data in calibration_data:
        T_fk = np.array(data["T_base_to_end"])
        T_cam_board = np.array(data["T_cam_to_board"])

        # T_end_board = inv(T_fk) @ T_base2cam @ T_cam_board
        T_end_board = np.linalg.inv(T_fk) @ T_base2cam @ T_cam_board
        end_to_board_list.append(T_end_board)

    # 位置标准差
    positions = np.array([T[:3, 3] for T in end_to_board_list])
    pos_std = np.mean(np.std(positions, axis=0))

    # 旋转标准差
    rotations = [Rotation.from_matrix(T[:3, :3]) for T in end_to_board_list]
    mean_rot = Rotation.mean(Rotation.concatenate(rotations))
    angle_diffs = []
    for r in rotations:
        diff = mean_rot.inv() * r
        angle_diffs.append(np.degrees(diff.magnitude()))
    rot_std = np.std(angle_diffs)

    return pos_std, rot_std


def save_result(R_cam2base, t_cam2base, pos_std, rot_std, calibration_count, result_file):
    """保存标定结果"""
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3:4] = t_cam2base

    euler = Rotation.from_matrix(R_cam2base).as_euler('xyz', degrees=True)

    fs = cv2.FileStorage(str(result_file), cv2.FileStorage_WRITE)
    fs.write("T_cam_to_base", T_cam2base)
    fs.write("R_cam_to_base", R_cam2base)
    fs.write("t_cam_to_base", t_cam2base)
    fs.write("euler_xyz_deg", euler.reshape(1, 3))
    fs.write("pos_std_m", float(pos_std))
    fs.write("rot_std_deg", float(rot_std))
    fs.write("num_poses", calibration_count)
    fs.release()

    print(f"\n✅ 标定结果已保存: {result_file}")
    print(f"\n=== 标定结果 ===")
    print(f"位姿数量: {calibration_count}")
    print(f"一致性误差: pos_std={pos_std*1000:.2f}mm, rot_std={rot_std:.3f}°")
    print(f"\n相机在基座坐标系下的位姿:")
    print(f"  平移: X={t_cam2base[0][0]:.4f}m, Y={t_cam2base[1][0]:.4f}m, Z={t_cam2base[2][0]:.4f}m")
    print(f"  旋转 (XYZ欧拉角): R={euler[0]:.1f}°, P={euler[1]:.1f}°, Y={euler[2]:.1f}°")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="手眼标定求解")
    parser.add_argument("--method", "-m", type=str, default=None,
                        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
                        help="指定使用的算法（默认选择pos_std最小的）")
    parser.add_argument("--show-all", "-a", action="store_true",
                        help="显示所有算法的结果")
    parser.add_argument("--data", "-d", type=str, default=None,
                        help="指定标定数据文件（默认使用 hand_eye_data.json）")
    args = parser.parse_args()

    print("=" * 60)
    print("手眼标定 - 求解")
    print("=" * 60)

    if args.data:
        calib_file = Path(args.data)
    else:
        calib_file = CALIB_DATA_FILE

    if not calib_file.exists():
        print(f"❌ 找不到标定数据: {calib_file}")
        print("请先运行 step5_capture.py 采集数据")
        return

    calibration_data = load_calibration_data(calib_file)
    print(f"\n已加载 {len(calibration_data)} 个位姿数据")

    if len(calibration_data) < 3:
        print(f"❌ 至少需要3个位姿，当前只有 {len(calibration_data)} 个")
        return

    # 尝试多种算法
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    print("\n=== 求解结果 ===")
    results = {}

    for name, cv_method in methods.items():
        try:
            R, t = solve_hand_eye(calibration_data, method=cv_method)
            pos_std, rot_std = compute_consistency_error(R, t, calibration_data)
            euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)

            results[name] = {
                "R": R,
                "t": t,
                "pos_std": pos_std,
                "rot_std": rot_std,
                "euler": euler,
            }

            print(f"  {name:>10s}: "
                  f"X={t[0][0]:+.4f} Y={t[1][0]:+.4f} Z={t[2][0]:+.4f}  "
                  f"σ_pos={pos_std*1000:.2f}mm  σ_rot={rot_std:.3f}°  "
                  f"R/P/Y={euler[0]:+.1f}/{euler[1]:+.1f}/{euler[2]:+.1f}°")

        except Exception as e:
            print(f"  {name:>10s}: 求解失败 - {e}")

    # 选择最佳算法
    valid_results = {k: v for k, v in results.items() if "pos_std" in v}
    if not valid_results:
        print("\n❌ 所有算法都求解失败")
        return

    if args.method:
        # 指定算法
        if args.method not in valid_results:
            print(f"\n❌ 算法 '{args.method}' 不可用")
            return
        best_name = args.method
        best = valid_results[best_name]
        print(f"\n使用指定算法: {best_name}")
    else:
        # 自动选择 park（实验表明它与物理测量更一致）
        # 如果需要其他算法，请用 --method 指定
        best_name = "park"
        best = valid_results[best_name]
        print(f"\n★ 自动选择: {best_name}  (与物理测量一致)")

    print(f"   σ_pos={best['pos_std']*1000:.2f}mm  σ_rot={best['rot_std']:.3f}°")

    # 保存结果
    save_result(
        best["R"], best["t"],
        best["pos_std"], best["rot_std"],
        len(calibration_data),
        RESULT_FILE
    )


if __name__ == "__main__":
    main()