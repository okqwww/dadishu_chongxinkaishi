#!/usr/bin/env python3
"""
Step 4-2: 计算相机内参
功能: 读取棋盘格图像，计算内参矩阵和畸变系数
前提: step4_capture.py 已采集并保存棋盘格图像，用户已删除不准确图片
"""

import cv2
import numpy as np
import glob
import os

# 标定板参数
CHESSBOARD_SIZE = (11, 8)  # 棋盘格内角点数量（列, 行）
SQUARE_SIZE = 0.005  # 棋盘格每个方格的物理尺寸（米）

# 输出文件
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = os.path.join(OUTPUT_DIR, "intrinsic_params.yaml")


def create_chessboard_points(chessboard_size, square_size):
    """生成棋盘格物理坐标点"""
    points = []
    for i in range(chessboard_size[1]):
        for j in range(chessboard_size[0]):
            points.append([j * square_size, i * square_size, 0])
    return np.array(points, dtype=np.float32)


def calibrate_from_images(image_files, chessboard_size, square_size, fix_distortion=False):
    """从一系列标定图像中计算内参

    Args:
        image_files: 标定图像路径列表
        chessboard_size: 棋盘格内角点数 (cols, rows)
        square_size: 方格物理尺寸（米）
        fix_distortion: True=固定畸变系数为0，False=求解畸变系数
    """
    obj_points = []  # 物理坐标
    img_points = []  # 像素坐标
    img_size = None

    objp = create_chessboard_points(chessboard_size, square_size)

    valid_count = 0
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"无法读取图像: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # 亚像素精细化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners)
            valid_count += 1

            print(f"✓ [{valid_count}/{len(image_files)}] {os.path.basename(fname)}: 找到棋盘格")
        else:
            print(f"✗ {os.path.basename(fname)}: 未找到棋盘格")

    if len(obj_points) == 0:
        raise RuntimeError("没有找到有效的标定图像")

    # 执行标定
    if fix_distortion:
        # 固定畸变系数为0，仅求解内参
        initial_dist = np.zeros(5)
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, initial_dist, None
        )
        print("\n注意: 畸变系数固定为0（--no-distortion 模式）")
    else:
        # 正常标定，同时求解内参和畸变
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None
        )

    print(f"\n=== 标定结果 ===")
    print(f"图像尺寸: {img_size}")
    print(f"有效标定图像数: {len(obj_points)}")
    print(f"重投影误差: {ret:.6f}")
    print(f"\n内参矩阵 K:")
    print(f"  fx = {K[0,0]:.4f}, fy = {K[1,1]:.4f}")
    print(f"  cx = {K[0,2]:.4f}, cy = {K[1,2]:.4f}")
    print(f"\n畸变系数 (k1, k2, p1, p2, k3):")
    print(f"  {dist.ravel()}")

    return K, dist, img_size


def save_params(K, dist, img_size, save_file):
    """保存内参到yaml文件"""
    fs = cv2.FileStorage(save_file, cv2.FileStorage_WRITE)
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("img_width", img_size[0])
    fs.write("img_height", img_size[1])
    fs.release()
    print(f"\n参数已保存到: {save_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="计算相机内参")
    parser.add_argument("--no-distortion", action="store_true",
                        help="不使用畸变纠正（适用于无畸变或畸变很小的相机）")
    args = parser.parse_args()

    # 获取标定图像（只用原图，不使用标注图）
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_images")
    image_files = sorted(glob.glob(os.path.join(image_dir, "calib_*_raw.jpg")))

    if len(image_files) == 0:
        print(f"错误: 在 {image_dir} 中未找到标定图像 (calib_*.jpg)")
        print("请先运行 step4_capture.py 采集棋盘格图像")
        return

    print(f"找到 {len(image_files)} 张标定图像:")
    for f in image_files:
        print(f"  - {os.path.basename(f)}")
    print()

    fix_distortion = not args.no_distortion

    K, dist, img_size = calibrate_from_images(image_files, CHESSBOARD_SIZE, SQUARE_SIZE, fix_distortion)
    save_params(K, dist, img_size, SAVE_FILE)

    # 去畸变测试
    if not args.no_distortion:
        print("\n=== 去畸变测试 ===")
        img = cv2.imread(image_files[0])
        h, w = img.shape[:2]
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, K, dist, None, newK)
        test_out = os.path.join(image_dir, "_undistorted_test.jpg")
        cv2.imwrite(test_out, undistorted)
        print(f"去畸变测试图已保存: {test_out}")


if __name__ == "__main__":
    main()