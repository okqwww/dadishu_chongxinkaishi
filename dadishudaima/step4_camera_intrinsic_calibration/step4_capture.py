#!/usr/bin/env python3
"""
Step 4-1: 采集棋盘格标定图像
功能: 采集棋盘格图像，检测角点并标注，保存到本地
不进行内参计算，由用户人工检查后删除不准确图片
"""

import cv2
import numpy as np
import os
import sys
import time
import threading

# 标定板参数
CHESSBOARD_SIZE = (11, 8)  # 棋盘格内角点数量（列, 行）
SQUARE_SIZE = 0.005  # 棋盘格每个方格的物理尺寸（米）

# 输出目录
CALIB_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_images")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="采集棋盘格标定图像")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="相机设备号")
    parser.add_argument("--num", "-n", type=int, default=20,
                        help="需要采集的图像数量（默认20）")
    args = parser.parse_args()

    os.makedirs(CALIB_IMAGE_DIR, exist_ok=True)

    print("=== 采集棋盘格标定图像 ===")
    print(f"标定板参数: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} 内角点, 方格{SQUARE_SIZE*1000}mm")
    print(f"图像保存到: {CALIB_IMAGE_DIR}")
    print(f"采集目标: {args.num} 张")
    print()
    print("操作说明:")
    print("  按 'c' 或 'Enter' 键 - 检测到棋盘格时采集当前图像")
    print("  按 'q' 键 - 退出采集")
    print("  按 's' 键 - 保存当前帧预览（带角点标注）")
    print()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误: 无法打开相机 {args.camera}")
        return

    print(f"相机分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print()

    capture_count = 0
    key_value = [None]

    def keyboard_listener():
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                key_value[0] = line.strip().lower()
            except:
                break

    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    last_status_time = time.time()
    last_detection = False

    print("开始采集... (等待键盘输入)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取帧失败")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 查找棋盘格
            ret_cb, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

            current_time = time.time()

            # 每秒打印一次状态
            if current_time - last_status_time >= 1.0:
                status = "✓ 棋盘格已检测" if ret_cb else "✗ 未检测到棋盘格"
                key_hint = " [可按 c 采集]" if ret_cb else ""
                print(f"[{capture_count}/{args.num}] {status}{key_hint}")
                last_status_time = current_time
                last_detection = ret_cb

            # 处理按键
            if key_value[0] is not None:
                key = key_value[0]
                key_value[0] = None

                if key == 'c' or key == '':
                    if ret_cb:
                        # 亚像素精细化
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        refined_corners = cv2.cornerSubPix(gray, corners.copy(), (11, 11), (-1, -1), criteria)

                        # 保存原图
                        raw_path = os.path.join(CALIB_IMAGE_DIR, f"calib_{capture_count:02d}_raw.jpg")
                        cv2.imwrite(raw_path, frame)

                        # 保存带角点标注的图
                        display_frame = frame.copy()
                        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, refined_corners, True)
                        ann_path = os.path.join(CALIB_IMAGE_DIR, f"calib_{capture_count:02d}_ann.jpg")
                        cv2.imwrite(ann_path, display_frame)

                        capture_count += 1
                        print(f"  → 已采集第 {capture_count} 张: 原图={raw_path}, 标注图={ann_path}")
                    else:
                        print(f"  → 未检测到棋盘格，无法采集")

                elif key == 'q':
                    print(f"\n用户退出采集，共 {capture_count} 张图像")
                    break

                elif key == 's':
                    # 保存当前帧预览（带角点标注，仅用于查看）
                    preview_display = frame.copy()
                    if ret_cb:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        refined_corners = cv2.cornerSubPix(gray, corners.copy(), (11, 11), (-1, -1), criteria)
                        cv2.drawChessboardCorners(preview_display, CHESSBOARD_SIZE, refined_corners, True)
                    preview_path = os.path.join(CALIB_IMAGE_DIR, "_preview.jpg")
                    cv2.imwrite(preview_path, preview_display)
                    print(f"  → 预览图已保存: {preview_path}")

                print("请输入命令 (c=采集, q=退出, s=预览): ", end="", flush=True)

            if capture_count >= args.num:
                print(f"\n已达到目标数量({args.num}张)，退出采集")
                break

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        cap.release()

    print(f"\n共采集 {capture_count} 张图像")
    print(f"图像位置: {CALIB_IMAGE_DIR}")
    print("\n请人工检查图像，删除角点识别不准确的图片")
    print("确认后运行 step4_calibrate.py 计算内参")


if __name__ == "__main__":
    main()