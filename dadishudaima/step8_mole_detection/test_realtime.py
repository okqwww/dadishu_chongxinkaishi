#!/usr/bin/env python3
"""
Step 8-2: 地鼠检测实时测试
加载配置，实时显示检测结果
"""

import sys
import numpy as np
import cv2

sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step8_mole_detection")
from mole_detector import MoleDetector

CONFIG_FILE = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step8_mole_detection/mole_calibration.json"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=int, default=1)
    parser.add_argument("--config", default=CONFIG_FILE)
    args = parser.parse_args()

    print("=" * 60)
    print("地鼠检测 - 实时测试")
    print("=" * 60)

    # 加载配置
    try:
        detector = MoleDetector.from_config(args.config)
        print(f"\n已加载配置: {args.config}")
        print(f"  HSV范围: {detector.hsv_low} ~ {detector.hsv_high}")
        print(f"  面积阈值: {detector.area_threshold}")
    except FileNotFoundError:
        print(f"\n❌ 配置文件不存在: {args.config}")
        print("请先运行 calibrate.py 进行标定")
        return
    except Exception as e:
        print(f"\n❌ 配置加载失败: {e}")
        return

    # 打开相机
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(10):
        cap.read()
    print("\n按 'q' 或 ESC 退出")

    cv2.namedWindow("Mole Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mole Detection", 960, 540)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # 检测
            centers, vis, mask = detector.detect_with_debug(frame)

            # 显示mask（左上角小图）
            h, w = frame.shape[:2]
            mask_resized = cv2.resize(mask, (w//4, h//4))
            mask_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            vis[0:h//4, 0:w//4] = mask_color
            cv2.putText(vis, "Mask", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 状态栏
            if centers:
                status = f"检测到 {len(centers)} 个地鼠: {centers}"
                cv2.putText(vis, status, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                print(f"\r检测到 {len(centers)} 个地鼠: {centers}", end="", flush=True)
            else:
                status = "未检测到地鼠"
                cv2.putText(vis, status, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                print(f"\r未检测到地鼠", end="", flush=True)

            cv2.imshow("Mole Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
