#!/usr/bin/env python3
"""
Step 8-1: 地鼠检测标定
用户从实时画面上采样黄色地鼠的颜色，确定HSV阈值
支持跨多帧采集不同亮度下的黄色样本
"""

import sys
import numpy as np
import cv2

sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step8_mole_detection")
from mole_detector import compute_hsv_range

OUTPUT_FILE = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step8_mole_detection/mole_calibration.json"


class HSVSampler:
    """交互式HSV采样器"""

    def __init__(self, camera_idx=1):
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(10):
            self.cap.read()

        # 存储HSV样本值（每次点击立即存储当前帧的HSV）
        self.hsv_samples = []

        cv2.namedWindow("HSV Calibrate")
        cv2.setMouseCallback("HSV Calibrate", self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 立即读取当前帧该像素的HSV值并存储
            ret, frame = self.cap.read()
            if ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                if 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]:
                    hsv_val = hsv[y, x].tolist()
                    self.hsv_samples.append(hsv_val)
                    print(f"  样本 #{len(self.hsv_samples)}: H={hsv_val[0]}, S={hsv_val[1]}, V={hsv_val[2]} at ({x},{y})")

    def run(self):
        print("=" * 60)
        print("地鼠检测标定 - HSV采样（跨多帧）")
        print("=" * 60)
        print("\n操作说明：")
        print("  1. 确保画面中有地鼠（黄色）")
        print("  2. 在地鼠黄色区域上点击多个采样点（可在不同帧/不同亮度下）")
        print("  3. 按 'c' 根据采样点计算HSV范围")
        print("  4. 按 'r' 重置采样点")
        print("  5. 按 's' 保存配置")
        print("  6. 按 'q' 或 ESC 退出")
        print()

        current_hsv_low = None
        current_hsv_high = None

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # 显示HSV阈值预览
                vis = frame.copy()
                if current_hsv_low is not None and current_hsv_high is not None:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array(current_hsv_low), np.array(current_hsv_high))
                    preview = cv2.bitwise_and(frame, frame, mask=mask)
                    cv2.putText(vis, f"HSV: [{current_hsv_low[0]:.0f},{current_hsv_low[1]:.0f},{current_hsv_low[2]:.0f}] ~ [{current_hsv_high[0]:.0f},{current_hsv_high[1]:.0f},{current_hsv_high[2]:.0f}]",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # 状态栏
                status = f"样本: {len(self.hsv_samples)} | 'c'=计算 'r'=重置 's'=保存 'q'=退出"
                cv2.putText(vis, status, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                cv2.imshow("HSV Calibrate", vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c'):
                    if len(self.hsv_samples) >= 3:
                        current_hsv_low, current_hsv_high = compute_hsv_range(self.hsv_samples, k=2.0)
                        print(f"\n计算的HSV范围 (均值±2σ):")
                        print(f"  Low:  [{current_hsv_low[0]:.1f}, {current_hsv_low[1]:.1f}, {current_hsv_low[2]:.1f}]")
                        print(f"  High: [{current_hsv_high[0]:.1f}, {current_hsv_high[1]:.1f}, {current_hsv_high[2]:.1f}]")
                        samples_arr = np.array(self.hsv_samples)
                        print(f"  样本统计: H={samples_arr[:,0].mean():.1f}±{samples_arr[:,0].std():.1f}, S={samples_arr[:,1].mean():.1f}±{samples_arr[:,1].std():.1f}, V={samples_arr[:,2].mean():.1f}±{samples_arr[:,2].std():.1f}")
                    else:
                        print(f"\n请至少采集3个样本，当前: {len(self.hsv_samples)}")
                elif key == ord('r'):
                    self.hsv_samples = []
                    current_hsv_low = None
                    current_hsv_high = None
                    print("\n已重置采样点")
                elif key == ord('s'):
                    if current_hsv_low is not None and current_hsv_high is not None:
                        import json
                        print("\n请输入最小面积阈值（默认200）: ", end="")
                        area_input = input().strip()
                        area_threshold = int(area_input) if area_input else 200

                        config = {
                            "hsv_low": [float(v) for v in current_hsv_low],
                            "hsv_high": [float(v) for v in current_hsv_high],
                            "area_threshold": area_threshold,
                            "num_samples": len(self.hsv_samples)
                        }
                        with open(OUTPUT_FILE, 'w') as f:
                            json.dump(config, f, indent=2)
                        print(f"\n配置已保存: {OUTPUT_FILE}")
                    else:
                        print("\n请先按 'c' 计算HSV范围")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=int, default=1)
    args = parser.parse_args()

    sampler = HSVSampler(camera_idx=args.camera)
    sampler.run()
