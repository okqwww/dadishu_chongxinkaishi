#!/usr/bin/env python3
"""
Step 8: 地鼠检测 + 坐标转换
整合Step7和Step8：检测地鼠像素坐标 → 基座坐标
"""

import sys
import numpy as np
import cv2

# 添加模块路径
sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step7_pixel_to_base")
sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step8_mole_detection")

from pixel_to_base import PixelToBaseConverter, load_params as load_coord_params
from mole_detector import MoleDetector

# Apriltag库兼容
try:
    import pupil_apriltags as apriltag_lib
    _APRILTAG_LIB = "pupil_apriltags"
except ImportError:
    import apriltag as apriltag_lib
    _APRILTAG_LIB = "apriltag"

TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.040
HALF_TAG = TAG_SIZE_M / 2.0
TAG_POSITIONS_MM = {
    0: np.array([0.0, 0.0, 0.0]),
    1: np.array([130.0, 0.0, 0.0]),
    3: np.array([0.0, 130.0, 0.0]),
    2: np.array([130.0, 130.0, 0.0]),
}

MOLE_CONFIG = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step8_mole_detection/mole_calibration.json"


class MoleWithCoordConverter:
    """地鼠检测 + 坐标转换"""

    def __init__(self, camera_idx=1):
        # 加载坐标转换参数
        K, T_cam_to_base = load_coord_params()
        self.converter = PixelToBaseConverter(K, T_cam_to_base)
        self.K = K

        # 加载地鼠检测器
        self.mole_detector = MoleDetector.from_config(MOLE_CONFIG)

        # Apriltag检测器
        if _APRILTAG_LIB == "pupil_apriltags":
            self.apriltag_detector = apriltag_lib.Detector(
                families=TAG_FAMILY,
                nthreads=2,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
            )
        else:
            options = apriltag_lib.DetectorOptions(families=TAG_FAMILY)
            self.apriltag_detector = apriltag_lib.Detector(options)

        # 相机
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(10):
            self.cap.read()

        self.current_T_tray_to_cam = None
        self.current_tags = []

    def detect_apriltag(self, frame):
        """检测Apriltag，返回T_tray_to_cam"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if _APRILTAG_LIB == "pupil_apriltags":
            detections = self.apriltag_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]),
                tag_size=TAG_SIZE_M,
            )
        else:
            detections = self.apriltag_detector.detect(gray)

        if len(detections) == 0:
            return None, []

        # 收集obj和img点
        obj_points = []
        img_points = []

        for det in detections:
            if det.tag_id not in TAG_POSITIONS_MM:
                continue

            tag_center_m = TAG_POSITIONS_MM[det.tag_id] / 1000.0
            corners_in_tag = np.array([
                [-HALF_TAG,  HALF_TAG, 0.0],
                [ HALF_TAG,  HALF_TAG, 0.0],
                [ HALF_TAG, -HALF_TAG, 0.0],
                [-HALF_TAG, -HALF_TAG, 0.0],
            ], dtype=np.float64)
            corners_in_tray = corners_in_tag + tag_center_m
            obj_points.extend(corners_in_tray)
            img_points.extend(det.corners.astype(np.float32))

        if len(obj_points) < 4:
            return None, []

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        try:
            _, rvecs, tvecs = cv2.solvePnPGeneric(
                obj_points, img_points, self.K, np.zeros(5),
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            min_error = float('inf')
            best_idx = 0
            for i in range(len(rvecs)):
                proj, _ = cv2.projectPoints(obj_points, rvecs[i], tvecs[i], self.K, np.zeros(5))
                error = np.linalg.norm(img_points - proj.reshape(-1, 2), axis=1).mean()
                if error < min_error:
                    min_error = error
                    best_idx = i
            rvec = rvecs[best_idx]
            tvec = tvecs[best_idx]
        except Exception:
            _, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, self.K, np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        R, _ = cv2.Rodrigues(rvec)
        T_tray_to_cam = np.eye(4)
        T_tray_to_cam[:3, :3] = R
        T_tray_to_cam[:3, 3] = tvec.flatten()

        used_tags = [det.tag_id for det in detections if det.tag_id in TAG_POSITIONS_MM]
        return T_tray_to_cam, used_tags

    def detect_moles_with_coord(self, frame):
        """检测地鼠并转换为基座坐标"""
        # 检测地鼠
        mole_centers = self.mole_detector.detect(frame)

        if not mole_centers or self.current_T_tray_to_cam is None:
            return []

        results = []
        for u, v in mole_centers:
            # 转换为基座坐标
            Xbase, Ybase, Zbase = self.converter.convert(u, v, self.current_T_tray_to_cam)
            if Xbase is not None:
                results.append({
                    'pixel': (u, v),
                    'base': (Xbase, Ybase, Zbase)
                })

        return results

    def run(self):
        print("=" * 60)
        print("地鼠检测 + 坐标转换 - 实时测试")
        print("=" * 60)
        print("\n操作说明：")
        print("  画面上绿色圆点=Apriltag边框，彩色轴=托盘坐标系")
        print("  红色圆点+标注=检测到的地鼠及其基座坐标(mm)")
        print("  按 'q' 或 ESC 退出")
        print()

        cv2.namedWindow("Mole with Coord", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mole with Coord", 960, 540)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Apriltag检测
                T_tray_to_cam, tags = self.detect_apriltag(frame)
                self.current_T_tray_to_cam = T_tray_to_cam
                self.current_tags = tags

                # 绘制Apriltag
                vis = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if _APRILTAG_LIB == "pupil_apriltags":
                    detections = self.apriltag_detector.detect(
                        gray,
                        estimate_tag_pose=True,
                        camera_params=(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]),
                        tag_size=TAG_SIZE_M,
                    )
                else:
                    detections = self.apriltag_detector.detect(gray)

                for det in detections:
                    corners = det.corners.astype(int)
                    cv2.polylines(vis, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                    ctr = corners.mean(axis=0).astype(int)
                    cv2.putText(vis, f"ID:{det.tag_id}", tuple(ctr - [20, 0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                # 绘制托盘坐标系
                if T_tray_to_cam is not None:
                    rvec, _ = cv2.Rodrigues(T_tray_to_cam[:3, :3])
                    tvec = T_tray_to_cam[:3, 3]
                    cv2.drawFrameAxes(vis, self.K, np.zeros(5), rvec, tvec, 0.05)

                # 地鼠检测+坐标转换
                mole_results = self.detect_moles_with_coord(frame)

                # 绘制地鼠
                for i, result in enumerate(mole_results):
                    u, v = result['pixel']
                    X, Y, Z = result['base']
                    cv2.circle(vis, (u, v), 8, (0, 0, 255), -1)
                    label = f"#{i+1} ({u},{v})→({X*1000:.0f},{Y*1000:.0f},{Z*1000:.0f})mm"
                    cv2.putText(vis, label, (u+10, v-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                    print(f"\r地鼠#{i+1}: 像素=({u},{v}) 基座=({X*1000:.1f},{Y*1000:.1f},{Z*1000:.1f})mm", end="", flush=True)

                # 状态栏
                if T_tray_to_cam is not None:
                    status = f"Tags: {tags} | Apriltag OK | 检测到 {len(mole_results)} 个地鼠"
                else:
                    status = "未检测到有效Tag"
                cv2.putText(vis, status, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1)

                if not mole_results:
                    print(f"\r未检测到地鼠", end="", flush=True)

                cv2.imshow("Mole with Coord", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=int, default=1)
    args = parser.parse_args()

    app = MoleWithCoordConverter(camera_idx=args.camera)
    app.run()
