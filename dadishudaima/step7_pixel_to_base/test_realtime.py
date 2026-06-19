#!/usr/bin/env python3
"""
Step 7: 实时像素→基座坐标转换测试
弹窗显示实时画面，用户点击画面上的点，实时转换模块把该点的XYZbase算出来输出
"""

import sys
import numpy as np
import cv2

sys.path.insert(0, "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step7_pixel_to_base")
from pixel_to_base import PixelToBaseConverter, load_params

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


class InteractiveConverter:
    def __init__(self, camera_idx=1):
        self.K, self.T_cam_to_base = load_params()
        self.converter = PixelToBaseConverter(self.K, self.T_cam_to_base)

        # Apriltag检测器
        if _APRILTAG_LIB == "pupil_apriltags":
            self.detector = apriltag_lib.Detector(
                families=TAG_FAMILY,
                nthreads=2,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
            )
        else:
            options = apriltag_lib.DetectorOptions(families=TAG_FAMILY)
            self.detector = apriltag_lib.Detector(options)

        # 相机
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(10):
            self.cap.read()

        # 点击点
        self.click_point = None
        self.current_T_tray_to_cam = None
        self.current_tags = []

        # 鼠标回调
        cv2.namedWindow("Click to Convert")
        cv2.setMouseCallback("Click to Convert", self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)

    def detect_apriltag(self, frame):
        """检测Apriltag，返回T_tray_to_cam"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if _APRILTAG_LIB == "pupil_apriltags":
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]),
                tag_size=TAG_SIZE_M,
            )
        else:
            detections = self.detector.detect(gray)

        if len(detections) == 0:
            return None, []

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
            retval, rvecs, tvecs = cv2.solvePnPGeneric(
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

    def run(self):
        print("=" * 60)
        print("Step 7: 像素坐标 → 基座坐标 实时转换测试")
        print("=" * 60)
        print("\n操作说明：")
        print("  1. 在实时画面上点击任意点")
        print("  2. 该点的像素坐标会被转换为基座坐标并输出")
        print("  按 'q' 或 ESC 退出")
        print()

        cv2.namedWindow("Click to Convert", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Click to Convert", 960, 540)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Apriltag检测
                T_tray_to_cam, tags = self.detect_apriltag(frame)
                self.current_T_tray_to_cam = T_tray_to_cam
                self.current_tags = tags

                # 绘制tag边框
                vis = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if _APRILTAG_LIB == "pupil_apriltags":
                    detections = self.detector.detect(
                        gray,
                        estimate_tag_pose=True,
                        camera_params=(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]),
                        tag_size=TAG_SIZE_M,
                    )
                else:
                    detections = self.detector.detect(gray)

                for det in detections:
                    corners = det.corners.astype(int)
                    cv2.polylines(vis, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                    ctr = corners.mean(axis=0).astype(int)
                    cv2.putText(vis, f"ID:{det.tag_id}", tuple(ctr - [20, 0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 绘制托盘坐标系
                if T_tray_to_cam is not None:
                    rvec, _ = cv2.Rodrigues(T_tray_to_cam[:3, :3])
                    tvec = T_tray_to_cam[:3, 3]
                    cv2.drawFrameAxes(vis, self.K, np.zeros(5), rvec, tvec, 0.05)

                # 处理点击
                if self.click_point is not None and T_tray_to_cam is not None:
                    u, v = self.click_point
                    self.click_point = None  # 清空
                    print(f"T_tray_to_cam: {T_tray_to_cam}")
                    # 转换
                    Xbase, Ybase, Zbase = self.converter.convert(u, v, T_tray_to_cam)

                    if Xbase is not None:
                        print(f"\n点击像素: ({u}, {v})")
                        print(f"  → 基座坐标: X={Xbase*1000:.2f}mm, Y={Ybase*1000:.2f}mm, Z={Zbase*1000:.2f}mm")

                        # 在图上标记点击点
                        cv2.circle(vis, (u, v), 8, (0, 0, 255), -1)
                        cv2.putText(vis, f"({u},{v})→({Xbase*1000:.1f},{Ybase*1000:.1f},{Zbase*1000:.1f})mm",
                                    (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        print(f"\n点击像素: ({u}, {v}) → 转换失败")

                # 状态栏
                if T_tray_to_cam is not None:
                    status = f"Tags: {tags} | Apriltag OK"
                else:
                    status = "未检测到有效Tag"
                cv2.putText(vis, status, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 180), 1)
                cv2.putText(vis, "[q/ESC] quit | 点击画面转换坐标", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                cv2.imshow("Click to Convert", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=int, default=1)
    args = parser.parse_args()

    app = InteractiveConverter(camera_idx=args.camera)
    app.run()