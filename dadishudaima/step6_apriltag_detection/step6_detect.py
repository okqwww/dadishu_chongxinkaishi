#!/usr/bin/env python3
"""
Step 6: Apriltag检测模块
实时检测托盘上的4个Apriltag，收集所有角点，一次性PnP计算 T_cam_to_tray
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Apriltag库兼容
try:
    import pupil_apriltags as apriltag_lib
    _APRILTAG_LIB = "pupil_apriltags"
except ImportError:
    import apriltag as apriltag_lib
    _APRILTAG_LIB = "apriltag"

# 路径配置
INTRINSIC_FILE = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step4_camera_intrinsic_calibration/intrinsic_params.yaml"
OUTPUT_DIR = Path(__file__).parent
RESULT_FILE = OUTPUT_DIR / "apriltag_params.yaml"

# Apriltag配置
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.040          # tag边长40mm
HALF_TAG = TAG_SIZE_M / 2   # 0.02m

# 托盘上4个tag中心的物理位置（mm）
# 托盘坐标系定义：Tag1中心为原点，X右，Y下，Z垂直于托盘平面向上
# Tag1(0,0,0) - 原点
# Tag0(0,130,0) - 右上角（注意：根据坐标系，Y向下为正，所以130在下方）
# Tag2(130,0,0) - 左下角（X向右为正）
# Tag3(130,130,0) - 右下角
TAG_POSITIONS_MM = {
    0: np.array([0.0, 0.0, 0.0]),
    1: np.array([130.0, 0.0, 0.0]),
    3: np.array([0.0, 130.0, 0.0]),
    2: np.array([130.0, 130.0, 0.0]),
}

# 从tag坐标系到托盘坐标系的旋转矩阵
# tag坐标系：X下，Y右，Z垂直指外
# 托盘坐标系：X右，Y下，Z向上
# 需要将tag坐标系的轴映射到托盘坐标系：
#   tag X (下) -> 托盘 Y (下)   =>  第二列 [0,1,0]
#   tag Y (右) -> 托盘 X (右)   =>  第一列 [1,0,0]
#   tag Z (外) -> 托盘 -Z (向里)？但Z向上，需要分析：tag Z垂直指外（朝向相机），当相机向下拍摄时，tag Z指向相机即向上，因此与托盘Z同向。
#   但实际R_tag_to_tray已在之前代码中定义为 [[0,1,0],[-1,0,0],[0,0,1]]，经过验证可保持，这里沿用。
R_TAG_TO_TRAY = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1]
], dtype=np.float64)


def load_intrinsics(intrinsic_file):
    """加载相机内参"""
    fs = cv2.FileStorage(intrinsic_file, cv2.FileStorage_READ)
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    img_width = int(fs.getNode("img_width").real())
    img_height = int(fs.getNode("img_height").real())
    fs.release()
    return K, dist, img_width, img_height


class AprilTagDetector:
    def __init__(self, K, dist, img_width, img_height):
        self.K = K
        self.dist = dist
        self.img_width = img_width
        self.img_height = img_height

        # 初始化检测器
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

    def detect(self, frame):
        """检测一帧图像中的所有Apriltag"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if _APRILTAG_LIB == "pupil_apriltags":
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]),
                tag_size=TAG_SIZE_M,
            )
        else:
            # 若使用老版apriltag，则需要手动PnP，但此处暂不处理
            detections = self.detector.detect(gray)
            # 对于老版库，需要手动估计姿态，但用户当前环境是pupil_apriltags，所以忽略
            print("使用老版apriltag，未实现手动PnP，请安装pupil_apriltags")
        return detections

    def get_T_cam_to_tray(self, detections):
        """
        收集所有tag的角点，一次性PnP求解 T_cam_to_tray。
        返回 T_cam_to_tray (4x4) 和 检测到的tag id列表。
        """
        obj_points = []  # 3D点（托盘坐标系）
        img_points = []  # 2D点（像素坐标）

        for det in detections:
            if det.tag_id not in TAG_POSITIONS_MM:
                continue

            # tag中心在托盘坐标系下的位置
            tag_center_m = TAG_POSITIONS_MM[det.tag_id] / 1000.0

            # 该tag的四个角点在tag坐标系下的坐标（顺序：左下、右下、右上、左上）
            corners_in_tag = np.array([
                [ -HALF_TAG, HALF_TAG, 0.0],  # 左下
                [ HALF_TAG,  HALF_TAG, 0.0],  # 右下
                [HALF_TAG,  -HALF_TAG, 0.0],  # 右上
                [-HALF_TAG, -HALF_TAG, 0.0],  # 左上
            ], dtype=np.float64)

            
            corners_in_tray =  corners_in_tag + tag_center_m

            obj_points.extend(corners_in_tray)

            # 检测到的像素角点（顺序保证为左下、右下、右上、左上）
            img_points.extend(det.corners.astype(np.float32))

        if len(obj_points) < 4:
            return None, []

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # 一次性PnP求解托盘 -> 相机的外参
        # 使用 SOLVEPNP_IPPE_SQUARE 更稳定，但要求4个点共面且提供两个解，我们选择重投影误差小的那个
        # 如果IPPE失败，回退到 SOLVEPNP_ITERATIVE
        try:
            retval, rvecs, tvecs = cv2.solvePnPGeneric(
                obj_points, img_points, self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            # solvePnPGeneric 返回两个解（rvecs[0],tvecs[0]）和（rvecs[1],tvecs[1]）
            # 计算重投影误差，选择小的
            min_error = float('inf')
            best_idx = 0
            for i in range(len(rvecs)):
                proj, _ = cv2.projectPoints(obj_points, rvecs[i], tvecs[i], self.K, self.dist)
                error = np.linalg.norm(img_points - proj.reshape(-1, 2), axis=1).mean()
                if error < min_error:
                    min_error = error
                    best_idx = i
            rvec = rvecs[best_idx]
            tvec = tvecs[best_idx]
        except Exception:
            # 回退到普通迭代法
            _, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, self.K, self.dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        # 构造 T_tray_to_cam
        R, _ = cv2.Rodrigues(rvec)
        T_tray_to_cam = np.eye(4)
        T_tray_to_cam[:3, :3] = R
        T_tray_to_cam[:3, 3] = tvec.flatten()

        # 求逆得到 T_cam_to_tray
        T_cam_to_tray = np.linalg.inv(T_tray_to_cam)

        used_tags = [det.tag_id for det in detections if det.tag_id in TAG_POSITIONS_MM]
        return T_cam_to_tray, used_tags

    def draw_debug(self, frame, detections, T_cam_to_tray=None):
        """绘制调试画面：显示tag边框、ID、以及托盘坐标系"""
        vis = frame.copy()

        # 绘制每个tag的边框和ID
        for det in detections:
            corners = det.corners.astype(int)
            cv2.polylines(vis, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            ctr = corners.mean(axis=0).astype(int)
            cv2.putText(vis, f"ID:{det.tag_id}", tuple(ctr - [20, 0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 如果得到了托盘位姿，绘制托盘坐标系和原点
        if T_cam_to_tray is not None:
            # 托盘坐标系原点在相机坐标系下的坐标：T_cam_to_tray 的平移部分就是托盘原点在相机系的位置
            t_tray_in_cam = np.linalg.inv(T_cam_to_tray)[:3, 3]

            R_tray_to_cam = np.linalg.inv(T_cam_to_tray[:3, :3])  # 也是 T_tray_to_cam[:3,:3]
            rvec, _ = cv2.Rodrigues(R_tray_to_cam)

            # 绘制托盘坐标轴（长度5cm）
            cv2.drawFrameAxes(vis, self.K, self.dist, rvec, t_tray_in_cam, 0.05)

            # 绘制托盘原点投影
            origin_px = self.K @ t_tray_in_cam
            if origin_px[2] > 0:
                u, v = int(origin_px[0] / origin_px[2]), int(origin_px[1] / origin_px[2])
                cv2.circle(vis, (u, v), 5, (0, 0, 255), -1)
                cv2.putText(vis, "Tray Origin", (u + 5, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return vis


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Apriltag检测模块")
    parser.add_argument("--camera", "-c", type=int, default=0, help="相机设备号")
    parser.add_argument("--save", "-s", action="store_true", help="保存结果到apriltag_params.yaml")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 6: Apriltag检测模块 (一次性PnP)")
    print("=" * 60)

    # 加载内参
    print(f"\n加载内参: {INTRINSIC_FILE}")
    K, dist, img_width, img_height = load_intrinsics(INTRINSIC_FILE)
    print(f"  内参矩阵:\n{K}")
    print(f"  图像分辨率: {img_width} x {img_height}")

    # 初始化检测器
    detector = AprilTagDetector(K, dist, img_width, img_height)

    # 连接相机
    print(f"\n连接相机 (index={args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(img_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(img_height))
    if not cap.isOpened():
        print(f"❌ 无法打开相机 {args.camera}")
        return
    # 预热
    for _ in range(10):
        cap.read()
    print("✅ 相机连接成功")

    print("\n按 'q' 退出，按 's' 保存当前结果")
    print("绿色边框 = tag边框，彩色轴 = 托盘坐标系，红点 = 托盘原点")

    cv2.namedWindow("AprilTag Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Detection", 960, 540)

    T_cam_to_tray_saved = None
    detected_tags = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # 检测
            detections = detector.detect(frame)

            # 计算 T_cam_to_tray
            T_cam_to_tray = None
            if detections:
                T_cam_to_tray, detected_ids = detector.get_T_cam_to_tray(detections)
                detected_tags = detected_ids

            # 绘制调试画面
            vis = detector.draw_debug(frame, detections, T_cam_to_tray)

            # 状态栏
            if T_cam_to_tray is not None:
                status = (f"Tags: {detected_tags} | "
                          f"T_tray_to_cam: [{np.linalg.inv(T_cam_to_tray)[0,3]:.3f}, {np.linalg.inv(T_cam_to_tray)[1,3]:.3f}, {np.linalg.inv(T_cam_to_tray)[2,3]:.3f}]")
            else:
                status = "未检测到有效Tag"
            cv2.putText(vis, status, (10, vis.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 180), 1)
            cv2.putText(vis, "[q] quit [s] save", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("AprilTag Detection", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s') and T_cam_to_tray is not None:
                T_cam_to_tray_saved = T_cam_to_tray.copy()
                print(f"\n已保存当前结果")

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 保存结果
    if args.save and T_cam_to_tray_saved is not None:
        T = T_cam_to_tray_saved
        fs = cv2.FileStorage(str(RESULT_FILE), cv2.FileStorage_WRITE)
        # fs.write("T_cam_to_tray", T)
        fs.write("T_tray_to_cam", np.linalg.inv(T))
        fs.write("detected_tags", detected_tags)
        fs.write("tag_positions_mm", {str(k): v.tolist() for k, v in TAG_POSITIONS_MM.items()})
        fs.release()
        print(f"\n✅ 结果已保存: {RESULT_FILE}")
    elif T_cam_to_tray_saved is not None:
        print(f"\n检测结果:")
        print(f"  T_cam_to_tray:\n{T_cam_to_tray_saved}")
        print(f"\n如需保存，请运行时加 --save 参数")


if __name__ == "__main__":
    main()