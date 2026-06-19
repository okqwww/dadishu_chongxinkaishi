#!/usr/bin/env python3
"""
Step 7: 像素坐标(u,v) → 机械臂基座坐标(Xbase, Ybase, Zbase)
集成Apriltag实时检测 + 两步坐标变换
"""

import numpy as np
import cv2

# Apriltag库兼容
try:
    import pupil_apriltags as apriltag_lib
    _APRILTAG_LIB = "pupil_apriltags"
except ImportError:
    import apriltag as apriltag_lib
    _APRILTAG_LIB = "apriltag"

# Apriltag配置
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.040
HALF_TAG = TAG_SIZE_M / 2.0

# 托盘上4个tag中心的物理位置（mm）
# 托盘坐标系：Tag1中心为原点，X右，Y下，Z垂直向上
TAG_POSITIONS_MM = {
    0: np.array([0.0, 0.0, 0.0]),
    1: np.array([130.0, 0.0, 0.0]),
    3: np.array([0.0, 130.0, 0.0]),
    2: np.array([130.0, 130.0, 0.0]),
}


class PixelToBaseConverter:
    """
    像素坐标(u,v) → 基座坐标(Xbase, Ybase, Zbase)
    集成Apriltag实时检测和两步坐标变换
    """

    def __init__(self, K, T_cam_to_base):
        """
        Args:
            K: 相机内参矩阵 (3x3)
            T_cam_to_base: 相机在基座坐标系下的位姿 (4x4)
        """
        self.K = K
        self.T_cam_to_base = T_cam_to_base
        self.T_base_to_cam = np.linalg.inv(T_cam_to_base)

        # 初始化Apriltag检测器
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

    def detect_apriltag(self, frame):
        """
        检测Apriltag，返回T_tray_to_cam和检测到的tag id列表
        """
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

        # 收集所有tag的角点
        obj_points = []
        img_points = []

        for det in detections:
            if det.tag_id not in TAG_POSITIONS_MM:
                continue

            tag_center_m = TAG_POSITIONS_MM[det.tag_id] / 1000.0

            # tag坐标系下4个角点（顺序：左下、右下、右上、左上）
            corners_in_tag = np.array([
                [-HALF_TAG,  HALF_TAG, 0.0],
                [ HALF_TAG,  HALF_TAG, 0.0],
                [ HALF_TAG, -HALF_TAG, 0.0],
                [-HALF_TAG, -HALF_TAG, 0.0],
            ], dtype=np.float64)

            # 转换到托盘坐标系
            corners_in_tray = corners_in_tag + tag_center_m
            obj_points.extend(corners_in_tray)
            img_points.extend(det.corners.astype(np.float32))

        if len(obj_points) < 4:
            return None, []

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # PnP求解 T_tray_to_cam
        try:
            retval, rvecs, tvecs = cv2.solvePnPGeneric(
                obj_points, img_points, self.K, np.zeros(5),
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            # 选择重投影误差最小的解
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

    def solve_Xtray_Ytray_Zc(self, u, v, T_tray_to_cam):
        """
        第一步：从像素(u,v)和T_tray_to_cam解出Xtray, Ytray, Zc
        公式：s * (u,v,1) = K × T_tray_to_cam × (Xtray, Ytray, 0, 1)
        """
        K = self.K
        R2 = T_tray_to_cam[:3, :3]
        t2 = T_tray_to_cam[:3, 3]

        # 展开方程
        # (u,v,1) = (1/Zc) * K * (R2*Xtray + t2[:2] + [0], R2*Ytray + t2[1], R2*Ztray + t2[2]) 其中Ztray=0
        # 实际上：(u,v,1) * Zc = K * (R2*[Xtray,Ytray,0]ᵀ + t2)
        # 展开：Zc*u = K[0,:]·([Xtray,Ytray,0]·R2 + t2)
        #      Zc*v = K[1,:]·([Xtray,Ytray,0]·R2 + t2)
        #      Zc   = K[2,:]·([Xtray,Ytray,0]·R2 + t2)

        # 写成 A × [Xtray, Ytray, Zc]ᵀ = b 的形式
        # 从 K × T_tray_to_cam 提取
        # M = K @ T_tray_to_cam  # 3x4 矩阵
        M = K @ T_tray_to_cam[:3, :] 
        # 方程：
        # M[0,0]*X + M[0,1]*Y + M[0,3]*Zc = u*(M[2,0]*X + M[2,1]*Y + M[2,3]*Zc)
        # M[1,0]*X + M[1,1]*Y + M[1,3]*Zc = v*(M[2,0]*X + M[2,1]*Y + M[2,3]*Zc)
        # M[2,0]*X + M[2,1]*Y + M[2,3]*Zc = M[2,0]*X + M[2,1]*Y + M[2,3]*Zc  (恒等式，弃掉)

        # 整理：
        # (M[0,0] - u*M[2,0])*X + (M[0,1] - u*M[2,1])*Y + (M[0,3] - u*M[2,3])*Zc = 0
        # (M[1,0] - v*M[2,0])*X + (M[1,1] - v*M[2,1])*Y + (M[1,3] - v*M[2,3])*Zc = 0
        # 加上归一化约束: X² + Y² = d²  (d是托盘对角线半长...或者用第三个方程)

        # 更直接的方法：用克莱默法则或直接求解
        # 实际上把Ztray=0代入后：
        # s*(u,v,1) = K * (R2[:,:2] @ [X,Y] + t2)  # R2的第三列为0因为Ztray=0
        # 所以：s*[:2] = K[:2,:2] @ R2[:2,:2] @ [X,Y] + K[:2,:] @ t2
        #      s = K[2,:] @ t2  (如果K[2,:]是[0,0,1])

        # 实际上相机模型是：
        # [u;v;1] = (1/Zc) * M * [X;Y;0;1]
        # 其中 M = K * T_tray_to_cam (3x4)

        # 提取 M 的前两列（对应Xtray和Ytray）
        M_left = M[:, :2]  # 3x2
        M_right = M[:, 3]   # 3x1

        # 重排方程：[u;v;1] * Zc = M_left * [Xtray;Ytray] + M_right
        # 写成：
        # u*Zc - M_left[0,:]@[X;Y] = M_right[0]
        # v*Zc - M_left[1,:]@[X;Y] = M_right[1]
        # 1*Zc - M_left[2,:]@[X;Y] = M_right[2]

        A = np.array([
            [u - M[0,0], u - M[0,1], M[0,3]],
            [v - M[1,0], v - M[1,1], M[1,3]],
            [  - M[2,0],   - M[2,1], M[2,3]],
        ])
        b = np.array([u*M[2,3] - M[0,3], v*M[2,3] - M[1,3], M[2,3] - M[2,3]])

        # 实际上更简单的方式：
        # 从 s*(u,v,1) = M * (X,Y,0,1)
        # 得到：
        # s*u = M[0,0]*X + M[0,1]*Y + M[0,3]
        # s*v = M[1,0]*X + M[1,1]*Y + M[1,3]
        # s   = M[2,0]*X + M[2,1]*Y + M[2,3]

        A = np.array([
            [M[0,0] - u*M[2,0], M[0,1] - u*M[2,1]],
            [M[1,0] - v*M[2,0], M[1,1] - v*M[2,1]],
        ])
        b = np.array([
            u*M[2,3] - M[0,3],
            v*M[2,3] - M[1,3],
        ])

        try:
            XY = np.linalg.solve(A, b)
            Xtray, Ytray = XY[0], XY[1]
        except np.linalg.LinAlgError:
            return None, None, None

        # 求Zc：用第三个方程
        Zc = M[2,0]*Xtray + M[2,1]*Ytray + M[2,3]

        return Xtray, Ytray, Zc

    # def solve_Xbase_Ybase_Zbase(self, u, v, Zc):
    #     """
    #     第二步：从像素(u,v)和Zc解出Xbase, Ybase, Zbase
    #     公式：(u,v,1) = (1/Zc) × K × T_base_to_cam × (Xbase, Ybase, Zbase, 1)
    #     """
    #     K = self.K
    #     R1 = self.T_base_to_cam[:3, :3]
    #     t1 = self.T_base_to_cam[:3, 3]

    #     # 同样的方法
    #     M = K @ self.T_base_to_cam[:3, :]  # 3x4

    #     A = np.array([
    #         [M[0,0] - u*M[2,0], M[0,1] - u*M[2,1], M[0,2] - u*M[2,2]],
    #         [M[1,0] - v*M[2,0], M[1,1] - v*M[2,1], M[1,2] - v*M[2,2]],
    #     ])
    #     b = np.array([
    #         u*M[2,3] - M[0,3],
    #         v*M[2,3] - M[1,3],
    #     ])

    #     try:
    #         XYZ = np.linalg.solve(A, b)
    #         Xbase, Ybase, Zbase = XYZ[0], XYZ[1], XYZ[2]
    #     except np.linalg.LinAlgError:
    #         return None, None, None

    #     return Xbase, Ybase, Zbase
    def solve_Xbase_Ybase_Zbase(self, u, v, Zc):
        """
        第二步：从像素 (u, v) 和已知深度 Zc 解析求解基座坐标 (Xbase, Ybase, Zbase)
        
        原理：
            Zc * K^{-1} * [u, v, 1]^T = R_base2cam * P_base + t_base2cam
            => P_base = R_base2cam^T * (Zc * K^{-1} * [u, v, 1]^T - t_base2cam)

        注意：
            self.T_base_to_cam 必须是 4x4 齐次矩阵，表示从基座坐标系到相机坐标系的变换，
            即 P_cam = R * P_base + t。
        """
        K_inv = np.linalg.inv(self.K)
        pixel_homogeneous = np.array([u, v, 1.0])
        
        # 相机坐标系下的三维点
        P_cam = Zc * K_inv @ pixel_homogeneous
        
        R = self.T_base_to_cam[:3, :3]
        t = self.T_base_to_cam[:3, 3]
        
        # 变换到基座坐标系
        P_base = R.T @ (P_cam - t)
        
        return P_base[0], P_base[1], P_base[2]

    def convert(self, u, v, T_tray_to_cam):
        """
        完整转换：像素(u,v) → 基座坐标(Xbase, Ybase, Zbase)

        Args:
            u, v: 像素坐标
            T_tray_to_cam: Apriltag实时检测得到的托盘到相机变换矩阵

        Returns:
            (Xbase, Ybase, Zbase) 或 None
        """
        # 第一步
        print(f"第一步的输入：u={u}, v={v}, T_tray_to_cam={T_tray_to_cam}")
        Xtray, Ytray, Zc = self.solve_Xtray_Ytray_Zc(u, v, T_tray_to_cam)
        
        print(f"第一步的输出：Xtray={Xtray}, Ytray={Ytray}, Zc={Zc}")
        if Xtray is None:
            return None, None, None

        # 第二步
        print(f"第二步的输入：u={u}, v={v}, Zc={Zc}")
        Xbase, Ybase, Zbase = self.solve_Xbase_Ybase_Zbase(u, v, Zc)

        print(f"第二步的输出：Xbase={Xbase}, Ybase={Ybase}, Zbase={Zbase}")
        if Xbase is None:
            return None, None, None

        return Xbase, Ybase, Zbase


def load_params():
    """加载所有参数"""
    # 内参
    intrinsic_file = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step4_camera_intrinsic_calibration/intrinsic_params.yaml"
    fs = cv2.FileStorage(intrinsic_file, cv2.FileStorage_READ)
    K = fs.getNode("K").mat()
    fs.release()

    # 手眼标定 T_cam_to_base
    hand_eye_file = "/home/zyj/dadishu_chongxinkaishi/dadishudaima/step5_hand_eye_calibration/hand_eye_params.yaml"
    fs = cv2.FileStorage(hand_eye_file, cv2.FileStorage_READ)
    T_cam_to_base = fs.getNode("T_cam_to_base").mat()
    fs.release()

    return K, T_cam_to_base