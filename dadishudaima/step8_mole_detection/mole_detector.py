#!/usr/bin/env python3
"""
Step 8: 地鼠中心点识别模块
基于颜色阈值检测黄色地鼠
"""

import numpy as np
import cv2


class MoleDetector:
    """地鼠检测器：检测图像中所有黄色连通区域"""

    def __init__(self, hsv_low, hsv_high, area_threshold=200):
        """
        Args:
            hsv_low: HSV下限 (h, s, v)
            hsv_high: HSV上限 (h, s, v)
            area_threshold: 最小连通区域面积阈值
        """
        self.hsv_low = np.array(hsv_low, dtype=np.uint8)
        self.hsv_high = np.array(hsv_high, dtype=np.uint8)
        self.area_threshold = area_threshold

    def detect(self, frame):
        """
        检测一帧图像中的所有地鼠中心

        Args:
            frame: BGR图像

        Returns:
            list of (u, v) 元组，表示每个地鼠的中心像素坐标
        """
        # 转换到HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 阈值分割
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        # 形态学处理：去噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤并计算中心
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.area_threshold:
                continue
            # 计算轮廓矩得到中心
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

        return centers

    def detect_with_debug(self, frame):
        """
        检测并返回调试信息

        Args:
            frame: BGR图像

        Returns:
            (centers, debug_image)
            centers: 检测到的地鼠中心列表
            debug_image: 带标注的调试图像
        """
        centers = self.detect(frame)

        # 绘制调试图像
        vis = frame.copy()

        # 转换HSV用于显示mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 画所有有效轮廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.area_threshold:
                continue
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)

        # 画中心点
        for i, (cx, cy) in enumerate(centers):
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(vis, f"#{i+1}", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return centers, vis, mask

    @classmethod
    def from_config(cls, config_file):
        """从配置文件加载"""
        import json
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cls(
            hsv_low=cfg["hsv_low"],
            hsv_high=cfg["hsv_high"],
            area_threshold=cfg.get("area_threshold", 200)
        )

    def save_config(self, config_file):
        """保存配置"""
        import json
        cfg = {
            "hsv_low": self.hsv_low.tolist(),
            "hsv_high": self.hsv_high.tolist(),
            "area_threshold": self.area_threshold
        }
        with open(config_file, 'w') as f:
            json.dump(cfg, f, indent=2)


def sample_hsv_at_points(frame, points):
    """
    在指定像素点处采样HSV值

    Args:
        frame: BGR图像
        points: [(x,y), ...] 像素坐标列表

    Returns:
        采样点的HSV值列表
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    samples = []
    for x, y in points:
        if 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]:
            samples.append(hsv[y, x].tolist())
    return samples


def compute_hsv_range(samples, k=2.0):
    """
    从样本点计算HSV范围（均值 ± k倍标准差）

    Args:
        samples: [[h,s,v], ...] HSV样本列表
        k: 标准差倍数（默认2.0，约覆盖95%）

    Returns:
        (hsv_low, hsv_high)
    """
    samples = np.array(samples)
    h_mean, s_mean, v_mean = samples.mean(axis=0)
    h_std, s_std, v_std = samples.std(axis=0)

    # HSV各通道范围：mean ± k*std
    h_low = max(0, h_mean - k * h_std)
    h_high = min(180, h_mean + k * h_std)
    s_low = max(0, s_mean - k * s_std)
    s_high = min(255, s_mean + k * s_std)
    v_low = max(0, v_mean - k * v_std)
    v_high = min(255, v_mean + k * v_std)

    return (h_low, s_low, v_low), (h_high, s_high, v_high)
