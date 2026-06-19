import cv2
import numpy as np
from pupil_apriltags import Detector

# 1. 初始化 pupil-apriltags 检测器 (以常用的 tag36h11 为例)
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# 2. 读取图片并转换为灰度图
image_path = '/home/zyj/图片/摄像头/2026-06-11-110910.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path)
if img is None:
    print(f"无法读取图片: {image_path}，请检查路径。")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 执行检测 (pupil-apriltags 直接接收灰度图)
tags = at_detector.detect(gray)
print(f"检测到 {len(tags)} 个 AprilTag。")

# 4. 遍历并可视化方向
for tag in tags:
    # 获取 4 个顶点的像素坐标 (形状为 4x2)
    # 正确顺序：0:左下, 1:右下, 2:右上, 3:左上
    corners = tag.corners.astype(int)
    p0 = tuple(corners[0])  # 0号点：左下 (Bottom-Left)
    p1 = tuple(corners[1])  # 1号点：右下 (Bottom-Right)
    p2 = tuple(corners[2])  # 2号点：右上 (Top-Right)
    p3 = tuple(corners[3])  # 3号点：左上 (Top-Left)
    
    # 标签中心点坐标
    center = tuple(tag.center.astype(int))
    
    # ------------------ 判定上下左右 ------------------
    # 修正：3号点到 2号点的连线就是“上方”那条边
    top_edge_x = int((p3[0] + p2[0]) / 2)
    top_edge_y = int((p3[1] + p2[1]) / 2)
    top_edge_center = (top_edge_x, top_edge_y)
    
    # 5. 绘制可视化图形
    # 按照 0->1->2->3->0 的顺序顺时针连线（画面上呈现为：下 -> 右 -> 上 -> 左）
    cv2.line(img, p0, p1, (0, 0, 255), 2)   # 下边：红色
    cv2.line(img, p1, p2, (0, 255, 0), 2)   # 右边：绿色
    cv2.line(img, p2, p3, (255, 0, 0), 2)   # 上边：蓝色
    cv2.line(img, p3, p0, (0, 255, 255), 2) # 左边：黄色
    
    # 标记 4 个顶点编号
    for i, corner in enumerate(corners):
        cv2.circle(img, tuple(corner), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i), (corner[0] - 10, corner[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    # 在 3-2 连线（上边缘）的上方标出 "TOP" 字样
    cv2.putText(img, "TOP", (top_edge_center[0], top_edge_center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 标出 Tag ID
    cv2.putText(img, f"ID: {tag.tag_id}", (center[0] - 20, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 6. 显示结果窗口
cv2.imshow('AprilTag pupil-detector', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
