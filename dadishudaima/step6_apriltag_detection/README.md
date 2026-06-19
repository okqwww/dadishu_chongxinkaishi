# Step 6: Apriltag检测模块

## 目的

实时检测托盘上的4个Apriltag，计算**相机到托盘坐标系**的变换矩阵 `T_cam_to_tray`。

## 托盘坐标系定义

- **原点**：Tag0（右上角tag）的中心
- **X轴**：朝左（Tag0→Tag1方向）
- **Y轴**：朝下（Tag0→Tag3方向）
- **Z轴**：垂直托盘平面朝上

## 4个Tag的物理位置（mm）

| Tag ID | 位置 (mm) | 说明 |
|--------|-----------|------|
| 0 | (0, 0, 0) | 右上角，原点 |
| 1 | (130, 0, 0) | 左上角 |
| 2 | (130, 130, 0) | 左下角 |
| 3 | (0, 130, 0) | 右下角 |

## 使用方法

```bash
cd ~/dadishu_chongxinkaishi/dadishudaima/step6_apriltag_detection
python step6_detect.py --camera 0
```

## 操作说明

- **按 `q` 或 `ESC`** — 退出
- **按 `s`** — 保存当前检测结果

## 可视化内容

- 绿色边框 — 检测到的tag边框和ID
- 彩色轴 — tag坐标系（X红、Y绿、Z蓝）
- 蓝色点 — 托盘原点（Tag0中心）

## 输出文件

`apriltag_params.yaml`：
- `T_cam_to_tray`: 4×4 齐次变换矩阵
- `tag_positions_mm`: 4个tag的物理位置

## 算法说明

1. 对每个检测到的tag，用PnP（`SOLVEPNP_IPPE_SQUARE`）估计其相对于相机的位姿
2. 对Tag0：`T_cam_to_tray = T_cam_to_tag0`（直接就是托盘原点）
3. 对其他tag：先用坐标变换得到各自的`T_cam_to_tray_i`
4. 多tag时，融合旋转（SO(3) SVD平均）和平移（均值）

## 注意事项

- tag边长：40mm（与你的实际tag一致）
- tag家族：tag36h11
- 需要先完成Step 4相机内参标定