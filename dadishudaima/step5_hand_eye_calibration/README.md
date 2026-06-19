# Step 5: 手眼标定（Eye-to-Hand）

## 目的

求取相机在机械臂基坐标系下的位姿 `T_cam_to_base`，用于打地鼠时将像素坐标转换为基座坐标系下的机械臂目标位置。

## 原理

通过移动机械臂末端到不同位姿，利用 OpenCV 的 `calibrateHandEye` 求解 AX=XB 问题：

- **A**：末端在基座坐标系下的位姿（FK计算 + FK_OFFSET补偿）
- **B**：棋盘格在相机坐标系下的位姿（PnP检测）
- **X**：相机在基座坐标系下的位姿（待求）

## 分步流程

### Step 5-1: 采集数据

1. 将棋盘格贴在机械臂末端（stylus 附近）
2. 确保棋盘格始终在相机视野内
3. 运行采集程序：

```bash
cd ~/dadishu_chongxinkaishi/dadishudaima/step5_hand_eye_calibration
python step5_capture.py --camera 0 --port /dev/ttyACM0 --num 15
```

4. 操作步骤：
   - 按 `d` 禁用机械臂力矩 → 手动移动到新位姿
   - 按 `e` 启动力矩并自动采集
   - 重复至少 10-15 个不同位姿
   - 按 `q` 退出

5. 检查 `calibration_images/` 中的标注图，删除检测不准确的图片

### Step 5-2: 求解标定

```bash
python step5_solve.py
```

会尝试多种算法（Tsai、Park、Horaud、Andreff、Daniilidis），选择一致性误差最小的结果。

## 输出文件

- `hand_eye_data.json` — 采集的标定数据
- `hand_eye_params.yaml` — 标定结果，包含：
  - `T_cam_to_base`: 4x4 齐次变换矩阵
  - `euler_xyz_deg`: 欧拉角形式
  - `pos_std_m`, `rot_std_deg`: 一致性误差

## 棋盘格参数

- 内角点：11 × 8
- 方格尺寸：5mm × 5mm
- 与 Step 4 相机内参标定使用的标定板相同

## 注意事项

- FK_OFFSET 补偿已在采集数据时应用（`T_base_to_end[:3, 3] += FK_OFFSET`）
- 采集时应覆盖尽可能多样的位姿（不同位置和角度）
- 标定板与末端之间的刚性连接在采集过程中不能改变