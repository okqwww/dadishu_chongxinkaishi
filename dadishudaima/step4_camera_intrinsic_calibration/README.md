# Step 4: 相机内参标定

## 分步流程

### Step 4-1: 采集图像
```bash
python step4_capture.py --camera 0 --num 20
```

- 检测到棋盘格后按 `c` 或 `Enter` 采集
- 按 `q` 退出采集
- 采集的图像已带角点标注

### Step 4-2: 人工检查 + 计算内参

1. 人工检查采集的图像，删除角点识别不准确的图片

2. 计算内参：
```bash
# 默认模式（计算畸变）
python step4_calibrate.py

# 无畸变模式（相机畸变很小时使用）
python step4_calibrate.py --no-distortion
```

## 标定板参数

- 棋盘格：11×8 内角点
- 方格尺寸：5mm × 5mm
- 可在代码中修改 `CHESSBOARD_SIZE` 和 `SQUARE_SIZE`

## 输出文件

`intrinsic_params.yaml`:
- K: 内参矩阵 (fx, fy, cx, cy)
- dist: 畸变系数 (k1, k2, p1, p2, k3)
- img_width, img_height: 图像尺寸

## 图像目录

图像保存在 `calibration_images/` 目录（脚本同目录下），包含：
- `calib_XX.jpg` — 带角点标注的标定图像
- `_preview.jpg` — 预览图