# Step 7: 像素坐标 → 基座坐标转换模块

## 功能

将地鼠的像素坐标(u,v)转换为机械臂基座坐标系下的三维坐标(Xbase, Ybase, Zbase)。

## 原理

分两步：

**第一步：像素 → 托盘坐标 + 深度Zc**
```
s * (u,v,1) = K × T_tray_to_cam × (Xtray, Ytray, 0, 1)
```
其中Ztray=0（地鼠在托盘平面上），3个方程解3个未知数Xtray, Ytray, Zc。

**第二步：像素 + Zc → 基座坐标**
```
(u,v,1) = (1/Zc) × K × T_base_to_cam × (Xbase, Ybase, Zbase, 1)
```
3个方程解3个未知数Xbase, Ybase, Zbase。

## 文件结构

```
step7_pixel_to_base/
├── pixel_to_base.py    # 核心转换模块
├── test_realtime.py   # 实时测试脚本
└── README.md
```

## 使用方法

```bash
cd ~/dadishu_chongxinkaishi/dadishudaima/step7_pixel_to_base
python test_realtime.py --camera 1
```

## 操作说明

1. 运行后显示实时画面
2. 在画面上点击任意点
3. 该点的像素坐标会被转换为基座坐标并打印
4. 按`q`或`ESC`退出

## 输入参数

- 内参K：来自Step 4 (`intrinsic_params.yaml`)
- T_cam_to_base：来自Step 5手眼标定 (`hand_eye_params.yaml`)
- T_tray_to_cam：Step 6 Apriltag实时检测（集成在本模块中）

## 验证方式

1. 点击托盘上已知位置的点（如某个Apriltag的中心）
2. 对比计算出的基座坐标与实测值
3. 误差应该在可接受范围内（<5mm）