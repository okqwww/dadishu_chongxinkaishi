# Step 8: 地鼠中心点识别模块

## 功能

基于HSV颜色阈值检测黄色地鼠，并转换为机械臂基座坐标系坐标。

## 文件结构

```
step8_mole_detection/
├── mole_detector.py       # 核心检测模块
├── calibrate.py            # HSV颜色标定
├── test_realtime.py        # 实时测试（仅检测）
├── test_with_coord.py      # 实时测试（检测+坐标转换）
└── README.md
```

## 使用流程

### 1. HSV标定

```bash
cd ~/dadishu_chongxinkaishi/dadishudaima/step8_mole_detection
python calibrate.py --camera 1
```

操作：
1. 确保画面中有地鼠（黄色）
2. 在地鼠黄色区域上点击多个采样点（建议5-10个）
3. 按 `c` 计算HSV范围
4. 按 `s` 保存配置

### 2. 仅检测测试

```bash
python test_realtime.py --camera 1
```

### 3. 检测+坐标转换测试

```bash
python test_with_coord.py --camera 1
```

此脚本整合了Step7的Apriltag检测和坐标转换功能，同时输出：
- 地鼠的像素坐标 `(u, v)`
- 地鼠的基座坐标 `(Xbase, Ybase, Zbase)` in mm

## 输出示例

```
地鼠#1: 像素=(320,240) 基座=(125.3,45.7,289.2)mm
```

## 配置

配置文件：`mole_calibration.json`

```json
{
  "hsv_low": [h, s, v],
  "hsv_high": [h, s, v],
  "area_threshold": 200
}
```
