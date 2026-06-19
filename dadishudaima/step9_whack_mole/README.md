# Step 9: 完整打地鼠链路

## 完整链路

```
相机帧 → Apriltag检测 → T_tray_to_cam
                      ↓
              地鼠检测 → 像素(u,v)
                      ↓
              坐标转换 → Xbase,Ybase,Zbase
                      ↓
              FK_OFFSET补偿
                      ↓
                  IK求解
                      ↓
              机械臂移动 → 点击 → 回到初始位置
```

## 文件结构

```
step9_whack_mole/
├── record_home.py       # 记录实际初始位置
├── record_ik_init.py   # 记录IK初始角度
├── whack_mole.py        # 完整链路
├── home_position.json   # 实际初始位置（运行后生成）
├── ik_init_joints.json  # IK初始角度（运行后生成）
└── README.md
```

## 使用流程

### 1. 记录实际初始位置（点击后回到这里）

```bash
cd ~/dadishu_chongxinkaishi/dadishudaima/step9_whack_mole
python record_home.py
```
→ 移动机械臂到点击完成后的位置 → 按 `s` 保存

### 2. 记录IK初始角度（用于IK求解）

```bash
python record_ik_init.py
```
→ 移动机械臂到IK求解的合适初始位置 → 按 `s` 保存

### 3. 运行打地鼠

```bash
python whack_mole.py --camera 1
```
→ 检测到地鼠后自动点击第一个

## 两个位置的区别

- **`home_position.json`**：机械臂完成点击后实际停留的位置，也是按 `h` 键时回到的位置
- **`ik_init_joints.json`**：IK求解时的初始关节角度，应该是一个合理的、与目标位置接近的姿态

## 操作说明

- 按 `h` 回到初始位置
- 按 `q` 退出

## 参数说明

- `FK_OFFSET = [0.0305, -0.0118, 0.0]` (m)
- `FIXED_ROTATION = np.eye(3)` — 末端姿态固定
- 点击后等待1秒后回到初始位置