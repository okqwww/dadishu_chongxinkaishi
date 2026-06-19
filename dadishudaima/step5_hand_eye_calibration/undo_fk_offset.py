#!/usr/bin/env python3
"""
撤销手眼标定数据中的FK_OFFSET补偿
将 hand_eye_data.json 中的 T_base_to_end 减去 FK_OFFSET
"""

import json
import numpy as np
from pathlib import Path

# FK_OFFSET（与step5_capture.py中相同）
FK_OFFSET = np.array([0.0305, -0.0118, 0.0])

INPUT_FILE = Path(__file__).parent / "hand_eye_data.json"
OUTPUT_FILE = Path(__file__).parent / "hand_eye_data_no_offset.json"


def main():
    if not INPUT_FILE.exists():
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    print(f"加载 {len(data)} 个位姿数据")
    print(f"FK_OFFSET: {FK_OFFSET}")

    for item in data:
        T = np.array(item["T_base_to_end"])
        # 减去之前加的偏移
        T[:3, 3] -= FK_OFFSET
        item["T_base_to_end"] = T.tolist()

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ 已保存到: {OUTPUT_FILE}")
    print("请用 --data hand_eye_data_no_offset.json 运行 step5_solve.py")


if __name__ == "__main__":
    main()