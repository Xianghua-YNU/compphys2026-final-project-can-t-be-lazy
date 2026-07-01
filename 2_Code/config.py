# config.py
# 项目名称：阻尼/驱动振子的物理神经网络求解
# 功能：集中管理随机种子、输出路径、物理参数和扫描范围。

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """固定随机种子，减少不同运行之间的随机差异。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 统一设备设置：有 CUDA 时使用 GPU，否则使用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 所有结果均使用相对路径保存，避免硬编码绝对路径。
OUTPUT_DIR = "../3_Data/experiment_results_final"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

# 手动指定综合较优 lambda：
# L03 是轨迹误差最小点，L10 是预测误差和物理残差综合较平衡点。
SELECTED_LAMBDA_LABEL = "L10"

# 参数扫描设置。lambda 使用对数间隔，zeta 使用线性间隔。
LAMBDA_VALUES = np.logspace(-4, 0, 20)
ZETA_VALUES = np.linspace(0.02, 0.50, 20)

# A 用来补充验证模型对不同驱动强度的适用性。
# 取 A=0~3，覆盖无驱动、弱驱动、中等驱动和较强驱动。
A_VALUES = np.linspace(0.0, 3.0, 7)


@dataclass
class OscillatorParams:
    """阻尼/驱动简谐振子的物理参数。"""
    zeta: float = 0.10       # 阻尼比 ζ
    omega0: float = 2.0      # 无阻尼固有角频率 ω0
    A: float = 0.0           # 外力项振幅 A
    Omega: float = 1.0       # 驱动角频率 Ω
    x0: float = 1.0          # 初始位移 x(0)
    v0: float = 0.0          # 初始速度 x'(0)
    T: float = 10.0          # 模拟总时间
    dt: float = 0.01         # RK4 时间步长
