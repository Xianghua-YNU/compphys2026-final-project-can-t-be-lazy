# physics.py
# 功能：实现阻尼/驱动振子的物理方程、A=0 解析解和 RK4 参考解。

import numpy as np
import torch

from config import OscillatorParams


def force_np(t, p: OscillatorParams):
    """NumPy 版本外力项：F(t)=A*cos(Omega*t)。"""
    return p.A * np.cos(p.Omega * t)


def force_torch(t, p: OscillatorParams):
    """PyTorch 版本外力项，用于 PINN 物理残差计算。"""
    return p.A * torch.cos(p.Omega * t)


def steady_state_amplitude(p: OscillatorParams):
    """估计线性受迫振动的稳态位移幅值。"""
    denominator = np.sqrt(
        (p.omega0 ** 2 - p.Omega ** 2) ** 2
        + (2.0 * p.zeta * p.omega0 * p.Omega) ** 2
    )
    return p.A / denominator if denominator != 0 else np.inf


def analytical_solution_A0(t, p: OscillatorParams):
    """
    A=0 欠阻尼解析解。

    控制方程：
        x'' + 2*zeta*omega0*x' + omega0^2*x = 0

    欠阻尼条件：
        zeta < 1

    解析解：
        x(t)=exp(-zeta*omega0*t)
        * [x0*cos(omega_d*t) + ((v0+zeta*omega0*x0)/omega_d)*sin(omega_d*t)]
    """
    if p.A != 0:
        raise ValueError("analytical_solution_A0 只用于 A=0 情况。")
    if p.zeta >= 1:
        raise ValueError("当前解析解函数只写了欠阻尼 zeta < 1 的形式。")

    omega_d = p.omega0 * np.sqrt(1.0 - p.zeta ** 2)
    c1 = p.x0
    c2 = (p.v0 + p.zeta * p.omega0 * p.x0) / omega_d

    x = np.exp(-p.zeta * p.omega0 * t) * (
        c1 * np.cos(omega_d * t) + c2 * np.sin(omega_d * t)
    )
    return x


def oscillator_rhs(t, y, p: OscillatorParams):
    """将二阶 ODE 化成一阶系统，供 RK4 使用。"""
    x, v = y
    dxdt = v
    dvdt = force_np(t, p) - 2.0 * p.zeta * p.omega0 * v - (p.omega0 ** 2) * x
    return np.array([dxdt, dvdt], dtype=np.float64)


def rk4_solve(p: OscillatorParams):
    """四阶 Runge-Kutta 方法生成参考解。"""
    t_values = np.arange(0.0, p.T + p.dt, p.dt)
    y = np.array([p.x0, p.v0], dtype=np.float64)

    x_values = []
    v_values = []

    # 逐个时间步推进状态变量 [x, v]。
    for t in t_values:
        x_values.append(y[0])
        v_values.append(y[1])

        h = p.dt
        k1 = oscillator_rhs(t, y, p)
        k2 = oscillator_rhs(t + h / 2.0, y + h * k1 / 2.0, p)
        k3 = oscillator_rhs(t + h / 2.0, y + h * k2 / 2.0, p)
        k4 = oscillator_rhs(t + h, y + h * k3, p)

        y = y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return t_values, np.array(x_values), np.array(v_values)


def sample_data_points(t_values, x_values, n_data: int = 25):
    """从 RK4 参考解中等间隔抽取训练数据点。"""
    indices = np.linspace(0, len(t_values) - 1, n_data).astype(int)
    t_data = t_values[indices].reshape(-1, 1)
    x_data = x_values[indices].reshape(-1, 1)
    return t_data, x_data
