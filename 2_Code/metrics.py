# metrics.py
# 功能：计算相对 L2 误差、RMSE、最大绝对误差和物理残差 MSE。

import numpy as np
import torch

from config import device, OscillatorParams
from physics import force_torch
from pinn_model import derivative


def relative_l2_error(x_pred, x_true):
    """相对 L2 误差，用于评价整体轨迹误差。"""
    numerator = np.sqrt(np.sum((x_pred - x_true) ** 2))
    denominator = np.sqrt(np.sum(x_true ** 2)) + 1e-12
    return numerator / denominator


def compute_physics_mse(model, p: OscillatorParams, t_values):
    """将 PINN 预测结果代回控制方程，计算物理残差 MSE。"""
    t = torch.tensor(
        t_values.reshape(-1, 1),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    x = model(t)
    dx_dt = derivative(x, t)
    d2x_dt2 = derivative(dx_dt, t)

    residual = (
        d2x_dt2
        + 2.0 * p.zeta * p.omega0 * dx_dt
        + (p.omega0 ** 2) * x
        - force_torch(t, p)
    )
    return torch.mean(residual ** 2).item()


def compute_error_metrics(x_pred, x_true, model, p: OscillatorParams, t_values):
    """统一返回四个主要误差指标。"""
    rel_l2 = relative_l2_error(x_pred, x_true)
    rmse = np.sqrt(np.mean((x_pred - x_true) ** 2))
    max_abs_error = np.max(np.abs(x_pred - x_true))
    physics_mse = compute_physics_mse(model, p, t_values)

    return {
        "relative_l2": rel_l2,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "physics_mse": physics_mse,
    }
