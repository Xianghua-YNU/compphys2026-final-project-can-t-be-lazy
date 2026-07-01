# train.py
# 功能：构造 PINN 损失函数并训练单个实验案例。

import numpy as np
import torch

from config import device, OscillatorParams
from physics import rk4_solve, sample_data_points, steady_state_amplitude, force_torch
from pinn_model import PINN, derivative
from metrics import compute_error_metrics


def compute_pinn_loss(model, t_data, x_data, t_phys_base, p: OscillatorParams, lambda_phys=1e-1):
    """PINN 总损失：数据损失 + lambda*物理残差损失 + 初始条件损失。"""
    # 数据损失：约束 PINN 预测值接近训练数据点。
    x_pred_data = model(t_data)
    loss_data = torch.mean((x_pred_data - x_data) ** 2)

    # 物理残差损失：将 x_hat(t) 代入控制方程。
    t_phys = t_phys_base.clone().detach().requires_grad_(True)
    x_pred_phys = model(t_phys)

    dx_dt = derivative(x_pred_phys, t_phys)
    d2x_dt2 = derivative(dx_dt, t_phys)

    residual = (
        d2x_dt2
        + 2.0 * p.zeta * p.omega0 * dx_dt
        + (p.omega0 ** 2) * x_pred_phys
        - force_torch(t_phys, p)
    )
    loss_physics = torch.mean(residual ** 2)

    # 初始条件损失：约束 x(0)=x0 与 x'(0)=v0。
    t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
    x0_pred = model(t0)
    dx0_dt = derivative(x0_pred, t0)
    loss_ic = torch.mean((x0_pred - p.x0) ** 2 + (dx0_dt - p.v0) ** 2)

    total_loss = loss_data + lambda_phys * loss_physics + loss_ic
    return total_loss, loss_data, loss_physics, loss_ic


def train_one_case(
    p: OscillatorParams,
    case_name="case",
    lambda_label="",
    zeta_label="",
    A_label="",
    n_data=25,
    n_phys=400,
    lambda_phys=1e-1,
    epochs=20000,
    lr=1e-4,
    hidden_dim=32,
    hidden_layers=4,
):
    """训练一个参数组合下的 PINN，并返回结果字典和作图数据。"""
    print("\n" + "=" * 80)
    print(f"开始训练：{case_name}")
    print(
        f"参数：zeta={p.zeta}, omega0={p.omega0}, "
        f"A={p.A}, Omega={p.Omega}, lambda={lambda_phys}"
    )

    B = steady_state_amplitude(p)
    if p.A != 0:
        print(f"稳态受迫位移幅值 B ≈ {B:.6f}")
        print(f"B / |x0| ≈ {B / (abs(p.x0) + 1e-12):.6f}")
    else:
        print("A=0，无外力驱动")

    t_values, x_true, _ = rk4_solve(p)
    t_data_np, x_data_np = sample_data_points(t_values, x_true, n_data=n_data)

    t_data = torch.tensor(t_data_np, dtype=torch.float32, device=device)
    x_data = torch.tensor(x_data_np, dtype=torch.float32, device=device)

    t_phys_np = np.linspace(0.0, p.T, n_phys).reshape(-1, 1)
    t_phys_base = torch.tensor(t_phys_np, dtype=torch.float32, device=device)

    model = PINN(hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"total": [], "data": [], "physics": [], "ic": []}

    # 训练循环：每一轮同时更新数据损失、物理损失和初值损失。
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        total_loss, loss_data, loss_physics, loss_ic = compute_pinn_loss(
            model=model,
            t_data=t_data,
            x_data=x_data,
            t_phys_base=t_phys_base,
            p=p,
            lambda_phys=lambda_phys,
        )

        total_loss.backward()
        optimizer.step()

        history["total"].append(total_loss.item())
        history["data"].append(loss_data.item())
        history["physics"].append(loss_physics.item())
        history["ic"].append(loss_ic.item())

        if epoch == 1 or epoch % 1000 == 0:
            print(
                f"Epoch {epoch:6d} | "
                f"Total={total_loss.item():.4e} | "
                f"Data={loss_data.item():.4e} | "
                f"Physics={loss_physics.item():.4e} | "
                f"IC={loss_ic.item():.4e}"
            )

    model.eval()
    t_eval = torch.tensor(t_values.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        x_pred = model(t_eval).cpu().numpy().flatten()

    metrics = compute_error_metrics(x_pred=x_pred, x_true=x_true, model=model, p=p, t_values=t_values)

    print(f"\n{case_name} 结果：")
    print(f"相对 L2 误差：      {metrics['relative_l2']:.6%}")
    print(f"RMSE：              {metrics['rmse']:.6e}")
    print(f"最大绝对误差：      {metrics['max_abs_error']:.6e}")
    print(f"物理残差 MSE：      {metrics['physics_mse']:.6e}")

    result = {
        "case_name": case_name,
        "lambda_label": lambda_label,
        "zeta_label": zeta_label,
        "A_label": A_label,
        "zeta": p.zeta,
        "omega0": p.omega0,
        "A": p.A,
        "Omega": p.Omega,
        "x0": p.x0,
        "v0": p.v0,
        "T": p.T,
        "dt": p.dt,
        "B_steady": B,
        "B_over_x0": B / (abs(p.x0) + 1e-12),
        "lambda_phys": lambda_phys,
        "n_data": n_data,
        "n_phys": n_phys,
        "epochs": epochs,
        "lr": lr,
        "relative_l2": metrics["relative_l2"],
        "rmse": metrics["rmse"],
        "max_abs_error": metrics["max_abs_error"],
        "physics_mse": metrics["physics_mse"],
    }

    artifact = {
        "case_name": case_name,
        "lambda_label": lambda_label,
        "zeta_label": zeta_label,
        "A_label": A_label,
        "params": p,
        "t": t_values,
        "x_true": x_true,
        "x_pred": x_pred,
        "t_data": t_data_np.flatten(),
        "x_data": x_data_np.flatten(),
        "history": history,
    }

    return result, artifact
