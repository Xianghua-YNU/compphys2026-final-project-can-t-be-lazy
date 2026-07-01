# plot_utils.py
# 功能：绘制论文所需的 Fig1-Fig14。

import os

import matplotlib.pyplot as plt
import numpy as np

from config import FIG_DIR
from physics import analytical_solution_A0


def plot_research_flowchart(filename="Fig1_research_flowchart.png"):
    """绘制 PINN 求解流程图。"""
    steps = [
        "Damped / driven oscillator equation",
        "Construct physics residual r(t)",
        "Physics loss: mean squared residual",
        "Total loss: L = L_data + lambda L_physics + L_IC",
        "PINN predicts x(t)",
        "Compare with analytical solution / RK4 reference",
        "Analyze effects of lambda, zeta and A",
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    y_positions = np.linspace(0.90, 0.10, len(steps))

    for i, (text, y) in enumerate(zip(steps, y_positions)):
        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45", linewidth=1.5),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[i + 1] + 0.055),
                xytext=(0.5, y - 0.055),
                arrowprops=dict(arrowstyle="->", linewidth=1.5),
            )

    ax.set_title("PINN workflow for damped / driven oscillator", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_metric_curve(results, x_key, y_key, x_label, y_label, title, filename, log_x=False):
    """绘制参数扫描下的误差指标变化曲线。"""
    x = np.array([item[x_key] for item in results], dtype=float)
    y = np.array([item[y_key] for item in results], dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    if log_x:
        plt.xscale("log")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_A0_trajectory(artifact, filename="Fig2_A0_trajectory.png"):
    """绘制 A=0 时解析解、RK4 参考解和 PINN 预测结果。"""
    p = artifact["params"]
    t = artifact["t"]
    x_true = artifact["x_true"]
    x_pred = artifact["x_pred"]
    t_data = artifact["t_data"]
    x_data = artifact["x_data"]
    x_ana = analytical_solution_A0(t, p)

    plt.figure(figsize=(8, 5))
    plt.plot(t, x_ana, label="Analytical solution", linewidth=2)
    plt.plot(t, x_true, "--", label="RK4 reference", linewidth=2)
    plt.plot(t, x_pred, ":", label="PINN prediction", linewidth=2)
    plt.scatter(t_data, x_data, s=20, label="Training data")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("A=0: PINN, analytical solution and RK4 reference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_loss(artifact, filename, title):
    """绘制总损失、数据损失、物理损失和初始条件损失。"""
    history = artifact["history"]

    plt.figure(figsize=(8, 5))
    plt.semilogy(history["total"], label="Total loss")
    plt.semilogy(history["data"], label="Data loss")
    plt.semilogy(history["physics"], label="Physics loss")
    plt.semilogy(history["ic"], label="IC loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_trajectory(artifact, filename, title):
    """绘制单个参数组合下 PINN 与 RK4 的轨迹对比。"""
    t = artifact["t"]
    x_true = artifact["x_true"]
    x_pred = artifact["x_pred"]
    t_data = artifact["t_data"]
    x_data = artifact["x_data"]

    plt.figure(figsize=(8, 5))
    plt.plot(t, x_true, label="RK4 reference", linewidth=2)
    plt.plot(t, x_pred, "--", label="PINN prediction", linewidth=2)
    plt.scatter(t_data, x_data, s=20, label="Training data")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_lambda_representative_error_compare(lambda_artifacts, selected_labels, filename):
    """绘制代表性 lambda 下的绝对误差对比。"""
    plt.figure(figsize=(8, 5))

    for label, description in selected_labels:
        artifact = lambda_artifacts[label]
        t = artifact["t"]
        x_true = artifact["x_true"]
        x_pred = artifact["x_pred"]
        plt.plot(t, np.abs(x_pred - x_true), linewidth=2, label=description)

    plt.xlabel("t")
    plt.ylabel("|PINN - RK4|")
    plt.title("Representative absolute error under different lambda")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def find_closest_result(results, key, target):
    """从扫描结果中找到最接近目标参数的结果。"""
    return min(results, key=lambda item: abs(item[key] - target))


def plot_zeta_RK4_reference_compare(zeta_artifacts, zeta_results, filename):
    """绘制不同阻尼比下的 RK4 参考轨迹对比。"""
    target_settings = [
        (0.02, "small damping"),
        (0.18, "medium damping"),
        (0.34, "large damping"),
        (0.50, "very large damping"),
    ]

    plt.figure(figsize=(8, 5))
    used = set()

    for target_zeta, desc in target_settings:
        result = find_closest_result(zeta_results, "zeta", target_zeta)
        label = result["zeta_label"]
        if label in used:
            continue
        used.add(label)
        artifact = zeta_artifacts[label]
        plt.plot(
            artifact["t"],
            artifact["x_true"],
            linewidth=2,
            label=f"{label}: zeta={result['zeta']:.3f}, {desc}",
        )

    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("RK4 reference trajectories under different zeta")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_zeta_representative_trajectory_compare(zeta_artifacts, zeta_results, filename):
    """绘制典型阻尼比下 PINN 与 RK4 轨迹对比。"""
    target_settings = [(0.02, "small damping"), (0.23, "medium damping"), (0.50, "large damping")]
    selected = []
    used = set()

    for target_zeta, desc in target_settings:
        result = find_closest_result(zeta_results, "zeta", target_zeta)
        label = result["zeta_label"]
        if label in used:
            continue
        used.add(label)
        selected.append((result, zeta_artifacts[label], desc))

    fig, axes = plt.subplots(len(selected), 1, figsize=(8, 4 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]

    for ax, (result, artifact, desc) in zip(axes, selected):
        ax.plot(artifact["t"], artifact["x_true"], label="RK4 reference", linewidth=2)
        ax.plot(artifact["t"], artifact["x_pred"], "--", label="PINN prediction", linewidth=2)
        ax.scatter(artifact["t_data"], artifact["x_data"], s=15, label="Training data")
        ax.set_ylabel("x(t)")
        ax.set_title(f"{result['zeta_label']}: zeta={result['zeta']:.3f}, {desc}")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("t")
    fig.suptitle("Representative PINN and RK4 comparison under different zeta", y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_A_representative_trajectory_compare(A_artifacts, A_results, filename):
    """绘制典型驱动强度下 PINN 与 RK4 轨迹对比。"""
    target_settings = [(0.5, "weak driving"), (1.0, "main case"), (3.0, "stronger driving")]
    selected = []
    used = set()

    for target_A, desc in target_settings:
        result = find_closest_result(A_results, "A", target_A)
        label = result["A_label"]
        if label in used:
            continue
        used.add(label)
        selected.append((result, A_artifacts[label], desc))

    fig, axes = plt.subplots(len(selected), 1, figsize=(8, 4 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]

    for ax, (result, artifact, desc) in zip(axes, selected):
        ax.plot(artifact["t"], artifact["x_true"], label="RK4 reference", linewidth=2)
        ax.plot(artifact["t"], artifact["x_pred"], "--", label="PINN prediction", linewidth=2)
        ax.scatter(artifact["t_data"], artifact["x_data"], s=15, label="Training data")
        ax.set_ylabel("x(t)")
        ax.set_title(f"{result['A_label']}: A={result['A']:.2f}, {desc}")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("t")
    fig.suptitle("Representative PINN and RK4 comparison under different A", y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()
