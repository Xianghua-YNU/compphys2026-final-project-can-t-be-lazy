# main.py
# 功能：项目入口，一键运行全部实验并生成论文图表。

import os

from config import (
    A_VALUES,
    CSV_DIR,
    FIG_DIR,
    OUTPUT_DIR,
    SELECTED_LAMBDA_LABEL,
    LAMBDA_VALUES,
    ZETA_VALUES,
    OscillatorParams,
    device,
    set_seed,
)
from io_utils import build_word_table, save_results_to_csv, save_word_table_csv
from plot_utils import (
    plot_A0_trajectory,
    plot_A_representative_trajectory_compare,
    plot_lambda_representative_error_compare,
    plot_loss,
    plot_metric_curve,
    plot_research_flowchart,
    plot_trajectory,
    plot_zeta_RK4_reference_compare,
    plot_zeta_representative_trajectory_compare,
)
from train import train_one_case


def run_basic_A0():
    """实验一：A=0 基础验证。"""
    p = OscillatorParams(zeta=0.10, omega0=2.0, A=0.0, Omega=1.0, x0=1.0, v0=0.0, T=10.0, dt=0.01)
    result, artifact = train_one_case(p=p, case_name="G1_A0", n_data=25, n_phys=400, lambda_phys=1e-1, epochs=20000, lr=1e-4)
    result["experiment_group"] = "group1_A0_basic"
    return [result], artifact


def run_lambda_sweep_A1():
    """实验二：固定 A=1、Omega=1、zeta=0.10，扫描 lambda。"""
    results = []
    artifacts = {}

    for idx, lam in enumerate(LAMBDA_VALUES, start=1):
        lambda_label = f"L{idx:02d}"
        p = OscillatorParams(zeta=0.10, omega0=2.0, A=1.0, Omega=1.0, x0=1.0, v0=0.0, T=10.0, dt=0.01)
        result, artifact = train_one_case(
            p=p,
            case_name=f"G2_{lambda_label}",
            lambda_label=lambda_label,
            n_data=25,
            n_phys=400,
            lambda_phys=float(lam),
            epochs=20000,
            lr=1e-4,
        )
        result["experiment_group"] = "group2_lambda_sweep_A1"
        results.append(result)
        artifacts[lambda_label] = artifact

    return results, artifacts


def run_zeta_sweep_A1(selected_lambda):
    """实验三：固定综合较优 lambda，扫描 zeta。"""
    results = []
    artifacts = {}

    for idx, zeta in enumerate(ZETA_VALUES, start=1):
        zeta_label = f"Z{idx:02d}"
        p = OscillatorParams(zeta=float(zeta), omega0=2.0, A=1.0, Omega=1.0, x0=1.0, v0=0.0, T=10.0, dt=0.01)
        result, artifact = train_one_case(
            p=p,
            case_name=f"G3_{zeta_label}",
            zeta_label=zeta_label,
            n_data=25,
            n_phys=400,
            lambda_phys=selected_lambda,
            epochs=20000,
            lr=1e-4,
        )
        result["experiment_group"] = "group3_zeta_sweep_A1"
        results.append(result)
        artifacts[zeta_label] = artifact

    return results, artifacts


def run_A_sweep(selected_lambda):
    """实验四：固定综合较优 lambda 和 zeta=0.10，扫描 A。"""
    results = []
    artifacts = {}

    for idx, A in enumerate(A_VALUES, start=1):
        A_label = f"A{idx:02d}"
        p = OscillatorParams(zeta=0.10, omega0=2.0, A=float(A), Omega=1.0, x0=1.0, v0=0.0, T=10.0, dt=0.01)
        result, artifact = train_one_case(
            p=p,
            case_name=f"G4_{A_label}",
            A_label=A_label,
            n_data=25,
            n_phys=400,
            lambda_phys=selected_lambda,
            epochs=20000,
            lr=1e-4,
        )
        result["experiment_group"] = "group4_A_sweep"
        results.append(result)
        artifacts[A_label] = artifact

    return results, artifacts


def get_result_by_label(results, label_key, label_value):
    """按编号提取某个实验结果。"""
    for item in results:
        if item[label_key] == label_value:
            return item
    raise ValueError(f"没有找到 {label_key} = {label_value} 的结果。")


def main():
    """一键运行全部实验。"""
    set_seed(42)
    print("当前使用设备：", device)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    all_results = []

    # 图1：研究流程图
    plot_research_flowchart("Fig1_research_flowchart.png")

    # 第一组：A=0 基础验证
    print("\n\n############################")
    print("第一组实验：A=0 基础验证")
    print("############################")
    basic_results, basic_artifact = run_basic_A0()
    save_results_to_csv(basic_results, "group1_A0_basic_results_full.csv")
    plot_A0_trajectory(basic_artifact, filename="Fig2_A0_trajectory.png")
    plot_loss(basic_artifact, filename="Fig3_A0_loss.png", title="A=0: training loss")

    basic_table = [{
        "实验": "A=0 基础验证",
        "相对L2误差(%)": f"{basic_results[0]['relative_l2'] * 100:.6f}",
        "RMSE": f"{basic_results[0]['rmse']:.6e}",
        "最大绝对误差": f"{basic_results[0]['max_abs_error']:.6e}",
        "物理残差MSE": f"{basic_results[0]['physics_mse']:.6e}",
    }]
    save_word_table_csv(basic_table, "table_group1_A0_for_word.csv")
    all_results.extend(basic_results)

    # 第二组：lambda 扫描
    print("\n\n############################")
    print("第二组实验：固定 A=1, Omega=1, zeta=0.10，扫描 lambda")
    print("############################")
    lambda_results, lambda_artifacts = run_lambda_sweep_A1()
    save_results_to_csv(lambda_results, "group2_lambda_sweep_A1_full.csv")
    save_word_table_csv(build_word_table(lambda_results, "lambda_phys", "lambda"), "table_lambda_for_word.csv")
    all_results.extend(lambda_results)

    plot_metric_curve(lambda_results, "lambda_phys", "relative_l2", "lambda", "Relative L2 error", "Relative L2 error vs lambda", "Fig4_lambda_relative_L2.png", log_x=True)
    plot_metric_curve(lambda_results, "lambda_phys", "physics_mse", "lambda", "Physics residual MSE", "Physics residual MSE vs lambda", "Fig5_lambda_physics_MSE.png", log_x=True)

    selected_lambda_result = get_result_by_label(lambda_results, "lambda_label", SELECTED_LAMBDA_LABEL)
    selected_lambda = selected_lambda_result["lambda_phys"]
    selected_lambda_artifact = lambda_artifacts[SELECTED_LAMBDA_LABEL]

    print("\n" + "-" * 80)
    print("综合较优 lambda 选择：")
    print(f"selected lambda label = {SELECTED_LAMBDA_LABEL}")
    print(f"selected lambda = {selected_lambda:.6e}")
    print(f"相对 L2 误差 = {selected_lambda_result['relative_l2']:.6%}")
    print(f"RMSE = {selected_lambda_result['rmse']:.6e}")
    print(f"最大绝对误差 = {selected_lambda_result['max_abs_error']:.6e}")
    print(f"物理残差 MSE = {selected_lambda_result['physics_mse']:.6e}")
    print("-" * 80)

    selected_lambda_labels = [
        ("L01", "L01: lambda too small"),
        (SELECTED_LAMBDA_LABEL, f"{SELECTED_LAMBDA_LABEL}: selected lambda"),
        ("L20", "L20: lambda large"),
    ]
    plot_lambda_representative_error_compare(lambda_artifacts, selected_lambda_labels, "Fig6_lambda_representative_error_compare.png")
    plot_trajectory(selected_lambda_artifact, "Fig7_selected_lambda_trajectory.png", f"Selected lambda ({SELECTED_LAMBDA_LABEL}) PINN and RK4 trajectory comparison")

    # 第三组：zeta 扫描
    print("\n\n############################")
    print("第三组实验：固定综合较优 lambda，扫描 zeta")
    print("############################")
    zeta_results, zeta_artifacts = run_zeta_sweep_A1(selected_lambda=selected_lambda)
    save_results_to_csv(zeta_results, "group3_zeta_sweep_A1_full.csv")
    save_word_table_csv(build_word_table(zeta_results, "zeta", "zeta"), "table_zeta_for_word.csv")
    all_results.extend(zeta_results)

    plot_metric_curve(zeta_results, "zeta", "relative_l2", "zeta", "Relative L2 error", "Relative L2 error vs zeta", "Fig8_zeta_relative_L2.png")
    plot_metric_curve(zeta_results, "zeta", "physics_mse", "zeta", "Physics residual MSE", "Physics residual MSE vs zeta", "Fig9_zeta_physics_MSE.png")
    plot_zeta_RK4_reference_compare(zeta_artifacts, zeta_results, "Fig10_zeta_RK4_reference_compare.png")
    plot_zeta_representative_trajectory_compare(zeta_artifacts, zeta_results, "Fig11_zeta_representative_trajectory_compare.png")

    # 第四组：A 扫描
    print("\n\n############################")
    print("第四组实验：固定综合较优 lambda 和 zeta=0.10，扫描 A")
    print("############################")
    A_results, A_artifacts = run_A_sweep(selected_lambda=selected_lambda)
    save_results_to_csv(A_results, "group4_A_sweep_full.csv")
    save_word_table_csv(build_word_table(A_results, "A", "A"), "table_A_for_word.csv")
    all_results.extend(A_results)

    plot_metric_curve(A_results, "A", "relative_l2", "A", "Relative L2 error", "Relative L2 error vs A", "Fig12_A_relative_L2.png")
    plot_metric_curve(A_results, "A", "physics_mse", "A", "Physics residual MSE", "Physics residual MSE vs A", "Fig13_A_physics_MSE.png")
    plot_A_representative_trajectory_compare(A_artifacts, A_results, "Fig14_A_representative_trajectory_compare.png")

    # 总结果
    save_results_to_csv(all_results, "all_experiment_results_full.csv")

    print("\n\n实验全部完成。")
    print(f"图片保存在：{FIG_DIR}")
    print(f"CSV 保存在：{CSV_DIR}")


if __name__ == "__main__":
    main()
