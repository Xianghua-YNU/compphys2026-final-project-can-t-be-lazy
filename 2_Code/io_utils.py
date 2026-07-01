# io_utils.py
# 功能：保存完整实验结果和论文表格所需 CSV 文件。

import csv
import os

from config import CSV_DIR


def save_results_to_csv(results, filename):
    """保存完整实验结果，便于复查全部参数和误差指标。"""
    if len(results) == 0:
        return

    filepath = os.path.join(CSV_DIR, filename)
    fieldnames = list(results[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"完整结果已保存：{filepath}")


def save_word_table_csv(rows, filename):
    """保存论文正文中使用的精简表格。"""
    if len(rows) == 0:
        return

    filepath = os.path.join(CSV_DIR, filename)
    fieldnames = list(rows[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Word 精简表已保存：{filepath}")


def build_word_table(rows, param_key, param_name):
    """从完整结果中提取论文表格需要的主要指标。"""
    table_rows = []

    for item in rows:
        label = item.get("lambda_label", "") or item.get("zeta_label", "") or item.get("A_label", "")

        table_rows.append({
            "编号": label,
            param_name: f"{item[param_key]:.6e}" if param_key == "lambda_phys" else f"{item[param_key]:.6f}",
            "相对L2误差(%)": f"{item['relative_l2'] * 100:.6f}",
            "RMSE": f"{item['rmse']:.6e}",
            "最大绝对误差": f"{item['max_abs_error']:.6e}",
            "物理残差MSE": f"{item['physics_mse']:.6e}",
        })

    return table_rows
