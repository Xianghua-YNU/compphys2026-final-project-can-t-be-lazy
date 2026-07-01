# pinn_model.py
# 功能：定义 PINN 网络结构和自动微分函数。

import torch
import torch.nn as nn


class PINN(nn.Module):
    """输入时间 t，输出位移预测 x_hat(t) 的全连接神经网络。"""

    def __init__(self, hidden_dim: int = 32, hidden_layers: int = 4):
        super().__init__()

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


def derivative(y, x):
    """通过自动微分计算 dy/dx，用于构造 x'(t) 和 x''(t)。"""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
