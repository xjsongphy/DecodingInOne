# decoding_in_one/training/trainer.py
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import OptimConfig


class Trainer:
    """通用 PyTorch 训练器"""

    def __init__(self, model: nn.Module, config: OptimConfig):
        """
        Args:
            model: PyTorch 模型
            config: 训练配置
        """
        self.model = model
        self.config = config
        self.device = self._get_device()
        self.model.to(self.device)

    def _get_device(self) -> torch.device:
        """获取训练设备"""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _set_seed(self, seed: int) -> None:
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module | None = None
    ) -> dict[str, Any]:
        """执行训练循环

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数（默认 BCEWithLogitsLoss）

        Returns:
            训练报告字典
        """
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        best_ckpt = out_dir / "best_model.pt"

        for epoch in range(1, self.config.epochs + 1):
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            seen = 0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * x.size(0)
                seen += x.size(0)

            train_loss = running_loss / max(seen, 1)

            # 验证阶段
            val_loss = self._evaluate(val_loader, criterion)

            row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            history.append(row)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": {
                            "lr": self.config.lr,
                            "weight_decay": self.config.weight_decay,
                        },
                        "epoch": epoch,
                    },
                    best_ckpt,
                )

        final_report = {
            "config": {
                "epochs": self.config.epochs,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
            },
            "device": str(self.device),
            "best_val_loss": float(best_val_loss),
            "best_checkpoint": str(best_ckpt),
            "history": history,
        }

        return final_report

    def _evaluate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * x.size(0)
                total += x.size(0)

        return total_loss / max(total, 1)
