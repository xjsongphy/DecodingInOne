# decoding_in_one/training/trainer.py
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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

        # 创建带时间戳的输出目录
        if hasattr(self, "run_dir"):
            run_dir = Path(self.run_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = Path(self.config.out_dir)
            run_dir = base_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, float]] = []
        iter_history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        best_ckpt = run_dir / "best_model.pt"

        print(f"[Train] Device: {self.device}")
        print(f"[Train] Epochs: {self.config.epochs}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"[Train] Output dir: {run_dir}")

        global_step = 0
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # 训练阶段
            self.model.train()
            running_loss = 0.0
            running_correct = 0
            seen = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)
            for batch_idx, (x, y) in enumerate(pbar):
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                batch_loss = float(loss.item())
                running_loss += batch_loss * x.size(0)
                seen += x.size(0)

                # 计算正确率
                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    batch_correct = (preds == y).float().mean().item()
                    running_correct += batch_correct * x.size(0)

                # 记录每个 iter
                iter_history.append({
                    "epoch": float(epoch),
                    "batch": float(batch_idx + 1),
                    "global_step": float(global_step),
                    "loss": batch_loss,
                    "accuracy": batch_correct,
                })

                pbar.set_postfix({
                    "loss": f"{batch_loss:.6f}",
                    "acc": f"{batch_correct:.4f}"
                })

                global_step += 1

            train_loss = running_loss / max(seen, 1)
            train_acc = running_correct / max(seen, 1)

            # 验证阶段
            val_loss, val_acc = self._evaluate(val_loader, criterion)

            epoch_time = time.time() - epoch_start
            remaining_epochs = self.config.epochs - epoch
            eta = epoch_time * remaining_epochs

            row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "time": float(epoch_time),
            }
            history.append(row)

            # 输出进度
            print(
                f"[Train] Epoch {epoch}/{self.config.epochs} | "
                f"train_loss: {train_loss:.6f} train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.6f} val_acc: {val_acc:.4f} | "
                f"time: {epoch_time:.1f}s | ETA: {self._format_time(eta)}"
            )

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
                        "val_loss": val_loss,
                    },
                    best_ckpt,
                )
                print(f"[Train] New best model saved (val_loss: {val_loss:.6f})")

        # 保存训练历史
        (run_dir / "train_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        (run_dir / "iter_history.json").write_text(
            json.dumps(iter_history, indent=2), encoding="utf-8"
        )

        final_report = {
            "run_dir": str(run_dir),
            "config": {
                "epochs": self.config.epochs,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
            },
            "device": str(self.device),
            "best_val_loss": float(best_val_loss),
            "best_checkpoint": str(best_ckpt),
            "history": history,
            "iter_count": len(iter_history),
        }

        # 保存完整报告
        (run_dir / "full_report.json").write_text(
            json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return final_report

    def _evaluate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> tuple[float, float]:
        """评估模型，返回 (loss, accuracy)"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * x.size(0)

                # 计算正确率
                preds = (torch.sigmoid(logits) > 0.5).float()
                total_correct += (preds == y).float().sum().item()
                total += y.numel()

        return total_loss / max(total // y.shape[1], 1), total_correct / max(total, 1)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m {s}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"
