# tests/test_training/test_trainer.py
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from decoding_in_one.training.trainer import Trainer
from decoding_in_one.training.config import OptimConfig

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def test_trainer_initialization():
    """Trainer 应该正确初始化"""
    model = DummyModel()
    config = OptimConfig(epochs=2, batch_size=4)
    trainer = Trainer(model, config)

    assert trainer.model is model
    assert trainer.config is config

def test_trainer_train_basic():
    """Trainer 应该能够执行基本训练"""
    model = DummyModel()
    config = OptimConfig(epochs=2, batch_size=4, lr=0.01, device="cpu")

    # 创建虚拟数据
    train_x = torch.randn(20, 10)
    train_y = torch.randn(20, 1)
    val_x = torch.randn(10, 10)
    val_y = torch.randn(10, 1)

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_ds, batch_size=4)
    val_loader = DataLoader(val_ds, batch_size=4)

    trainer = Trainer(model, config)
    report = trainer.train(train_loader, val_loader)

    # 检查报告结构
    assert "history" in report
    assert len(report["history"]) == 2  # 2 个 epochs
    assert "best_val_loss" in report
