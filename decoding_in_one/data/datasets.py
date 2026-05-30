# decoding_in_one/data/datasets.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

def build_dataloaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """构建训练和验证数据加载器

    Args:
        train_x: 训练输入数据
        train_y: 训练目标数据
        val_x: 验证输入数据
        val_y: 验证目标数据
        batch_size: 批大小

    Returns:
        (train_loader, val_loader): 数据加载器元组
    """
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader
