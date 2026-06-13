# decoding_in_one/decoders/neural.py
"""神经网络解码器：使用训练好的模型进行推理"""
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .base import Decoder, Correction
from ..models.surface_code.transforms import dets_to_conv3d_input, reduce_conv3d_output


class NeuralDecoder(Decoder):
    """使用训练好的神经网络模型进行 dense logical field 预测

    输出说明：
    - 模型输出：dense logical field (B,4,T,D,D)
    - Correction.predictions：dense field 或聚合后的 logical observable

    注意：这里的 Correction.predictions 表示 dense logical field 预测，
    不是传统意义上的 qubit correction。
    """

    def __init__(
        self,
        model,
        checkpoint_path: str,
        rounds: int,
        distance: int,
        basis: str = "X",
        code_rotation: str = "XV",
        threshold: float = 0.5,
        reduce_output: bool = False,
        reduce_method: Literal["parity", "mean", "max", "vote"] = "parity"
    ):
        """
        Args:
            model: DecodingModel 实例
            checkpoint_path: 模型权重路径
            rounds: 电路轮数（用于数据变换）
            distance: 码距（用于数据变换）
            basis: 测量基 ('X' 或 'Z')
            code_rotation: 码旋转方向 ('XV', 'XH', 'ZV', 'ZH')
            threshold: 预测阈值
            reduce_output: 是否将 dense output 聚合为 observable 向量
            reduce_method: 聚合方法 ("parity" 推荐, "mean", "max", "vote")
        """
        self.model = model
        self.rounds = rounds
        self.distance = distance
        self.basis = basis.upper()
        self.code_rotation = code_rotation.upper()
        self.threshold = threshold
        self.reduce_output = reduce_output
        self.reduce_method = reduce_method

        # 加载权重
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)

        # 设置设备和评估模式
        self.device = next(self.model.parameters()).device
        self.model.eval()

    def decode(self, syndrome, observables=None) -> Correction:
        """
        Args:
            syndrome: 原始 ancilla 测量结果
                      形状 (batch, T, D²-1) 或 (batch, T*(D²-1))
            observables: 观测量（可选，用于评估）

        Returns:
            Correction: dense logical field 预测 (B,4,T,D,D)
                       或聚合后的 logical observable (B,)
        """
        # 1. 转换为 numpy
        if torch.is_tensor(syndrome):
            syndrome_np = syndrome.cpu().numpy()
        else:
            syndrome_np = np.asarray(syndrome, dtype=np.float32)

        # 2. 使用 transforms 转换为模型输入
        x_np = dets_to_conv3d_input(
            syndrome_np,
            self.rounds,
            self.distance,
            basis=self.basis,
            code_rotation=self.code_rotation,
        )

        # 3. 转换为 tensor 并移到正确设备
        x = torch.from_numpy(x_np).to(self.device)

        # 4. 模型推理
        with torch.no_grad():
            logits = self.model(x)
            pred = (torch.sigmoid(logits) >= self.threshold).float()

        # 5. 可选：聚合为 observable 向量
        if self.reduce_output:
            pred_np = pred.cpu().numpy()
            reduced = reduce_conv3d_output(
                pred_np,
                distance=self.distance,
                code_rotation=self.code_rotation,
                basis=self.basis,
                method=self.reduce_method,
            )
            pred = torch.from_numpy(reduced).to(self.device)

        return Correction(predictions=pred)

    def get_name(self) -> str:
        """返回解码器名称"""
        return "NeuralDecoder"
