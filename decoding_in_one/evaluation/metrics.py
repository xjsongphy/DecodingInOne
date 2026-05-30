# decoding_in_one/evaluation/metrics.py
import torch
from typing import Optional

class MetricsCalculator:
    """解码性能指标计算器"""

    @staticmethod
    def logical_error_rate(
        predicted: torch.Tensor,
        actual: torch.Tensor
    ) -> float:
        """
        计算逻辑错误率

        Args:
            predicted: 预测的逻辑观测值 (batch, n_observables)
            actual: 实际的逻辑观测值 (batch, n_observables)

        Returns:
            逻辑错误率 [0, 1]
        """
        if predicted.shape != actual.shape:
            raise ValueError("Shape mismatch")

        # 检查有多少样本的预测与实际不符
        errors = torch.any(predicted != actual, dim=1)
        ler = errors.float().mean().item()

        return ler

    @staticmethod
    def syndrome_density(syndrome: torch.Tensor) -> float:
        """
        计算症状密度

        Args:
            syndrome: 检测器测量结果 (batch, n_detectors)

        Returns:
            平均症状密度
        """
        return syndrome.float().mean().item()

    @staticmethod
    def syndrome_density_reduction(
        before: torch.Tensor,
        after: torch.Tensor
    ) -> float:
        """
        计算症状密度减少因子

        Args:
            before: 预处理前的症状
            after: 预处理后的症状

        Returns:
            减少因子 (>1 表示改善)
        """
        density_before = MetricsCalculator.syndrome_density(before)
        density_after = MetricsCalculator.syndrome_density(after)

        if density_after == 0:
            return float('inf') if density_before > 0 else 1.0

        return density_before / density_after
