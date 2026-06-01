#!/usr/bin/env python3
"""测试 GNN 解码器功能"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from decoding_in_one.models import GNNModelConfig, SurfaceCodeGNNDecoder
from decoding_in_one.models.config import Conv3DModelConfig


def test_gnn_decoder_creation():
    """测试 GNN 解码器创建"""
    print("=" * 60)
    print("测试 GNN 解码器创建")
    print("=" * 60)

    # GNN 配置（与 Conv3D 参数量相当）
    gnn_config = GNNModelConfig(
        layer_type='GAT',
        num_layers=4,
        hidden_channels=128,
        attention_heads=4,
        dropout=0.1,
    )

    print(f"\n[GNN 配置]")
    print(f"  层数: {gnn_config.num_layers}")
    print(f"  隐藏维度: {gnn_config.hidden_channels}")
    print(f"  注意力头: {gnn_config.attention_heads}")
    print(f"  层类型: {gnn_config.layer_type}")

    # 创建模型
    model = SurfaceCodeGNNDecoder(gnn_config)

    print(f"\n[模型创建成功]")
    print(f"  模型类型: {type(model).__name__}")
    print(f"  输入通道: {model.get_input_channels()}")
    print(f"  输出形状: {model.output_shape()}")
    print(f"  预期输入rank: {model.expected_input_rank()}")

    return model


def test_gnn_forward():
    """测试 GNN 前向传播"""
    print("\n" + "=" * 60)
    print("测试 GNN 前向传播")
    print("=" * 60)

    model = test_gnn_decoder_creation()

    # 创建测试输入 (B=2, C=4, T=5, D=5)
    batch_size = 2
    rounds = 5
    distance = 5

    x = torch.randn(batch_size, 4, rounds, distance, distance)
    print(f"\n[输入数据]")
    print(f"  形状: {x.shape}")
    print(f"  batch_size: {batch_size}")
    print(f"  通道: {x.size(1)}")
    print(f"  时间步: {rounds}")
    print(f"  码距: {distance}")

    try:
        with torch.no_grad():
            output = model(x, distance, rounds)

        print(f"\n[前向传播成功]")
        print(f"  输出形状: {output.shape}")
        print(f"  期望形状: ({batch_size}, 4, {rounds}, {distance}, {distance})")

        if output.shape == (batch_size, 4, rounds, distance, distance):
            print(f"  [OK] 形状匹配！")
        else:
            print(f"  [FAIL] 形状不匹配")

    except Exception as e:
        print(f"\n[FAIL] 前向传播失败: {e}")
        import traceback
        traceback.print_exc()


def test_parameter_count():
    """测试参数量（与 Conv3D 对比）"""
    print("\n" + "=" * 60)
    print("参数量对比")
    print("=" * 60)

    # GNN 模型
    gnn_config = GNNModelConfig(
        layer_type='GAT',
        num_layers=4,
        hidden_channels=128,
        attention_heads=4,
    )
    gnn_model = SurfaceCodeGNNDecoder(gnn_config)
    gnn_params = sum(p.numel() for p in gnn_model.parameters())
    gnn_trainable = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)

    print(f"\n[GNN 模型]")
    print(f"  总参数: {gnn_params:,}")
    print(f"  可训练参数: {gnn_trainable:,}")

    # Conv3D 模型（参考配置）
    conv_config = Conv3DModelConfig(
        num_filters=[128, 128, 128, 128, 4],
        kernel_sizes=[3, 3, 3, 3, 3],
    )
    from decoding_in_one.models import SurfaceCodeConv3DDecoder
    conv_model = SurfaceCodeConv3DDecoder(conv_config)
    conv_params = sum(p.numel() for p in conv_model.parameters())
    conv_trainable = sum(p.numel() for p in conv_model.parameters() if p.requires_grad)

    print(f"\n[Conv3D 模型]")
    print(f"  总参数: {conv_params:,}")
    print(f"  可训练参数: {conv_trainable:,}")

    print(f"\n[对比]")
    print(f"  GNN: {gnn_params:,} 参数")
    print(f"  Conv3D: {conv_params:,} 参数")
    ratio = gnn_params / conv_params if conv_params > 0 else 0
    print(f"  比例: {ratio:.2%}")


if __name__ == "__main__":
    try:
        test_gnn_decoder_creation()
        test_gnn_forward()
        test_parameter_count()

        print("\n" + "=" * 60)
        print("[SUCCESS] 所有测试成功完成！")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
