# tests/test_models/test_config.py
from decoding_in_one.models.config import Conv3DModelConfig

def test_default_config():
    """默认配置应该正确初始化"""
    config = Conv3DModelConfig()

    assert config.input_channels == 4
    assert config.out_channels == 4
    assert config.num_filters == [64, 64, 64, 4]
    assert config.kernel_sizes == [3, 3, 3, 3]
    assert config.activation == "gelu"
    assert config.dropout == 0.1

def test_custom_config():
    """自定义配置应该正确保存"""
    config = Conv3DModelConfig(
        input_channels=3,
        out_channels=2,
        num_filters=[32, 32, 2],
        kernel_sizes=[5, 5, 5],
        activation="relu",
        dropout=0.2
    )

    assert config.input_channels == 3
    assert config.out_channels == 2
    assert config.num_filters == [32, 32, 2]
    assert config.kernel_sizes == [5, 5, 5]
    assert config.activation == "relu"
    assert config.dropout == 0.2

def test_num_filters_must_match_out_channels():
    """num_filters 最后一层必须等于 out_channels"""
    # 如果不匹配，__post_init__ 应该抛出错误
    try:
        config = Conv3DModelConfig(out_channels=4, num_filters=[64, 64, 64, 8])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_filters[-1] must equal out_channels" in str(e)

def test_num_filters_and_kernel_sizes_must_match_length():
    """num_filters 和 kernel_sizes 长度必须相同"""
    try:
        config = Conv3DModelConfig(num_filters=[64, 64, 4], kernel_sizes=[3, 3])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have the same length" in str(e)
