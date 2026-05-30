# tests/test_models/test_base.py
import pytest
import torch
from decoding_in_one.models.base import DecodingModel

def test_decoding_model_is_abstract():
    """DecodingModel 应该是抽象类，不能直接实例化"""
    with pytest.raises(TypeError):
        model = DecodingModel()

def test_decoding_model_has_required_methods():
    """DecodingModel 子类必须实现抽象方法"""
    from decoding_in_one.models.surface_code.conv3d_decoder import SurfaceCodeConv3DDecoder
    from decoding_in_one.models.config import Conv3DModelConfig

    config = Conv3DModelConfig()
    model = SurfaceCodeConv3DDecoder(config)

    assert callable(model.get_input_channels)
    assert callable(model.expected_input_rank)
    assert callable(model.output_shape)

    # 方法返回正确类型
    assert isinstance(model.get_input_channels(), int)
    assert isinstance(model.expected_input_rank(), int)
    assert isinstance(model.output_shape(), tuple)
