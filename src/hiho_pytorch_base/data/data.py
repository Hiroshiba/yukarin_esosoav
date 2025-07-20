"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor

from hiho_pytorch_base.config import DatasetConfig
from hiho_pytorch_base.data.sampling_data import SamplingData


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    feature_vector: numpy.ndarray
    feature_variable: numpy.ndarray
    target_vector: SamplingData
    target_scalar: float
    speaker_id: int


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    feature_vector: Tensor
    feature_variable: Tensor
    target_vector: Tensor
    target_scalar: Tensor
    speaker_id: Tensor


def preprocess(d: InputData, config: DatasetConfig, is_eval: bool) -> OutputData:
    """データ処理"""
    variable_scalar = numpy.mean(d.feature_variable)
    enhanced_feature = d.feature_vector + variable_scalar

    if not is_eval:
        enhanced_feature += numpy.random.randn(*enhanced_feature.shape) * 0.01

    resampled_data = d.target_vector.resample(
        sampling_rate=config.frame_rate, length=config.frame_length
    )
    target_class = numpy.bincount(resampled_data[:, 0]).argmax()

    return OutputData(
        feature_vector=torch.from_numpy(enhanced_feature).float(),
        feature_variable=torch.from_numpy(d.feature_variable).float(),
        target_vector=torch.tensor(target_class).long(),
        target_scalar=torch.tensor(d.target_scalar).float(),
        speaker_id=torch.tensor(d.speaker_id).long(),
    )
