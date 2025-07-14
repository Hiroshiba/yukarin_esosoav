"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    feature_vector: numpy.ndarray
    feature_variable: numpy.ndarray
    target_vector: numpy.ndarray
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


def preprocess(d: InputData, is_eval: bool) -> OutputData:
    """データ処理"""
    variable_scalar = numpy.mean(d.feature_variable)
    enhanced_feature = d.feature_vector + variable_scalar

    if is_eval:
        enhanced_feature += numpy.random.randn(*enhanced_feature.shape) * 0.01

    return OutputData(
        feature_vector=torch.from_numpy(enhanced_feature).float(),
        feature_variable=torch.from_numpy(d.feature_variable).float(),
        target_vector=torch.from_numpy(d.target_vector).long(),
        target_scalar=torch.tensor(d.target_scalar).float(),
        speaker_id=torch.tensor(d.speaker_id).long(),
    )
