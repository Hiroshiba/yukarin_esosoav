"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    feature_vector: numpy.ndarray  # 固定長入力ダミーデータ
    feature_variable: numpy.ndarray  # 可変長入力ダミーデータ
    target_vector: numpy.ndarray  # 固定長目標ダミーデータ
    target_scalar: float  # スカラー目標ダミーデータ


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    feature_vector: Tensor
    feature_variable: Tensor
    target_vector: Tensor
    target_scalar: Tensor


def preprocess(d: InputData) -> OutputData:
    """データ処理"""
    variable_scalar = numpy.mean(d.feature_variable)
    enhanced_feature = d.feature_vector + variable_scalar

    return OutputData(
        feature_vector=torch.from_numpy(enhanced_feature).float(),
        feature_variable=torch.from_numpy(d.feature_variable).float(),
        target_vector=torch.from_numpy(d.target_vector).long(),
        target_scalar=torch.tensor(d.target_scalar).float(),
    )
