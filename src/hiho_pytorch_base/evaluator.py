"""モデル評価モジュール"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hiho_pytorch_base.dataset import BatchOutput
from hiho_pytorch_base.generator import Generator, GeneratorOutput
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価時の出力。特に含まないといけない値はない"""

    loss: Tensor
    accuracy: Tensor


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値を計算する関数。高いほど良い。"""
    return output.accuracy


class Evaluator(nn.Module):
    """モデルの評価を行うクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    def forward(self, data: BatchOutput) -> EvaluatorOutput:
        """バッチデータを用いて評価結果を返す"""
        feature_vector = data.feature_vector
        feature_variable = data.feature_variable
        target = data.target_vector

        output_result: GeneratorOutput = self.generator(
            feature_vector, feature_variable
        )
        output = output_result.output

        loss = torch.nn.functional.cross_entropy(output, target)

        with torch.no_grad():
            indexes = torch.argmax(output, dim=1)
            correct = torch.eq(indexes, target).view(-1)
            accuracy = correct.float().mean()

        return EvaluatorOutput(
            loss=loss,
            accuracy=accuracy,
            data_num=feature_vector.shape[0],
        )
