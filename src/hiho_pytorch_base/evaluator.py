"""評価値計算モジュール"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.generator import Generator, GeneratorOutput
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    loss: Tensor
    accuracy: Tensor


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return output.accuracy


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        feature_vector = batch.feature_vector
        feature_variable_list = batch.feature_variable_list
        target = batch.target_vector

        output_result: GeneratorOutput = self.generator(
            feature_vector=feature_vector, feature_variable_list=feature_variable_list
        )
        output = output_result.vector_output

        loss = torch.nn.functional.cross_entropy(output, target)

        indexes = torch.argmax(output, dim=1)
        correct = torch.eq(indexes, target).view(-1)
        accuracy = correct.float().mean()

        return EvaluatorOutput(
            loss=loss,
            accuracy=accuracy,
            data_num=batch.data_num,
        )
