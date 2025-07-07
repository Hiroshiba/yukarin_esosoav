"""モデル評価モジュール"""

from typing import Literal, TypedDict

import torch
from torch import Tensor, nn

from hiho_pytorch_base.dataset import BatchOutput
from hiho_pytorch_base.generator import Generator, GeneratorOutput


class EvaluatorOutput(TypedDict):
    """評価結果の出力型定義"""

    value: Tensor
    loss: Tensor
    accuracy: Tensor
    data_num: int


class Evaluator(nn.Module):
    """モデルの評価を行うクラス"""

    judge: Literal["min", "max"] = "min"

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    def forward(self, data: BatchOutput) -> EvaluatorOutput:
        """バッチデータを用いて評価結果を返す"""
        feature = data.feature_vector
        target = data.target_vector

        output_result: GeneratorOutput = self.generator(feature)
        output = output_result.output

        loss = torch.nn.functional.cross_entropy(output, target)

        with torch.no_grad():
            indexes = torch.argmax(output, dim=1)
            correct = torch.eq(indexes, target).view(-1)
            accuracy = correct.float().mean()

        return EvaluatorOutput(
            value=loss,
            loss=loss,
            accuracy=accuracy,
            data_num=feature.shape[0],
        )
