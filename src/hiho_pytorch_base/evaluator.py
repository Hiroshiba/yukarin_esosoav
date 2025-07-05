from typing import Literal, TypedDict

import torch
from torch import Tensor, nn

from hiho_pytorch_base.dataset import DatasetOutput
from hiho_pytorch_base.generator import Generator, GeneratorOutput


class EvaluatorOutput(TypedDict):
    value: Tensor
    loss: Tensor
    accuracy: Tensor
    data_num: int


class Evaluator(nn.Module):
    judge: Literal["min", "max"] = "min"

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    def forward(self, data: DatasetOutput) -> EvaluatorOutput:
        feature = torch.stack(data["feature"])
        target = torch.stack(data["target"])

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
