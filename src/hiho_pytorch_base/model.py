from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy
from typing_extensions import TypedDict

from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.dataset import DatasetOutput


class ModelOutput(TypedDict):
    loss: Tensor
    accuracy: Tensor
    data_num: int


def reduce_result(results: list[ModelOutput]):
    result: dict[str, Any] = {}
    sum_data_num = sum([r["data_num"] for r in results])
    for key in set(results[0].keys()) - {"data_num"}:
        values = [r[key] * r["data_num"] for r in results]
        if isinstance(values[0], Tensor):
            result[key] = torch.stack(values).sum() / sum_data_num
        else:
            result[key] = sum(values) / sum_data_num
    return result


def accuracy(output: Tensor, target: Tensor):
    with torch.no_grad():
        indexes = torch.argmax(output, dim=1)
        correct = torch.eq(indexes, target).view(-1)
        return correct.float().mean()


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: nn.Module):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, data: DatasetOutput) -> ModelOutput:
        feature = torch.stack(data["feature"])
        target = torch.stack(data["target"])

        output = self.predictor(feature)
        loss = cross_entropy(output, target)
        acc = accuracy(output, target)

        return ModelOutput(
            loss=loss,
            accuracy=acc,
            data_num=feature.shape[0],
        )
