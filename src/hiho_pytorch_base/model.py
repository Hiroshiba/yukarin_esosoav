from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy

from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.dataset import DatasetOutput
from hiho_pytorch_base.network.predictor import Predictor


class ModelOutput:
    def __init__(self, loss: Tensor, accuracy: Tensor, data_num: int):
        self.loss = loss
        self.accuracy = accuracy
        self.data_num = data_num


def reduce_result(results):
    result = {}
    sum_data_num = sum([r.data_num for r in results])
    for key in ["loss", "accuracy"]:
        values = [getattr(r, key) * r.data_num for r in results]
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
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, data: DatasetOutput) -> ModelOutput:
        feature = data["feature"]
        target = data["target"]
        
        output = self.predictor(feature)
        loss = cross_entropy(output, target)
        acc = accuracy(output, target)
        
        return ModelOutput(
            loss=loss,
            accuracy=acc,
            data_num=feature.shape[0],
        )
