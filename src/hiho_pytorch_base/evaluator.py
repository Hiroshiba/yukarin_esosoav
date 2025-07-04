import torch
from torch import Tensor, nn

from hiho_pytorch_base.dataset import DatasetOutput
from hiho_pytorch_base.generator import Generator, GeneratorOutput


class EvaluatorOutput:
    def __init__(self, loss: Tensor, accuracy: Tensor, data_num: int):
        self.loss = loss
        self.accuracy = accuracy
        self.data_num = data_num


class Evaluator(nn.Module):
    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    def forward(self, data: DatasetOutput) -> EvaluatorOutput:
        feature = data["feature"]
        target = data["target"]
        
        output_result: GeneratorOutput = self.generator(feature)
        output = output_result.output
        
        loss = torch.nn.functional.cross_entropy(output, target)
        
        with torch.no_grad():
            indexes = torch.argmax(output, dim=1)
            correct = torch.eq(indexes, target).view(-1)
            accuracy = correct.float().mean()
        
        return EvaluatorOutput(
            loss=loss,
            accuracy=accuracy,
            data_num=feature.shape[0],
        )