"""学習済みモデルからの推論モジュール"""

from pathlib import Path

import numpy
import torch
from torch import Tensor, nn

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.network.predictor import create_predictor


class GeneratorOutput:
    """推論結果のデータ構造"""

    def __init__(self, output: Tensor):
        self.output = output


def to_tensor(array):
    """様々な形式のデータをTensorに変換"""
    if not isinstance(array, Tensor | numpy.ndarray):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


class Generator(nn.Module):
    """学習済みモデルからの推論を行うジェネレーター"""

    def __init__(
        self,
        config: Config,
        predictor,
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def forward(self, feature):
        """推論モードでGeneratorOutputを返す"""
        feature = to_tensor(feature).to(self.device)

        with torch.inference_mode():
            output = self.predictor(feature)

        return GeneratorOutput(output=output)

    def generate(self, feature):
        """生成モードでnumpy配列を返す"""
        if isinstance(feature, numpy.ndarray):
            feature = torch.from_numpy(feature)
        feature = feature.to(self.device)

        with torch.inference_mode():
            output = self.predictor(feature)
        return output.cpu().numpy()
