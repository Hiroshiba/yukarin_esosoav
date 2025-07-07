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

    def forward(self, feature_vector, feature_variable):
        """推論モードでGeneratorOutputを返す"""
        feature_vector = to_tensor(feature_vector).to(self.device)
        feature_variable = [to_tensor(fv).to(self.device) for fv in feature_variable]

        with torch.inference_mode():
            vector_output, scalar_output = self.predictor(
                feature_vector, feature_variable
            )

        return GeneratorOutput(output=vector_output)

    def generate(self, feature_vector, feature_variable):
        """生成モードでnumpy配列を返す"""
        if isinstance(feature_vector, numpy.ndarray):
            feature_vector = torch.from_numpy(feature_vector)
        feature_vector = feature_vector.to(self.device)

        feature_variable = [
            torch.from_numpy(fv) if isinstance(fv, numpy.ndarray) else fv
            for fv in feature_variable
        ]
        feature_variable = [fv.to(self.device) for fv in feature_variable]

        with torch.inference_mode():
            vector_output, scalar_output = self.predictor(
                feature_vector, feature_variable
            )
        return vector_output.cpu().numpy()
