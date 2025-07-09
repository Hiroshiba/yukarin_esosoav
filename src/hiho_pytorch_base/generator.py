"""学習済みモデルからの推論モジュール"""

from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from torch import Tensor, nn

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.network.predictor import Predictor, create_predictor


@dataclass
class GeneratorOutput:
    """推論結果のデータ構造"""

    vector_output: Tensor
    scalar_output: Tensor


def to_tensor(array: Tensor | numpy.ndarray, device: torch.device) -> Tensor:
    """データをTensorに変換する"""
    if not isinstance(array, Tensor | numpy.ndarray):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    tensor = tensor.to(device)
    return tensor


class Generator(nn.Module):
    """学習済みモデルからの推論を行うジェネレーター"""

    def __init__(
        self,
        config: Config,
        predictor: Predictor | Path,
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

    def forward(
        self,
        feature_vector: Tensor | numpy.ndarray,
        feature_variable_list: list[Tensor | numpy.ndarray],
    ) -> GeneratorOutput:
        """推論モードでGeneratorOutputを返す"""
        feature_vector = to_tensor(feature_vector, self.device)
        feature_variable_list = [
            to_tensor(d, self.device) for d in feature_variable_list
        ]

        with torch.inference_mode():
            vector_output, scalar_output = self.predictor(
                feature_vector, feature_variable_list
            )

        return GeneratorOutput(vector_output=vector_output, scalar_output=scalar_output)
