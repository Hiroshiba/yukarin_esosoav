"""学習済みモデルからの推論モジュール"""

from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from torch import Tensor, nn

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.network.predictor import Predictor, create_predictor

TensorLike = Tensor | numpy.ndarray


@dataclass
class GeneratorOutput:
    """生成したデータ"""

    spec: Tensor  # (fL, ?)
    wave: Tensor  # (wL,)


def to_tensor(array: TensorLike, device: torch.device) -> Tensor:
    """データをTensorに変換する"""
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    tensor = tensor.to(device)
    return tensor


class Generator(nn.Module):
    """生成経路で推論するクラス"""

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

    @torch.no_grad()
    def forward(
        self,
        *,
        f0_list: list[TensorLike],  # [(L,)]
        phoneme_list: list[TensorLike],  # [(L,)]
        speaker_id: TensorLike,  # (B,)
    ) -> list[GeneratorOutput]:
        """生成経路で推論する"""
        f0_list_t = [to_tensor(x, self.device) for x in f0_list]
        phoneme_list_t = [to_tensor(x, self.device) for x in phoneme_list]
        speaker_id_t = to_tensor(speaker_id, self.device)

        _, post_list, wave_list = self.predictor(
            f0_list=f0_list_t,
            phoneme_list=phoneme_list_t,
            speaker_id=speaker_id_t,
        )

        return [
            GeneratorOutput(spec=spec, wave=wave)
            for spec, wave in zip(post_list, wave_list, strict=True)
        ]
