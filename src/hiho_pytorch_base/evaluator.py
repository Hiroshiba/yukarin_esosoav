"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from .batch import BatchOutput
from .generator import Generator, GeneratorOutput
from .utility.audio_utility import log_mel_spectrogram
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    spec_loss: Tensor
    """音響特徴量の損失"""

    wave_spec_loss: Tensor
    """波形のメルスペクトログラム損失"""

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.spec_loss = detach_cpu(self.spec_loss)
        self.wave_spec_loss = detach_cpu(self.wave_spec_loss)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return -1 * output.wave_spec_loss


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        output_list: list[GeneratorOutput] = self.generator(
            f0_list=batch.f0_list,
            phoneme_list=batch.phoneme_list,
            speaker_id=batch.speaker_id,
        )

        spec_all = torch.cat([o.spec for o in output_list], dim=0)  # (sum(fL), ?)
        target_spec_all = torch.cat(batch.spec_list, dim=0)  # (sum(fL), ?)

        spec_loss = l1_loss(spec_all, target_spec_all)

        num_mels = target_spec_all.size(-1)
        frame_size = self.generator.predictor.frame_size
        sampling_rate = self.generator.predictor.sampling_rate

        pred_wave_spec_list = [
            log_mel_spectrogram(
                output.wave.unsqueeze(0),
                frame_size=frame_size,
                spec_size=num_mels,
                sampling_rate=sampling_rate,
            )
            .squeeze(0)
            .transpose(0, 1)
            for output in output_list
        ]

        pred_wave_spec = torch.cat(pred_wave_spec_list, dim=0)  # (sum(fL), ?)
        target_wave_spec = torch.cat(batch.spec_list, dim=0)  # (sum(fL), ?)
        wave_spec_loss = l1_loss(pred_wave_spec, target_wave_spec)

        return EvaluatorOutput(
            spec_loss=spec_loss,
            wave_spec_loss=wave_spec_loss,
            data_num=batch.data_num,
        )
