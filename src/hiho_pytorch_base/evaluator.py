"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.generator import Generator, GeneratorOutput
from hiho_pytorch_base.model import log_mel_spectrogram
from hiho_pytorch_base.utility.pytorch_utility import detach_cpu
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


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

        pred_spec = torch.cat([o.spec for o in output_list], dim=0)  # (sum(fL), ?)
        target_spec = torch.cat(batch.spec_list, dim=0)  # (sum(fL), ?)

        spec_loss = l1_loss(pred_spec, target_spec)

        frame_size = batch.framed_wave_list[0].shape[-1]
        wave_start = batch.wave_start_frame.long()
        num_mels = target_spec.size(-1)
        sampling_rate = self._estimate_sampling_rate(frame_size)

        pred_spec_segments: list[Tensor] = []
        target_spec_segments: list[Tensor] = []

        for i, output in enumerate(output_list):
            target_frames = batch.framed_wave_list[i]  # (wfL, ?)
            if target_frames.numel() == 0:
                continue

            target_wave = target_frames.reshape(-1)

            start_frame = int(wave_start[i].item())
            frame_length = target_frames.size(0)
            start_sample = start_frame * frame_size
            end_sample = start_sample + frame_length * frame_size
            pred_segment = output.wave[start_sample:end_sample]

            segment_length = min(pred_segment.numel(), target_wave.numel())
            pred_segment = pred_segment[:segment_length].unsqueeze(0)
            target_segment = target_wave[:segment_length].unsqueeze(0)

            pred_mel = log_mel_spectrogram(
                pred_segment,
                frame_size=frame_size,
                spec_size=num_mels,
                sampling_rate=sampling_rate,
            ).squeeze(0)
            target_mel = log_mel_spectrogram(
                target_segment,
                frame_size=frame_size,
                spec_size=num_mels,
                sampling_rate=sampling_rate,
            ).squeeze(0)

            pred_spec_segments.append(pred_mel.transpose(0, 1))
            target_spec_segments.append(target_mel.transpose(0, 1))

        pred_wave_spec = torch.cat(pred_spec_segments, dim=0)
        target_wave_spec = torch.cat(target_spec_segments, dim=0)
        wave_spec_loss = l1_loss(pred_wave_spec, target_wave_spec)

        return EvaluatorOutput(
            spec_loss=spec_loss,
            wave_spec_loss=wave_spec_loss,
            data_num=batch.data_num,
        )

    def _estimate_sampling_rate(self, frame_size: int) -> int:
        """推定したサンプリングレートを返す。将来的に設定から取得予定。"""
        return frame_size * 100
