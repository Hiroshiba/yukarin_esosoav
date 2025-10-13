"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.network.predictor import Predictor
from hiho_pytorch_base.utility.audio_utility import log_mel_spectrogram
from hiho_pytorch_base.utility.pytorch_utility import detach_cpu
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    spec_loss1: Tensor
    """PostNet前の音響特徴量の損失"""

    spec_loss2: Tensor
    """PostNet後の音響特徴量の損失"""

    wave_spec_loss: Tensor
    """波形のメルスペクトログラム損失"""

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.spec_loss1 = detach_cpu(self.spec_loss1)
        self.spec_loss2 = detach_cpu(self.spec_loss2)
        self.wave_spec_loss = detach_cpu(self.wave_spec_loss)
        return self


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.acoustic_predictor = predictor.acoustic_predictor
        self.vocoder = predictor.vocoder
        self.sampling_rate = predictor.sampling_rate
        self.frame_size = predictor.frame_size

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        # 音響特徴量の計算
        (
            spec1_list,  # [(fL, ?)]
            spec2_list,  # [(fL, ?)]
        ) = self.acoustic_predictor.forward_list(
            f0_list=batch.f0_list,
            phoneme_list=batch.phoneme_list,
            speaker_id=batch.speaker_id,
        )

        spec1_all = torch.cat(spec1_list, dim=0)  # (sum(fL), ?)
        spec2_all = torch.cat(spec2_list, dim=0)  # (sum(fL), ?)
        target_spec_all = torch.cat(batch.spec_list, dim=0)  # (sum(fL), ?)

        spec_loss1 = l1_loss(spec1_all, target_spec_all)
        spec_loss2 = l1_loss(spec2_all, target_spec_all)

        # 音響特徴量の切り出し
        segmented_spec_list = [
            spec[start_frame : start_frame + framed_wave.shape[0]]
            for spec, start_frame, framed_wave in zip(
                spec2_list,
                batch.wave_start_frame,
                batch.framed_wave_list,
                strict=True,
            )
        ]

        # 波形の計算
        pred_wave_list = self.vocoder.forward_list(segmented_spec_list)  # [(wL,)]

        num_mels = target_spec_all.size(-1)
        pred_wave_spec_list = [
            log_mel_spectrogram(
                pred_wave.unsqueeze(0),
                frame_size=self.frame_size,
                spec_size=num_mels,
                sampling_rate=self.sampling_rate,
            )
            .squeeze(0)
            .transpose(0, 1)
            for pred_wave in pred_wave_list
        ]
        target_wave_spec_list = [
            spec[start_frame : start_frame + framed_wave.shape[0]]
            for spec, start_frame, framed_wave in zip(
                batch.spec_list,
                batch.wave_start_frame,
                batch.framed_wave_list,
                strict=True,
            )
        ]

        pred_wave_spec = torch.cat(pred_wave_spec_list, dim=0)  # (sum(fwL), ?)
        target_wave_spec = torch.cat(target_wave_spec_list, dim=0)  # (sum(fwL), ?)
        wave_spec_loss = l1_loss(pred_wave_spec, target_wave_spec)

        loss = (
            self.model_config.acoustic_loss1_weight * spec_loss1
            + self.model_config.acoustic_loss2_weight * spec_loss2
            + self.model_config.vocoder_spec_loss_weight * wave_spec_loss
        )

        return ModelOutput(
            loss=loss,
            spec_loss1=spec_loss1,
            spec_loss2=spec_loss2,
            wave_spec_loss=wave_spec_loss,
            data_num=batch.data_num,
        )
