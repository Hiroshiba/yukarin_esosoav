"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.network.discriminator import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from hiho_pytorch_base.network.predictor import Predictor
from hiho_pytorch_base.utility.audio_utility import log_mel_spectrogram
from hiho_pytorch_base.utility.pytorch_utility import detach_cpu
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


def _slice_specs_for_wave(
    spec_list: list[Tensor],
    wave_start_frame: Tensor,
    framed_wave_list: list[Tensor],
) -> list[Tensor]:
    start_list = wave_start_frame.detach().cpu().tolist()
    return [
        spec[start : start + framed_wave.size(0)]
        for spec, start, framed_wave in zip(
            spec_list, start_list, framed_wave_list, strict=True
        )
    ]


def _log_mel_list(
    wave_list: list[Tensor],
    *,
    frame_size: int,
    spec_size: int,
    sampling_rate: int,
) -> list[Tensor]:
    return [
        log_mel_spectrogram(
            wave.unsqueeze(0),
            frame_size=frame_size,
            spec_size=spec_size,
            sampling_rate=sampling_rate,
        )
        .squeeze(0)
        .transpose(0, 1)
        for wave in wave_list
    ]


def _feature_loss(
    fmap_real: list[list[Tensor]], fmap_generated: list[list[Tensor]]
) -> Tensor:
    loss = fmap_real[0][0].new_zeros(())
    for real_layers, generated_layers in zip(fmap_real, fmap_generated, strict=True):
        for real_feature, generated_feature in zip(
            real_layers, generated_layers, strict=True
        ):
            loss = loss + torch.mean(torch.abs(real_feature - generated_feature))
    return loss * 2.0


def _generator_loss(disc_generated_outputs: list[Tensor]) -> Tensor:
    loss = disc_generated_outputs[0].new_zeros(())
    for generated_output in disc_generated_outputs:
        loss = loss + torch.mean((1.0 - generated_output) ** 2)
    return loss


def _discriminator_loss(
    disc_real_outputs: list[Tensor], disc_generated_outputs: list[Tensor]
) -> Tensor:
    loss = disc_real_outputs[0].new_zeros(())
    for real_output, generated_output in zip(
        disc_real_outputs, disc_generated_outputs, strict=True
    ):
        loss = (
            loss
            + torch.mean((1.0 - real_output) ** 2)
            + torch.mean(generated_output**2)
        )
    return loss


@dataclass
class GeneratorModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    spec_loss1: Tensor
    """PostNet前の損失"""

    spec_loss2: Tensor
    """PostNet後の損失"""

    wave_spec_loss: Tensor
    """波形のメルスペクトログラム損失"""

    adversarial_loss: Tensor
    """GANのGenerator損失"""

    feature_matching_loss: Tensor
    """Feature Matching損失"""

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.spec_loss1 = detach_cpu(self.spec_loss1)
        self.spec_loss2 = detach_cpu(self.spec_loss2)
        self.wave_spec_loss = detach_cpu(self.wave_spec_loss)
        self.adversarial_loss = detach_cpu(self.adversarial_loss)
        self.feature_matching_loss = detach_cpu(self.feature_matching_loss)
        return self


@dataclass
class DiscriminatorModelOutput(DataNumProtocol):
    """Discriminatorの出力損失"""

    loss: Tensor
    """逆伝播させる損失"""

    mpd_loss: Tensor
    """Multi-Period Discriminatorの損失"""

    msd_loss: Tensor
    """Multi-Scale Discriminatorの損失"""

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.mpd_loss = detach_cpu(self.mpd_loss)
        self.msd_loss = detach_cpu(self.msd_loss)
        return self


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(
        self,
        model_config: ModelConfig,
        predictor: Predictor,
        *,
        mpd: MultiPeriodDiscriminator,
        msd: MultiScaleDiscriminator,
    ):
        super().__init__()
        self.model_config = model_config
        self.acoustic_predictor = predictor.acoustic_predictor
        self.vocoder = predictor.vocoder
        self.mpd = mpd
        self.msd = msd

        self.sampling_rate = predictor.sampling_rate
        self.frame_size = predictor.frame_size

    def forward(
        self, batch: BatchOutput
    ) -> tuple[
        list[Tensor],  # [(fL, ?)]
        list[Tensor],  # [(fL, ?)]
        list[Tensor],  # [(wL,)]
    ]:
        """データをネットワークに入力して損失などを計算する"""
        (
            spec1_list,  # [(fL, ?)]
            spec2_list,  # [(fL, ?)]
        ) = self.acoustic_predictor.forward_list(
            f0_list=batch.f0_list,
            phoneme_list=batch.phoneme_list,
            speaker_id=batch.speaker_id,
        )
        pred_segment_spec_list = _slice_specs_for_wave(
            spec_list=spec2_list,
            wave_start_frame=batch.wave_start_frame,
            framed_wave_list=batch.framed_wave_list,
        )
        pred_wave_list = self.vocoder.forward_list(pred_segment_spec_list)
        return spec1_list, spec2_list, pred_wave_list

    def calc_generator(
        self,
        batch: BatchOutput,
        *,
        spec1_list: list[Tensor],
        spec2_list: list[Tensor],
        pred_wave_list: list[Tensor],
    ) -> GeneratorModelOutput:
        """Generatorの損失を計算する"""
        # 音響特徴量損失
        spec1_all = torch.cat(spec1_list, dim=0)  # (sum(fL), ?)
        spec2_all = torch.cat(spec2_list, dim=0)  # (sum(fL), ?)
        target_spec_all = torch.cat(batch.spec_list, dim=0)  # (sum(fL), ?)

        spec_loss1 = l1_loss(spec1_all, target_spec_all)
        spec_loss2 = l1_loss(spec2_all, target_spec_all)

        # 波形損失
        target_segment_spec_list = _slice_specs_for_wave(
            spec_list=batch.spec_list,
            wave_start_frame=batch.wave_start_frame,
            framed_wave_list=batch.framed_wave_list,
        )
        num_mels = target_spec_all.size(-1)
        pred_wave_spec = torch.cat(
            _log_mel_list(
                pred_wave_list,
                frame_size=self.frame_size,
                spec_size=num_mels,
                sampling_rate=self.sampling_rate,
            ),
            dim=0,
        )
        target_wave_spec = torch.cat(target_segment_spec_list, dim=0)
        wave_spec_loss = l1_loss(pred_wave_spec, target_wave_spec)

        # 判別器損失
        target_wave = torch.stack(
            [frames.reshape(-1) for frames in batch.framed_wave_list], dim=0
        ).unsqueeze(1)  # (B, 1, wL)
        pred_wave = torch.stack(pred_wave_list, dim=0).unsqueeze(1)  # (B, 1, wL)
        _, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = self.mpd(target_wave, pred_wave)
        _, y_d_gs_msd, fmap_rs_msd, fmap_gs_msd = self.msd(target_wave, pred_wave)

        adversarial_loss = _generator_loss(y_d_gs_mpd) + _generator_loss(y_d_gs_msd)
        feature_matching_loss = _feature_loss(fmap_rs_mpd, fmap_gs_mpd) + _feature_loss(
            fmap_rs_msd, fmap_gs_msd
        )

        loss = (
            self.model_config.acoustic_loss1_weight * spec_loss1
            + self.model_config.acoustic_loss2_weight * spec_loss2
            + self.model_config.vocoder_spec_loss_weight * wave_spec_loss
            + self.model_config.vocoder_adv_loss_weight * adversarial_loss
            + self.model_config.vocoder_fm_loss_weight * feature_matching_loss
        )

        return GeneratorModelOutput(
            loss=loss,
            spec_loss1=spec_loss1,
            spec_loss2=spec_loss2,
            wave_spec_loss=wave_spec_loss,
            adversarial_loss=adversarial_loss,
            feature_matching_loss=feature_matching_loss,
            data_num=batch.data_num,
        )

    def calc_discriminator(
        self,
        batch: BatchOutput,
        *,
        pred_wave_list: list[Tensor],
    ) -> DiscriminatorModelOutput:
        """Discriminator用の損失を計算する"""
        # 判別器の計算
        target_wave = torch.stack(
            [frames.reshape(-1) for frames in batch.framed_wave_list], dim=0
        ).unsqueeze(1)  # (B, 1, wL)
        pred_wave = (
            torch.stack(pred_wave_list, dim=0).unsqueeze(1).detach()
        )  # (B, 1, wL)
        y_d_rs_mpd, y_d_gs_mpd, _, _ = self.mpd(target_wave, pred_wave)
        y_d_rs_msd, y_d_gs_msd, _, _ = self.msd(target_wave, pred_wave)

        mpd_loss = _discriminator_loss(y_d_rs_mpd, y_d_gs_mpd)
        msd_loss = _discriminator_loss(y_d_rs_msd, y_d_gs_msd)

        loss = mpd_loss + msd_loss

        return DiscriminatorModelOutput(
            loss=loss,
            mpd_loss=mpd_loss,
            msd_loss=msd_loss,
            data_num=batch.data_num,
        )
