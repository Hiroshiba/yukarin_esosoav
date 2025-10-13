"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self

import librosa
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.network.predictor import Predictor
from hiho_pytorch_base.utility.pytorch_utility import detach_cpu
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    spec_loss1: Tensor
    """PostNet前の損失"""

    spec_loss2: Tensor
    """PostNet後の損失"""

    wave_spec_loss: Tensor
    """波形のメルスペクトログラム損失"""

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.spec_loss1 = detach_cpu(self.spec_loss1)
        self.spec_loss2 = detach_cpu(self.spec_loss2)
        self.wave_spec_loss = detach_cpu(self.wave_spec_loss)
        return self


_mel_basis_cache: dict[
    tuple[torch.device, torch.dtype, int, int, int, float, float], Tensor
] = {}
_hann_window_cache: dict[tuple[torch.device, torch.dtype, int], Tensor] = {}


def _next_power_of_two(value: int) -> int:
    """value以上の最小の2の冪を返す"""
    power = 1
    while power < value:
        power <<= 1
    return power


def log_mel_spectrogram(
    wave: Tensor,
    *,
    frame_size: int,
    spec_size: int,
    sampling_rate: int | None = None,
    mel_fmin: float = 0.0,
    mel_fmax: float | None = None,
) -> Tensor:
    """波形から対数メルスペクトログラム（log spec）を計算する"""
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    elif wave.dim() != 2:
        raise ValueError("waveは(B, L)または(L,)である必要があります")

    if sampling_rate is None:
        sampling_rate = frame_size * 100
    if mel_fmax is None:
        mel_fmax = sampling_rate / 2

    hop_size = frame_size
    win_size = frame_size * 4
    n_fft = _next_power_of_two(win_size)
    win_size = min(win_size, n_fft)

    device = wave.device
    dtype = wave.dtype
    mel_key = (device, dtype, n_fft, spec_size, sampling_rate, mel_fmin, mel_fmax)
    hann_key = (device, dtype, win_size)

    if mel_key not in _mel_basis_cache:
        mel_basis = torch.from_numpy(
            librosa.filters.mel(
                sr=sampling_rate,
                n_fft=n_fft,
                n_mels=spec_size,
                fmin=mel_fmin,
                fmax=mel_fmax,
            )
        )
        _mel_basis_cache[mel_key] = mel_basis.to(device=device, dtype=dtype)

    if hann_key not in _hann_window_cache:
        _hann_window_cache[hann_key] = torch.hann_window(win_size).to(
            device=device, dtype=dtype
        )

    mel_basis = _mel_basis_cache[mel_key]
    hann_window = _hann_window_cache[hann_key]

    pad = int((n_fft - hop_size) // 2)
    wave = F.pad(wave.unsqueeze(1), (pad, pad), mode="reflect")
    wave = wave.squeeze(1)

    spec = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec_magnitude = torch.sqrt(torch.clamp(spec.pow(2).sum(-1), min=1e-9))
    mel_spec = torch.matmul(mel_basis.unsqueeze(0), spec_magnitude)
    return torch.log(torch.clamp(mel_spec, min=1e-5))


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        (
            output1_list,  # [(fL, ?)]
            output2_list,  # [(fL, ?)]
            wave_list,  # [(wL,)]
        ) = self.predictor(
            f0_list=batch.f0_list,
            phoneme_list=batch.phoneme_list,
            speaker_id=batch.speaker_id,
        )

        pred1_all = torch.cat(output1_list, dim=0)  # (sum(fL), ?)
        pred2_all = torch.cat(output2_list, dim=0)  # (sum(fL), ?)
        target_spec_all = torch.cat(batch.spec_list, dim=0)  # (sum(fL), ?)

        spec_loss1 = l1_loss(pred1_all, target_spec_all)
        spec_loss2 = l1_loss(pred2_all, target_spec_all)

        frame_size = batch.framed_wave_list[0].shape[-1]
        wave_start = batch.wave_start_frame.long()
        num_mels = target_spec_all.size(-1)

        pred_spec_segments: list[Tensor] = []
        target_spec_segments: list[Tensor] = []

        for i, pred_wave in enumerate(wave_list):
            target_frames = batch.framed_wave_list[i]  # (wfL, ?)
            if target_frames.numel() == 0:
                continue

            target_wave = target_frames.reshape(-1)  # (wfL * ?,)

            start_frame = int(wave_start[i].item())
            frame_length = target_frames.size(0)
            start_sample = start_frame * frame_size
            end_sample = start_sample + frame_length * frame_size
            pred_segment = pred_wave[start_sample:end_sample]

            segment_length = min(pred_segment.numel(), target_wave.numel())
            pred_segment = pred_segment[:segment_length].unsqueeze(0)
            target_segment = target_wave[:segment_length].unsqueeze(0)

            pred_mel = log_mel_spectrogram(
                pred_segment,
                frame_size=frame_size,
                spec_size=num_mels,
                sampling_rate=self.predictor.sampling_rate,
            ).squeeze(0)
            target_mel = log_mel_spectrogram(
                target_segment,
                frame_size=frame_size,
                spec_size=num_mels,
                sampling_rate=self.predictor.sampling_rate,
            ).squeeze(0)

            pred_spec_segments.append(pred_mel.transpose(0, 1))
            target_spec_segments.append(target_mel.transpose(0, 1))

        pred_wave_spec = torch.cat(pred_spec_segments, dim=0)
        target_wave_spec = torch.cat(target_spec_segments, dim=0)
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
