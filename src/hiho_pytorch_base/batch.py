"""バッチ処理モジュール"""

# TODO: まだかなり途中

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from hiho_pytorch_base.data.data import OutputData
from hiho_pytorch_base.utility.pytorch_utility import to_device


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    f0_list: list[Tensor]  # [(fL,)]
    phoneme_list: list[Tensor]  # [(fL,)]
    spec_list: list[Tensor]  # [(fL, ?)]
    framed_wave_list: list[Tensor]  # [(wfL, ?)]
    wave_start_frame: Tensor  # (B,)
    speaker_id: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.speaker_id.shape[0]

    def to_device(self, device: str, non_blocking: bool) -> Self:
        """データを指定されたデバイスに移動"""
        self.f0_list = to_device(self.f0_list, device, non_blocking=non_blocking)
        self.phoneme_list = to_device(
            self.phoneme_list, device, non_blocking=non_blocking
        )
        self.spec_list = to_device(self.spec_list, device, non_blocking=non_blocking)
        self.framed_wave_list = to_device(
            self.framed_wave_list, device, non_blocking=non_blocking
        )
        self.wave_start_frame = to_device(
            self.wave_start_frame, device, non_blocking=non_blocking
        )
        self.speaker_id = to_device(self.speaker_id, device, non_blocking=non_blocking)
        return self


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    return BatchOutput(
        f0_list=[d.f0 for d in data_list],
        phoneme_list=[d.phoneme for d in data_list],
        spec_list=[d.spec for d in data_list],
        framed_wave_list=[d.framed_wave for d in data_list],
        wave_start_frame=collate_stack([d.wave_start_frame for d in data_list]),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
    )
