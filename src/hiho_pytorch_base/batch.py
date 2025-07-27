"""バッチ処理モジュール"""

# TODO: まだかなり途中

from dataclasses import dataclass

import torch
from torch import Tensor

from hiho_pytorch_base.data.data import OutputData


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    phoneme_ids_list: list[Tensor]  # [(L,)]
    phoneme_durations_list: list[Tensor]  # [(L,)]
    phoneme_stress_list: list[Tensor]  # [(L,)]
    f0_data_list: list[Tensor]  # [(T,)]
    volume_data_list: list[Tensor]  # [(T,)]
    vowel_f0_means_list: list[Tensor]  # [(vL,)]
    vowel_voiced_list: list[Tensor]  # [(vL,)]
    vowel_index_list: list[Tensor]  # [(vL,)]
    speaker_id: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.speaker_id.shape[0]


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_list(values: list[Tensor]) -> list[Tensor]:
    """Tensorのリストをそのまま返す"""
    return values  # TODO: ここでpadする？datasetからmaskを受け取る形で


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    return BatchOutput(
        phoneme_ids_list=collate_list([d.phoneme_id for d in data_list]),
        phoneme_durations_list=collate_list([d.phoneme_duration for d in data_list]),
        phoneme_stress_list=collate_list([d.phoneme_stress for d in data_list]),
        f0_data_list=collate_list([d.f0 for d in data_list]),
        volume_data_list=collate_list([d.volume for d in data_list]),
        vowel_f0_means_list=collate_list([d.vowel_f0_means for d in data_list]),
        vowel_voiced_list=collate_list([d.vowel_voiced for d in data_list]),
        vowel_index_list=collate_list([d.vowel_index for d in data_list]),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
    )
