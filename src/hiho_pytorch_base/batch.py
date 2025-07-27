"""バッチ処理モジュール"""

# TODO: まだかなり途中

from dataclasses import dataclass

import torch
from torch import Tensor

from hiho_pytorch_base.data.data import OutputData


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    lab_phoneme_ids: Tensor  # (B, L)
    lab_durations: Tensor  # (B, L)
    f0_data: Tensor  # (B, T)
    volume_data: Tensor  # (B, T)
    vowel_f0_means: Tensor  # (B, vL)
    vowel_voiced: Tensor  # (B, vL)
    speaker_id: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.lab_phoneme_ids.shape[0]


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする（可変長の場合はパディング）"""
    # TODO: ここはまだ未完成
    if not values:
        raise ValueError("Empty tensor list")

    # 全て同じ形状の場合は通常のstack
    shapes = [v.shape for v in values]
    if all(s == shapes[0] for s in shapes):
        return torch.stack(values)

    # 可変長の場合はパディング
    max_len = max(v.shape[0] for v in values)
    padded_values = []
    for v in values:
        if len(v.shape) == 1:
            # 1D tensor (音素ID等)
            pad_size = max_len - v.shape[0]
            padded = torch.cat([v, torch.zeros(pad_size, dtype=v.dtype)])
        else:
            # 多次元tensorの場合は0次元でパディング
            pad_size = max_len - v.shape[0]
            pad_shape = (pad_size,) + v.shape[1:]
            padded = torch.cat([v, torch.zeros(pad_shape, dtype=v.dtype)])
        padded_values.append(padded)

    return torch.stack(padded_values)


def collate_list(values: list[Tensor]) -> list[Tensor]:
    """Tensorのリストをそのまま返す"""
    return values  # TODO: ここでpadする？datasetからmaskを受け取る形で


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    return BatchOutput(
        lab_phoneme_ids=collate_stack([d.phoneme_id for d in data_list]),
        lab_durations=collate_stack([d.phoneme_duration for d in data_list]),
        f0_data=collate_stack([d.f0 for d in data_list]),
        volume_data=collate_stack([d.volume for d in data_list]),
        vowel_f0_means=collate_stack([d.vowel_f0_means for d in data_list]),
        vowel_voiced=collate_stack([d.vowel_voiced for d in data_list]),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
    )
