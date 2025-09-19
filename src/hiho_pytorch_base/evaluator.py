"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.generator import Generator, GeneratorOutput
from hiho_pytorch_base.utility.pytorch_utility import detach_cpu
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    value: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.value = detach_cpu(self.value)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return -1 * output.value


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

        pred_spec = torch.cat([o.spec for o in output_list], dim=0)  # (sum(L), ?)
        target_spec = torch.cat(batch.spec_list, dim=0)  # (sum(L), ?)

        value = l1_loss(pred_spec, target_spec)

        return EvaluatorOutput(
            value=value,
            data_num=batch.data_num,
        )
