"""評価値計算モジュール"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.generator import Generator, GeneratorOutput
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    loss: Tensor
    vuv_accuracy: Tensor


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return -1 * output.loss


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        # TODO: 適当な実装なので変更する

        output_result: GeneratorOutput = self.generator(
            phoneme_ids_list=batch.phoneme_ids_list,
            phoneme_durations_list=batch.phoneme_durations_list,
            phoneme_stress_list=batch.phoneme_stress_list,
            vowel_index_list=batch.vowel_index_list,
            speaker_id=batch.speaker_id,
        )

        # 予測結果とターゲットを結合して一括計算
        predicted_f0_all = torch.cat(output_result.f0, dim=0)  # (sum(vL),)
        predicted_vuv_all = torch.cat(output_result.vuv, dim=0)  # (sum(vL),)
        target_f0_all = torch.cat(batch.vowel_f0_means_list, dim=0)  # (sum(vL),)
        target_vuv_all = torch.cat(batch.vowel_voiced_list, dim=0)  # (sum(vL),)

        # MSE損失を計算
        loss = mse_loss(predicted_f0_all, target_f0_all)

        # 有声かどうかの精度
        predicted_vuv_binary = predicted_vuv_all > 0.0
        vuv_accuracy = (predicted_vuv_binary == target_vuv_all).float().mean()

        return EvaluatorOutput(
            loss=loss,
            vuv_accuracy=vuv_accuracy,
            data_num=batch.data_num,
        )
