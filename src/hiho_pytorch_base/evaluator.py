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
    accuracy: Tensor


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

        # ターゲットとして母音F0の平均を使用
        target_f0 = batch.vowel_f0_means.mean(dim=1)  # (B, V) -> (B,)

        output_result: GeneratorOutput = self.generator(
            lab_phoneme_ids=batch.lab_phoneme_ids,
            lab_durations=batch.lab_durations,
            f0_data=batch.f0_data,
            volume_data=batch.volume_data,
            speaker_id=batch.speaker_id,
        )
        predicted_f0 = output_result.f0_output  # (B,)

        # MSE損失を計算
        loss = mse_loss(predicted_f0, target_f0)

        # 適当な精度指標（相対誤差の逆数）
        relative_error = torch.abs(predicted_f0 - target_f0) / (target_f0 + 1e-8)
        accuracy = 1.0 / (1.0 + torch.mean(relative_error))

        return EvaluatorOutput(
            loss=loss,
            accuracy=accuracy,
            data_num=batch.data_num,
        )
