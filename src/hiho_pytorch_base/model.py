"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.network.predictor import Predictor
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    f0_loss: Tensor
    """F0損失"""

    vuv_loss: Tensor
    """vuv損失"""


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        f0_output, vuv_output = self.predictor(
            lab_phoneme_ids=batch.lab_phoneme_ids,
            lab_durations=batch.lab_durations,
            f0_data=batch.f0_data,
            volume_data=batch.volume_data,
            speaker_id=batch.speaker_id,
        )  # (B,), (B,)

        # ターゲットとして母音F0の平均を使用
        target_f0 = batch.vowel_f0_means.mean(dim=1)  # (B, vL) -> (B,)
        target_vuv = batch.vowel_voiced.any(dim=1)  # (B, vL) -> (B,)

        # vuv損失
        vuv_loss = binary_cross_entropy_with_logits(vuv_output, target_vuv.float())

        # F0損失
        voiced_mask = target_vuv  # (B,)
        if voiced_mask.any():
            f0_loss = mse_loss(f0_output[voiced_mask], target_f0[voiced_mask])
        else:
            f0_loss = mse_loss(f0_output, target_f0) * 0.0

        # 全体損失
        loss = f0_loss + vuv_loss

        return ModelOutput(
            loss=loss,
            f0_loss=f0_loss,
            vuv_loss=vuv_loss,
            data_num=batch.data_num,
        )
