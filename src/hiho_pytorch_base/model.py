"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn.functional import mse_loss

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
    """F0予測損失"""


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        f0_output = self.predictor(
            lab_phoneme_ids=batch.lab_phoneme_ids,
            lab_durations=batch.lab_durations,
            f0_data=batch.f0_data,
            volume_data=batch.volume_data,
            speaker_id=batch.speaker_id,
        )  # (B,)

        # ターゲットとして母音F0の平均を使用
        target_f0 = batch.vowel_f0_means.mean(dim=1)  # (B, V) -> (B,)

        # MSE損失を計算
        f0_loss = mse_loss(f0_output, target_f0)

        return ModelOutput(
            loss=f0_loss,
            f0_loss=f0_loss,
            data_num=batch.data_num,
        )
