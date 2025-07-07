"""モデル定義モジュール"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, mse_loss

from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.dataset import BatchOutput
from hiho_pytorch_base.network.predictor import Predictor
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。Lossと、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    loss_vector: Tensor
    loss_scalar: Tensor
    accuracy: Tensor


def accuracy(output: Tensor, target: Tensor) -> Tensor:
    """分類精度を計算"""
    with torch.no_grad():
        indexes = torch.argmax(output, dim=1)
        correct = torch.eq(indexes, target).view(-1)
        return correct.float().mean()


class Model(nn.Module):
    """学習用のモデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, data: BatchOutput) -> ModelOutput:
        """ネットワークに入力して損失などを計算する"""
        batch_size = data.feature_vector.shape[0]

        vector_output, scalar_output = self.predictor(
            feature_vector=data.feature_vector, feature_variable=data.feature_variable
        )

        target_vector = data.target_vector
        target_scalar = data.target_scalar

        loss_vector = cross_entropy(vector_output, target_vector)
        loss_scalar = mse_loss(scalar_output, target_scalar)
        total_loss = loss_vector + loss_scalar
        acc = accuracy(vector_output, target_vector)

        return ModelOutput(
            loss=total_loss,
            loss_vector=loss_vector,
            loss_scalar=loss_scalar,
            accuracy=acc,
            data_num=batch_size,
        )
