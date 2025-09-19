"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self

import torch
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

    loss1: Tensor
    """PostNet前の損失"""

    loss2: Tensor
    """PostNet後の損失"""

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.loss1 = detach_cpu(self.loss1)
        self.loss2 = detach_cpu(self.loss2)
        return self


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        output1_list, output2_list = self.predictor(
            f0_list=batch.f0_list,
            phoneme_list=batch.phoneme_list,
            speaker_id=batch.speaker_id,
        )  # [(L, ?)], [(L, ?)]

        pred1_all = torch.cat(output1_list, dim=0)  # (sum(L), ?)
        pred2_all = torch.cat(output2_list, dim=0)  # (sum(L), ?)
        target_all = torch.cat(batch.spec_list, dim=0)  # (sum(L), ?)

        loss1 = l1_loss(pred1_all, target_all)
        loss2 = l1_loss(pred2_all, target_all)
        loss = loss1 + loss2

        return ModelOutput(
            loss=loss,
            loss1=loss1,
            loss2=loss2,
            data_num=batch.data_num,
        )
