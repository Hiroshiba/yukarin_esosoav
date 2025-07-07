"""モデル定義モジュール"""

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy
from typing_extensions import TypedDict

from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.dataset import BatchOutput


class ModelOutput(TypedDict):
    """モデル出力の型定義"""

    loss: Tensor
    loss_vector: Tensor
    loss_scalar: Tensor
    accuracy: Tensor
    data_num: int


def reduce_result(results: list[ModelOutput]):
    """複数のModelOutput結果をデータ数で重み付けして統計"""
    result: dict[str, Any] = {}
    sum_data_num = sum([r["data_num"] for r in results])
    for key in set(results[0].keys()) - {"data_num"}:
        values = [r[key] * r["data_num"] for r in results]
        if isinstance(values[0], Tensor):
            result[key] = torch.stack(values).sum() / sum_data_num
        else:
            result[key] = sum(values) / sum_data_num
    result["data_num"] = sum_data_num
    return result


def accuracy(output: Tensor, target: Tensor):
    """分類精度を計算"""
    with torch.no_grad():
        indexes = torch.argmax(output, dim=1)
        correct = torch.eq(indexes, target).view(-1)
        return correct.float().mean()


class Model(nn.Module):
    """マルチタスク学習対応のメインモデル"""

    def __init__(self, model_config: ModelConfig, predictor: nn.Module):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

        # 可変長データ処理用の線形層
        self.variable_processor = nn.Linear(
            32, predictor.input_size
        )  # feature_variableの次元を合わせる

        # スカラー出力用の追加ヘッド
        self.scalar_head = nn.Linear(
            predictor.hidden_size, 1
        )  # hidden_sizeから1次元出力

    def forward(self, data: BatchOutput) -> ModelOutput:
        """順伝播で分類と回帰の両方を予測"""
        feature_vector = data.feature_vector
        feature_variable = data.feature_variable
        target_vector = data.target_vector
        target_scalar = data.target_scalar

        variable_means = []
        for var_data in feature_variable:
            var_mean = torch.mean(var_data, dim=0)
            var_processed = self.variable_processor(var_mean)
            variable_means.append(var_processed)

        variable_features = torch.stack(variable_means)
        combined_features = feature_vector + variable_features

        vector_output = self.predictor(combined_features)
        predictor_hidden = self.predictor.layers[:-1](combined_features)
        scalar_output = self.scalar_head(predictor_hidden).squeeze(-1)

        loss_vector = cross_entropy(vector_output, target_vector)
        loss_scalar = nn.functional.mse_loss(scalar_output, target_scalar)
        total_loss = loss_vector + loss_scalar
        acc = accuracy(vector_output, target_vector)

        return ModelOutput(
            loss=total_loss,
            loss_vector=loss_vector,
            loss_scalar=loss_scalar,
            accuracy=acc,
            data_num=feature_vector.shape[0],
        )
