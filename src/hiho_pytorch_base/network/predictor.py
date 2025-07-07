"""予測器ネットワークの実装"""

import torch
from torch import Tensor, nn

from hiho_pytorch_base.config import NetworkConfig


class Predictor(nn.Module):
    """マルチタスク学習対応の予測器"""

    def __init__(
        self,
        feature_vector_size: int,
        feature_variable_size: int,
        hidden_size: int,
        target_vector_size: int,
    ):
        super().__init__()

        # 可変長データ処理用の線形層
        self.variable_processor = nn.Linear(feature_variable_size, feature_vector_size)

        # メイン特徴量処理
        self.main_layers = nn.Sequential(
            nn.Linear(feature_vector_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # ベクトル出力用ヘッド
        self.vector_head = nn.Linear(hidden_size, target_vector_size)

        # スカラー出力用ヘッド
        self.scalar_head = nn.Linear(hidden_size, 1)

    def forward(
        self, feature_vector: Tensor, feature_variable: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """バッチデータを処理してベクトルとスカラーの両方を予測"""
        # 可変長データの平均化処理
        variable_means = []
        for var_data in feature_variable:
            var_mean = torch.mean(var_data, dim=0)
            var_processed = self.variable_processor(var_mean)
            variable_means.append(var_processed)

        # 特徴量の結合
        variable_features = torch.stack(variable_means)
        combined_features = feature_vector + variable_features

        # メイン処理
        hidden = self.main_layers(combined_features)

        # 各出力の生成
        vector_output = self.vector_head(hidden)
        scalar_output = self.scalar_head(hidden).squeeze(-1)

        return vector_output, scalar_output


def create_predictor(config: NetworkConfig) -> Predictor:
    """ネットワーク設定からPredictorを作成"""
    return Predictor(
        feature_vector_size=config.feature_vector_size,
        feature_variable_size=config.feature_variable_size,
        hidden_size=config.hidden_size,
        target_vector_size=config.target_vector_size,
    )
