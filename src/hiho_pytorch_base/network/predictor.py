"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn

from hiho_pytorch_base.config import NetworkConfig


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        feature_vector_size: int,
        feature_variable_size: int,
        hidden_size: int,
        target_vector_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
    ):
        super().__init__()

        self.variable_processor = nn.Linear(feature_variable_size, feature_vector_size)
        self.speaker_embedder = nn.Embedding(speaker_size, speaker_embedding_size)

        self.main_layers = nn.Sequential(
            nn.Linear(feature_vector_size + speaker_embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.vector_head = nn.Linear(hidden_size, target_vector_size)

        self.scalar_head = nn.Linear(hidden_size, 1)

    def forward(  # noqa: D102
        self,
        *,
        feature_vector: Tensor,  # (B, ?)
        feature_variable_list: list[Tensor],  # [(vL, ?)]
        speaker_id: Tensor,  # (B,)
    ) -> tuple[Tensor, Tensor]:  # (B, ?), (B,)
        variable_means = []
        for var_data in feature_variable_list:  # (vL, ?)
            var_mean = torch.mean(var_data, dim=0)  # (?)
            var_processed = self.variable_processor(var_mean)  # (?)
            variable_means.append(var_processed)

        variable_features = torch.stack(variable_means)  # (B, ?)
        combined_features = feature_vector + variable_features  # (B, ?)

        speaker_embedding = self.speaker_embedder(speaker_id)  # (B, ?)
        final_features = torch.cat(
            [combined_features, speaker_embedding], dim=1
        )  # (B, ?)

        hidden = self.main_layers(final_features)  # (B, ?)

        vector_output = self.vector_head(hidden)  # (B, ?)
        scalar_output = self.scalar_head(hidden).squeeze(-1)  # (B,)

        return vector_output, scalar_output


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    return Predictor(
        feature_vector_size=config.feature_vector_size,
        feature_variable_size=config.feature_variable_size,
        hidden_size=config.hidden_size,
        target_vector_size=config.target_vector_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
    )
