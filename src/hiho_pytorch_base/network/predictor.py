"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn

from hiho_pytorch_base.config import NetworkConfig
from hiho_pytorch_base.network.conformer import ConformerEncoder


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        feature_vector_size: int,
        feature_variable_size: int,
        hidden_size: int,
        target_vector_size: int,
        conformer_layers: int,
        conformer_heads: int,
        conformer_ff_dim: int,
        conformer_kernel_size: int,
        dropout: float,
        speaker_size: int,
        speaker_embedding_size: int,
    ):
        super().__init__()

        self.feature_vector_size = feature_vector_size
        self.feature_variable_size = feature_variable_size
        self.hidden_size = hidden_size

        self.variable_processor = nn.Linear(feature_variable_size, feature_vector_size)
        self.speaker_embedder = nn.Embedding(speaker_size, speaker_embedding_size)

        input_size = feature_vector_size + speaker_embedding_size

        self.pre_layer = nn.Linear(input_size, hidden_size)

        self.conformer_encoder = ConformerEncoder(
            d_model=hidden_size,
            n_layers=conformer_layers,
            n_heads=conformer_heads,
            d_ff=conformer_ff_dim,
            conv_kernel_size=conformer_kernel_size,
            dropout=dropout,
        )

        self.vector_head = nn.Linear(hidden_size, target_vector_size)
        self.scalar_head = nn.Linear(hidden_size, 1)

    def forward(  # noqa: D102
        self,
        *,
        feature_vector: Tensor,
        feature_variable_list: list[Tensor],
        speaker_id: Tensor,
    ) -> tuple[Tensor, Tensor]:
        variable_means = []
        for var_data in feature_variable_list:
            var_mean = torch.mean(var_data, dim=0)
            var_processed = self.variable_processor(var_mean)
            variable_means.append(var_processed)

        variable_features = torch.stack(variable_means)
        combined_features = feature_vector + variable_features

        speaker_embedding = self.speaker_embedder(speaker_id)
        final_features = torch.cat([combined_features, speaker_embedding], dim=1)

        h = self.pre_layer(final_features)

        h = h.unsqueeze(1)

        h, _ = self.conformer_encoder(h)

        h = h.squeeze(1)

        vector_output = self.vector_head(h)
        scalar_output = self.scalar_head(h).squeeze(-1)

        return vector_output, scalar_output


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    return Predictor(
        feature_vector_size=config.feature_vector_size,
        feature_variable_size=config.feature_variable_size,
        hidden_size=config.hidden_size,
        target_vector_size=config.target_vector_size,
        conformer_layers=config.conformer_layers,
        conformer_heads=config.conformer_heads,
        conformer_ff_dim=config.conformer_ff_dim,
        conformer_kernel_size=config.conformer_kernel_size,
        dropout=config.conformer_dropout,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
    )
