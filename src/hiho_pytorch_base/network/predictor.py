"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn

from hiho_pytorch_base.config import NetworkConfig


class Predictor(nn.Module):
    """メインのネットワーク"""

    # TODO: ちゃんとしたネットワーク設計にする

    def __init__(
        self,
        phoneme_size: int,
        hidden_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
    ):
        super().__init__()

        self.phoneme_size = phoneme_size
        self.hidden_size = hidden_size

        self.phoneme_embedder = nn.Embedding(phoneme_size, hidden_size)

        # TODO: 推論時は行列演算を焼き込める。精度的にdoubleにする必要があるかも
        self.phoneme_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.speaker_embedder = nn.Embedding(speaker_size, speaker_embedding_size)

        # 適当な線形層の組み合わせ
        input_size = (
            hidden_size + speaker_embedding_size + 2
        )  # 音素埋め込み + 話者埋め込み + F0平均 + Volume平均
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)

        # 出力ヘッド
        self.f0_head = nn.Linear(hidden_size // 2, 1)  # F0予測用
        self.vuv_head = nn.Linear(hidden_size // 2, 1)  # VUV予測用

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(  # noqa: D102
        self,
        *,
        lab_phoneme_ids: Tensor,  # (B, L)
        lab_durations: Tensor,  # (B, L)
        f0_data: Tensor,  # (B, T)
        volume_data: Tensor,  # (B, T)
        speaker_id: Tensor,  # (B,)
    ) -> tuple[Tensor, Tensor]:  # (B,), (B,)
        # 音素埋め込みの平均を計算（超雑）
        phoneme_embed = self.phoneme_embedder(lab_phoneme_ids)  # (B, L, ?)
        phoneme_embed = self.phoneme_transform(phoneme_embed)  # (B, L, ?)
        phoneme_mean = torch.mean(phoneme_embed, dim=1)  # (B, ?)

        # 話者埋め込み
        speaker_embed = self.speaker_embedder(speaker_id)  # (B, ?)

        # F0とVolumeの平均を計算（超雑）
        f0_mean = torch.mean(f0_data, dim=1).squeeze(-1).unsqueeze(-1)  # (B, 1)
        volume_mean = torch.mean(volume_data, dim=1).squeeze(-1).unsqueeze(-1)  # (B, 1)

        # 全部をconcatenate
        features = torch.cat(
            [phoneme_mean, speaker_embed, f0_mean, volume_mean], dim=1
        )  # (B, ?)

        # 適当な線形層を通す
        h = self.relu(self.fc1(features))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))
        h = self.dropout(h)

        f0_output = self.f0_head(h).squeeze(-1)  # (B,)
        vuv_output = self.vuv_head(h).squeeze(-1)  # (B,)

        return f0_output, vuv_output


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    return Predictor(
        phoneme_size=config.phoneme_size,
        hidden_size=config.hidden_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
    )
