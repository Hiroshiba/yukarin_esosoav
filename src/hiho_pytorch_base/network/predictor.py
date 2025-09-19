"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from hiho_pytorch_base.config import NetworkConfig
from hiho_pytorch_base.network.conformer.encoder import Encoder
from hiho_pytorch_base.network.transformer.utility import make_non_pad_mask


class PostNet(nn.Module):
    """出力後処理用の畳み込みスタック"""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        blocks: list[nn.Module] = []
        for layer in range(layers - 1):
            in_ch = channels if layer == 0 else hidden_channels
            out_ch = hidden_channels
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )

        in_ch = hidden_channels if layers != 1 else channels
        blocks.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
                nn.Dropout(dropout),
            )
        )

        self.postnet = nn.Sequential(*blocks)

    def forward(  # noqa: D102
        self,
        x: Tensor,  # (B, C, T)
    ) -> Tensor:
        return self.postnet(x)


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        f0_embedding_size: int,
        hidden_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        output_size: int,
        encoder: Encoder,
        postnet_layers: int,
        postnet_kernel_size: int,
        postnet_dropout: float,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: 推論時は行列演算を焼き込める。精度的にdoubleにする必要があるかも
        self.phoneme_embedder = nn.Sequential(
            nn.Embedding(phoneme_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
        )

        # TODO: 推論時は行列演算を焼き込める。精度的にdoubleにする必要があるかも
        self.speaker_embedder = nn.Sequential(
            nn.Embedding(speaker_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
        )
        self.f0_linear = nn.Linear(1, f0_embedding_size)

        embedding_size = (
            f0_embedding_size + phoneme_embedding_size + speaker_embedding_size
        )
        self.pre = nn.Linear(embedding_size, hidden_size)

        self.encoder = encoder

        self.post = nn.Linear(hidden_size, output_size)
        self.postnet = PostNet(
            channels=output_size,
            hidden_channels=hidden_size,
            layers=postnet_layers,
            kernel_size=postnet_kernel_size,
            dropout=postnet_dropout,
        )

    def forward(  # noqa: D102
        self,
        *,
        f0_list: list[Tensor],  # [(L,)]
        phoneme_list: list[Tensor],  # [(L,)]
        speaker_id: Tensor,  # (B,)
    ) -> tuple[list[Tensor], list[Tensor]]:  # [(L, ?)], [(L, ?)]
        device = speaker_id.device
        batch_size = len(f0_list)

        lengths = torch.tensor([x.shape[0] for x in f0_list], device=device)

        # パディング
        f0 = pad_sequence(f0_list, batch_first=True)  # (B, L)
        phoneme_ids = pad_sequence(phoneme_list, batch_first=True)  # (B, L)

        # 埋め込み・連結
        h_phoneme = self.phoneme_embedder(phoneme_ids)  # (B, L, ?)
        h_f0 = self.f0_linear(f0.unsqueeze(-1))  # (B, L, ?))

        h_speaker = self.speaker_embedder(speaker_id)  # (B, ?))
        h_speaker = h_speaker.unsqueeze(1).expand(
            batch_size, f0.size(1), -1
        )  # (B, L, ?))

        h = torch.cat([h_f0, h_phoneme, h_speaker], dim=2)  # (B, L, ?))
        h = self.pre(h)  # (B, L, ?))

        # マスク
        mask = make_non_pad_mask(lengths).unsqueeze(-2).to(device)  # (B, 1, L)

        # Conformer
        h, _ = self.encoder(x=h, cond=None, mask=mask)  # (B, L, ?))
        y1 = self.post(h)  # (B, L, ?))

        # PostNet
        y2 = y1 + self.postnet(y1.transpose(1, 2)).transpose(1, 2)  # (B, L, ?))

        # リストに戻す
        lengths_list: list[int] = lengths.tolist()
        out1_list = [y1[i, :length] for i, length in enumerate(lengths_list)]
        out2_list = [y2[i, :length] for i, length in enumerate(lengths_list)]
        return out1_list, out2_list


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.conformer_block_num,
        dropout_rate=config.conformer_dropout_rate,
        positional_dropout_rate=config.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.conformer_dropout_rate,
        use_macaron_style=True,
        use_conv_glu_module=True,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        f0_embedding_size=config.f0_embedding_size,
        hidden_size=config.hidden_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        output_size=config.output_size,
        encoder=encoder,
        postnet_layers=config.postnet_layers,
        postnet_kernel_size=config.postnet_kernel_size,
        postnet_dropout=config.postnet_dropout,
    )
