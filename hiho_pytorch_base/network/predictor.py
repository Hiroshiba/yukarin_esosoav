"""メインのネットワークモジュール"""

from torch import Tensor, nn

from ..config import NetworkConfig
from .acoustic_predictor import AcousticPredictor
from .conformer.encoder import Encoder
from .vocoder import Vocoder


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        acoustic_predictor: AcousticPredictor,
        vocoder: Vocoder,
        frame_size: int,
        sampling_rate: int,
    ):
        super().__init__()
        self.acoustic_predictor = acoustic_predictor
        self.vocoder = vocoder
        self.frame_size = frame_size
        self.sampling_rate = sampling_rate

    def forward(  # noqa: D102
        self,
        *,
        f0_list: list[Tensor],  # [(fL,)]
        phoneme_list: list[Tensor],  # [(fL,)]
        speaker_id: Tensor,  # (B,)
    ) -> tuple[
        list[Tensor],  # [(fL, ?)]
        list[Tensor],  # [(fL, ?)]
        list[Tensor],  # [(wL,)]
    ]:
        spec1_list, spec2_list = self.acoustic_predictor.forward_list(
            f0_list=f0_list,
            phoneme_list=phoneme_list,
            speaker_id=speaker_id,
        )
        wave_list = self.vocoder.forward_list(spec2_list)

        return spec1_list, spec2_list, wave_list


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    encoder = Encoder(
        hidden_size=config.acoustic.hidden_size,
        condition_size=0,
        block_num=config.acoustic.conformer_block_num,
        dropout_rate=config.acoustic.conformer_dropout_rate,
        positional_dropout_rate=config.acoustic.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.acoustic.conformer_dropout_rate,
        use_macaron_style=True,
        use_conv_glu_module=True,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.acoustic.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    acoustic_predictor = AcousticPredictor(
        phoneme_size=config.acoustic.phoneme_size,
        phoneme_embedding_size=config.acoustic.phoneme_embedding_size,
        f0_embedding_size=config.acoustic.f0_embedding_size,
        hidden_size=config.acoustic.hidden_size,
        speaker_size=config.acoustic.speaker_size,
        speaker_embedding_size=config.acoustic.speaker_embedding_size,
        output_size=config.acoustic.output_size,
        encoder=encoder,
        postnet_layers=config.acoustic.postnet_layers,
        postnet_kernel_size=config.acoustic.postnet_kernel_size,
        postnet_dropout=config.acoustic.postnet_dropout,
    )

    vocoder = Vocoder(
        input_channels=acoustic_predictor.output_size,
        upsample_rates=config.vocoder.upsample_rates,
        upsample_kernel_sizes=config.vocoder.upsample_kernel_sizes,
        upsample_initial_channel=config.vocoder.upsample_initial_channel,
        resblock=config.vocoder.resblock,
        resblock_kernel_sizes=config.vocoder.resblock_kernel_sizes,
        resblock_dilation_sizes=config.vocoder.resblock_dilation_sizes,
    )

    return Predictor(
        acoustic_predictor=acoustic_predictor,
        vocoder=vocoder,
        frame_size=config.frame_size,
        sampling_rate=config.sampling_rate,
    )
