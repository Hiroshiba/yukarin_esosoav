"""データ処理関数のテスト"""

import numpy
import torch

from hiho_pytorch_base.data.data import InputData, OutputData, preprocess
from hiho_pytorch_base.data.phoneme import ArpaPhoneme
from hiho_pytorch_base.data.sampling_data import SamplingData


def create_arpa_phonemes(
    phoneme_names: list[str],
    durations: list[float],
    stress_values: list[int | None],
) -> list[ArpaPhoneme]:
    """ArpaPhonemeのリストを作成"""
    phonemes = []
    start = 0.0
    for name, duration, stress in zip(
        phoneme_names, durations, stress_values, strict=True
    ):
        end = start + duration
        phoneme = ArpaPhoneme(phoneme=name, start=start, end=end, stress=stress)
        phonemes.append(phoneme)
        start = end

    return phonemes


def create_basic_input_data(
    phoneme_names: list[str],
    durations: list[float],
    stress_values: list[int | None],
    f0_values: numpy.ndarray,
    volume_amplitude: float,
    frame_rate: float,
    speaker_id: int,
) -> InputData:
    """テスト用の基本的なInputDataを作成"""
    phonemes = create_arpa_phonemes(phoneme_names, durations, stress_values)

    total_time = sum(durations)
    total_frames = int(total_time * frame_rate)

    volume_db = numpy.full(total_frames, 20 * numpy.log10(volume_amplitude))

    spec_dim = 8
    spec_array = (
        numpy.random.default_rng(0)
        .normal(0, 1, (total_frames, spec_dim))
        .astype(numpy.float32)
    )
    silence_array = numpy.random.default_rng(1).random(total_frames) < 0.2

    f0_data = SamplingData(array=f0_values[:, numpy.newaxis], rate=frame_rate)
    volume_data = SamplingData(array=volume_db[:, numpy.newaxis], rate=frame_rate)
    silence_data = SamplingData(array=silence_array[:, numpy.newaxis], rate=frame_rate)
    spec_data = SamplingData(array=spec_array, rate=frame_rate)

    return InputData(
        phonemes=phonemes,
        f0_data=f0_data,
        volume_data=volume_data,
        silence_data=silence_data,
        spec_data=spec_data,
        speaker_id=speaker_id,
    )


def assert_output_data_types(output_data: OutputData) -> None:
    """OutputDataの型が正しいことを検証"""
    assert isinstance(output_data, OutputData)
    assert output_data.f0.dtype == torch.float
    assert output_data.phoneme.dtype == torch.long
    assert output_data.spec.dtype == torch.float
    assert output_data.speaker_id.dtype == torch.long


def test_input_data_structure():
    """InputData構造の基本動作テスト"""
    input_data = create_basic_input_data(
        phoneme_names=["HH", "AA", "L"],
        durations=[0.1, 0.2, 0.1],
        stress_values=[None, 1, None],
        f0_values=numpy.ones(40) * 150.0,
        volume_amplitude=0.8,
        frame_rate=100.0,
        speaker_id=0,
    )

    # 基本構造の確認
    assert len(input_data.phonemes) == 3
    assert input_data.phonemes[0].phoneme == "HH"
    assert input_data.phonemes[1].phoneme == "AA"
    assert input_data.phonemes[2].phoneme == "L"

    assert input_data.phonemes[0].stress is None  # 子音
    assert input_data.phonemes[1].stress == 1  # 母音
    assert input_data.phonemes[2].stress is None  # 子音

    assert isinstance(input_data.f0_data, SamplingData)
    assert isinstance(input_data.volume_data, SamplingData)
    assert input_data.f0_data.rate == input_data.volume_data.rate

    output_data = preprocess(
        input_data,
        prepost_silence_length=2,
        max_sampling_length=1000,
        is_eval=True,
    )
    assert_output_data_types(output_data)

    length = len(output_data.f0)
    assert length == len(output_data.phoneme)
    assert length == len(output_data.spec)
