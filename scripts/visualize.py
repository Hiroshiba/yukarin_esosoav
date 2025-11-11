"""データセット可視化ツール"""

import argparse
import tempfile
from dataclasses import dataclass
from typing import Any

import gradio as gr
import japanize_matplotlib  # noqa: F401 日本語フォントに必須
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data.data import OutputData
from hiho_pytorch_base.data.phoneme import ArpaPhoneme
from hiho_pytorch_base.dataset import (
    Dataset,
    DatasetCollection,
    DatasetType,
    LazyInputData,
    create_dataset,
)


@dataclass
class DataInfo:
    """データ情報"""

    phoneme_info: str
    speaker_id: str
    audio_path: str
    details: str


@dataclass
class FigureState:
    """図の状態"""

    main_plot_fig: Figure | None = None


@dataclass
class PhonemeSegment:
    """音素セグメント情報"""

    phoneme_id: int
    start_frame: int
    end_frame: int

    def to_time(self, frame_rate: float) -> tuple[float, float]:
        """フレームを時間に変換"""
        return self.start_frame / frame_rate, self.end_frame / frame_rate


def extract_phoneme_segments(phoneme_ids: np.ndarray) -> list[PhonemeSegment]:
    """フレームごとの音素ID配列から音素区間を逆算"""
    segments: list[PhonemeSegment] = []
    current_id = phoneme_ids[0]
    start_frame = 0

    for i in range(1, len(phoneme_ids)):
        if phoneme_ids[i] != current_id:
            segments.append(
                PhonemeSegment(
                    phoneme_id=int(current_id),
                    start_frame=start_frame,
                    end_frame=i,
                )
            )
            current_id = phoneme_ids[i]
            start_frame = i

    segments.append(
        PhonemeSegment(
            phoneme_id=int(current_id),
            start_frame=start_frame,
            end_frame=len(phoneme_ids),
        )
    )

    return segments


class VisualizationApp:
    """可視化アプリケーション"""

    def __init__(self, config_path: UPath, initial_dataset_type: DatasetType):
        self.config_path = config_path
        self.initial_dataset_type = initial_dataset_type

        self.dataset_collection = self._create_dataset()
        self.figure_state = FigureState()

    def _create_dataset(self) -> DatasetCollection:
        """データセットを作成"""
        config = Config.from_dict(yaml.safe_load(self.config_path.read_text()))
        return create_dataset(config.dataset)

    def _get_dataset_and_data(
        self, index: int, dataset_type: DatasetType
    ) -> tuple[Dataset, OutputData, LazyInputData]:
        """データセットとデータを取得する共通処理"""
        dataset = self.dataset_collection.get(dataset_type)
        output_data = dataset[index]
        lazy_data = dataset.datas[index]
        return dataset, output_data, lazy_data

    def _get_file_info(self, index: int, dataset_type: DatasetType) -> str:
        """ファイル関連の情報テキストを取得"""
        dataset = self.dataset_collection.get(dataset_type)
        lazy_data = dataset.datas[index]

        return f"""設定ファイル: {self.config_path}

F0データパス: {lazy_data.f0_path}
Volumeデータパス: {lazy_data.volume_path}
LABデータパス: {lazy_data.lab_path}
話者ID: {lazy_data.speaker_id}
音声ファイル: {lazy_data.wave_path}"""

    def _create_data_processing_text(
        self,
        *,
        frame_rate: float,
        f0_len: int,
        phoneme_len: int,
        spec_len: int,
        framed_wave_len: int,
        wave_start_frame: int,
        phoneme_count: int,
        speaker_id: int,
        is_eval: bool,
        prepost_silence_frame_length: int,
        max_frame_length: int,
        wave_frame_length: int,
    ) -> str:
        """可視化対象のフレーム系列概要を表示する"""
        mode = "評価モード" if is_eval else "学習モード"
        return (
            f"--- 前処理パラメータ ---\n"
            f"データモード: {mode}\n"
            f"前後無音長: {prepost_silence_frame_length} frames\n"
            f"最大フレーム長: {max_frame_length} frames\n"
            f"波形切り出し長: {wave_frame_length} frames\n"
            f"\n--- 前処理後のデータ ---\n"
            f"フレームレート: {frame_rate:.2f} Hz\n"
            f"F0 shape: ({f0_len},)\n"
            f"Phoneme shape: ({phoneme_len},)\n"
            f"Spec shape: ({spec_len}, ?)\n"
            f"Framed wave shape: ({framed_wave_len}, ?)\n"
            f"Wave start frame: {wave_start_frame}\n"
            f"音素数: {phoneme_count}\n"
            f"話者ID: {speaker_id}"
        )

    def _create_f0_phoneme_plot(
        self,
        *,
        frame_rate: float,
        f0: np.ndarray,
        phoneme_array: np.ndarray,
        time_start: float,
        time_end: float,
    ) -> Figure:
        """フレームF0/母音F0重心/Volume/音素区間の可視化を行う"""
        self.figure_state.main_plot_fig, ax1 = plt.subplots(1, 1, figsize=(24, 6))

        t = np.arange(len(f0)) / frame_rate

        f0_disp = f0.astype(float).copy()
        f0_disp[f0_disp == 0] = np.nan
        ax1.plot(t, f0_disp, "b-", linewidth=2.5, label="母音F0重心")
        ax1.set_xlabel("時間 (秒)", fontsize=20)
        ax1.set_ylabel("対数F0", fontsize=20, color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax1.set_xlim(time_start, time_end)

        segments = extract_phoneme_segments(phoneme_array)

        y_min, y_max = ax1.get_ylim()
        cmap = plt.cm.tab20  # type: ignore
        max_phoneme_id = len(ArpaPhoneme.phoneme_list)
        for segment in segments:
            start_time, end_time = segment.to_time(frame_rate)
            if end_time >= time_start and start_time <= time_end:
                color = cmap(segment.phoneme_id / max_phoneme_id)
                ax1.axvspan(start_time, end_time, alpha=0.18, color=color)
                mid_time = (start_time + end_time) / 2
                if time_start <= mid_time <= time_end:
                    phoneme_name = ArpaPhoneme.phoneme_list[segment.phoneme_id]
                    ax1.text(
                        mid_time,
                        y_max - (y_max - y_min) * 0.1,
                        phoneme_name,
                        ha="center",
                        va="top",
                        fontsize=15,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.6),
                    )

        ax1.legend(fontsize=16, loc="lower right")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="both", which="major", labelsize=16)

        plt.tight_layout()
        return self.figure_state.main_plot_fig

    def _create_spectrogram_plot(
        self,
        *,
        frame_rate: float,
        spec_data: np.ndarray,
        time_start: float,
        time_end: float,
    ) -> Figure:
        """スペクトログラムを可視化"""
        fig, ax = plt.subplots(1, 1, figsize=(24, 6))

        # 対数メルスペクトログラム
        # spec_dataは (time, freq) の形状を想定
        spec_db = spec_data.T

        # スペクトログラム表示
        ax.imshow(
            spec_db,
            aspect="auto",
            origin="lower",
            extent=(0, len(spec_data) / frame_rate, 0, spec_data.shape[1]),
            cmap="viridis",
            interpolation="none",
        )

        ax.set_xlabel("時間 (秒)", fontsize=20)
        ax.set_ylabel("メル周波数ビン", fontsize=20)
        ax.set_title("メルスペクトログラム", fontsize=22)
        ax.set_xlim(time_start, time_end)
        ax.tick_params(axis="both", which="major", labelsize=16)

        plt.tight_layout()
        return fig

    def _create_framed_wave_plot(
        self,
        *,
        frame_rate: float,
        framed_wave: np.ndarray,
        wave_start_frame: int,
        time_start: float,
        time_end: float,
    ) -> Figure:
        """フレーム化された波形を可視化"""
        fig, ax = plt.subplots(1, 1, figsize=(24, 4))

        wave = framed_wave.reshape(-1)
        sample_rate = frame_rate * framed_wave.shape[1]
        wave_start_time = wave_start_frame / frame_rate
        t = wave_start_time + np.arange(len(wave)) / sample_rate

        ax.plot(t, wave, linewidth=0.5, label="波形")

        ax.axvline(
            x=wave_start_time,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"切り出し開始 (frame={wave_start_frame})",
        )

        ax.set_xlabel("時間 (秒)", fontsize=20)
        ax.set_ylabel("振幅", fontsize=20)
        ax.set_title("フレーム化された波形", fontsize=22)
        ax.set_xlim(time_start, time_end)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16, loc="upper right")

        plt.tight_layout()
        return fig

    def _create_plots(
        self,
        index: int,
        dataset_type: DatasetType,
        time_start: float,
        time_end: float,
    ) -> tuple[Figure, Figure, Figure]:
        """プロットを作成"""
        dataset, output_data, lazy_data = self._get_dataset_and_data(
            index, dataset_type
        )

        input_data = lazy_data.fetch()

        frame_rate = float(input_data.spec_data.rate)
        f0_array = output_data.f0.cpu().numpy()
        phoneme_array = output_data.phoneme.cpu().numpy()
        spec_array = output_data.spec.cpu().numpy()
        framed_wave_array = output_data.framed_wave.cpu().numpy()
        wave_start_frame = int(output_data.wave_start_frame.item())

        f0_plot = self._create_f0_phoneme_plot(
            frame_rate=frame_rate,
            f0=f0_array,
            phoneme_array=phoneme_array,
            time_start=time_start,
            time_end=time_end,
        )

        spectrogram_plot = self._create_spectrogram_plot(
            frame_rate=frame_rate,
            spec_data=spec_array,
            time_start=time_start,
            time_end=time_end,
        )

        framed_wave_plot = self._create_framed_wave_plot(
            frame_rate=frame_rate,
            framed_wave=framed_wave_array,
            wave_start_frame=wave_start_frame,
            time_start=time_start,
            time_end=time_end,
        )

        return f0_plot, spectrogram_plot, framed_wave_plot

    def _get_data_info(self, index: int, dataset_type: DatasetType) -> DataInfo:
        """データ情報を取得"""
        dataset, output_data, lazy_data = self._get_dataset_and_data(
            index, dataset_type
        )

        input_data = lazy_data.fetch()
        phonemes = input_data.phonemes
        frame_rate = float(input_data.spec_data.rate)

        phoneme_info_list = []
        for p in phonemes:
            info_line = f"  {p.phoneme}: {p.start:.3f}-{p.end:.3f}s"
            if ArpaPhoneme.is_vowel(p.phoneme) and p.stress is not None:
                info_line += f" [stress: {p.stress}]"
            phoneme_info_list.append(info_line)
        phoneme_info = "\n".join(phoneme_info_list)

        audio_path_str = str(lazy_data.wave_path)
        speaker_id = f"{output_data.speaker_id.item()}"

        file_info = self._get_file_info(index, dataset_type)

        f0_len = int(output_data.f0.shape[0])
        phoneme_len = int(output_data.phoneme.shape[0])
        spec_len = int(output_data.spec.shape[0])
        framed_wave_len = int(output_data.framed_wave.shape[0])
        wave_start_frame = int(output_data.wave_start_frame.item())

        data_processing_info = self._create_data_processing_text(
            frame_rate=frame_rate,
            f0_len=f0_len,
            phoneme_len=phoneme_len,
            spec_len=spec_len,
            framed_wave_len=framed_wave_len,
            wave_start_frame=wave_start_frame,
            phoneme_count=len(phonemes),
            speaker_id=int(output_data.speaker_id.item()),
            is_eval=dataset.is_eval,
            prepost_silence_frame_length=dataset.config.prepost_silence_frame_length,
            max_frame_length=dataset.config.max_frame_length,
            wave_frame_length=dataset.config.wave_frame_length,
        )
        details = f"{file_info}\n\n{data_processing_info}"

        return DataInfo(
            phoneme_info=phoneme_info,
            speaker_id=speaker_id,
            audio_path=audio_path_str,
            details=details,
        )

    def launch(self) -> None:
        """Gradio UIを起動"""
        initial_dataset = self.dataset_collection.get(self.initial_dataset_type)
        initial_max_index = len(initial_dataset) - 1

        with gr.Blocks() as demo:
            # 状態管理
            current_index = gr.State(0)
            current_dataset_type = gr.State(self.initial_dataset_type)

            # UI コンポーネント
            with gr.Row():
                dataset_type_dropdown = gr.Dropdown(
                    choices=list(DatasetType),
                    value=self.initial_dataset_type,
                    label="データセットタイプ",
                    scale=1,
                )
                index_slider = gr.Slider(
                    minimum=0,
                    maximum=initial_max_index,
                    value=0,
                    step=1,
                    label="サンプルインデックス",
                    scale=3,
                )

            # 状態管理
            current_time_start = gr.State(0.0)
            current_time_end = gr.State(2.0)

            @gr.render(
                inputs=[
                    current_index,
                    current_dataset_type,
                    current_time_start,
                    current_time_end,
                ]
            )
            def render_content(
                index: int,
                dataset_type: DatasetType,
                time_start: float,
                time_end: float,
            ):
                f0_plot, spectrogram_plot, framed_wave_plot = self._create_plots(
                    index, dataset_type, time_start, time_end
                )
                data_info = self._get_data_info(index, dataset_type)

                try:
                    _, _, lazy_data = self._get_dataset_and_data(index, dataset_type)
                    input_data = lazy_data.fetch()
                    wave_data = input_data.wave_data
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(temp_file.name, wave_data.wave, wave_data.sampling_rate)
                    audio_for_gradio = temp_file.name
                except Exception as e:
                    print(f"音声取得エラー: {e}")
                    audio_for_gradio = None

                with gr.Row():
                    time_start_input = gr.Number(
                        value=time_start, label="開始時間 (秒)", scale=1
                    )
                    time_end_input = gr.Number(
                        value=time_end, label="終了時間 (秒)", scale=1
                    )
                    left_btn = gr.Button("← 左へ", scale=1)
                    right_btn = gr.Button("右へ →", scale=1)

                def update_time_range(new_start, new_end):
                    return new_start, new_end

                time_start_input.change(
                    update_time_range,
                    inputs=[time_start_input, time_end_input],
                    outputs=[current_time_start, current_time_end],
                )

                time_end_input.change(
                    update_time_range,
                    inputs=[time_start_input, time_end_input],
                    outputs=[current_time_start, current_time_end],
                )

                def move_time_window(direction, current_start, current_end):
                    window_size = current_end - current_start
                    if direction == "left":
                        new_start = current_start - window_size * 1.0
                    else:
                        new_start = current_start + window_size * 1.0
                    new_end = new_start + window_size
                    return new_start, new_end, new_start, new_end

                left_btn.click(
                    lambda s, e: move_time_window("left", s, e),
                    inputs=[current_time_start, current_time_end],
                    outputs=[
                        current_time_start,
                        current_time_end,
                        time_start_input,
                        time_end_input,
                    ],
                )

                right_btn.click(
                    lambda s, e: move_time_window("right", s, e),
                    inputs=[current_time_start, current_time_end],
                    outputs=[
                        current_time_start,
                        current_time_end,
                        time_start_input,
                        time_end_input,
                    ],
                )

                with gr.Row():
                    if audio_for_gradio:
                        gr.Audio(value=audio_for_gradio, label="全体の音声再生")
                    else:
                        gr.Audio(value=None, label="音声ファイルが見つかりません")

                with gr.Row():
                    gr.Plot(value=f0_plot, label="母音F0重心/音素区間")

                with gr.Row():
                    gr.Plot(value=spectrogram_plot, label="メルスペクトログラム")

                with gr.Row():
                    gr.Plot(value=framed_wave_plot, label="フレーム化された波形")

                with gr.Row():
                    with gr.Column():
                        gr.Textbox(
                            value=data_info.speaker_id,
                            label="話者ID",
                            interactive=False,
                        )
                    with gr.Column():
                        gr.Textbox(
                            value=data_info.audio_path,
                            label="音声ファイル",
                            interactive=False,
                        )

                gr.Markdown("---")
                gr.Textbox(
                    value=data_info.phoneme_info,
                    label="音素区間情報",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                )

                gr.Markdown("---")
                gr.Textbox(
                    value=data_info.details,
                    label="詳細情報",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                )

            # UI操作から状態への更新
            index_slider.change(
                lambda new_index: new_index,
                inputs=[index_slider],
                outputs=[current_index],
            )

            def handle_dataset_change(new_type):
                dataset = self.dataset_collection.get(new_type)
                max_index = len(dataset) - 1
                return (
                    0,  # current_index
                    new_type,  # current_dataset_type
                    gr.update(value=0, maximum=max_index),  # スライダー
                )

            dataset_type_dropdown.change(
                handle_dataset_change,
                inputs=[dataset_type_dropdown],
                outputs=[current_index, current_dataset_type, index_slider],
            )

            # 初期化
            demo.load(
                lambda: (0, self.initial_dataset_type, 0.0, 2.0),
                outputs=[
                    current_index,
                    current_dataset_type,
                    current_time_start,
                    current_time_end,
                ],
            )

        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


def visualize(config_path: UPath, dataset_type: DatasetType) -> None:
    """指定されたデータセットをGradio UIで可視化する"""
    app = VisualizationApp(config_path, dataset_type)
    app.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセットのビジュアライゼーション")
    parser.add_argument("config_path", type=UPath, help="設定ファイルのパス")
    parser.add_argument(
        "--dataset_type",
        type=DatasetType,
        default=DatasetType.TRAIN,
        help="データセットタイプ",
    )

    args = parser.parse_args()
    visualize(config_path=args.config_path, dataset_type=args.dataset_type)
