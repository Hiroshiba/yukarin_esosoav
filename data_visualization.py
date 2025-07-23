import tempfile
from pathlib import Path

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data.phoneme import ArpaPhoneme
from hiho_pytorch_base.dataset import create_dataset

try:
    # 日本語フォント対応
    import japanize_matplotlib  # noqa: F401
except ImportError:
    print(
        "japanize_matplotlib インポートに失敗（日本語フォントが使えない可能性があります）"
    )


def get_audio_path_from_lab(lab_file_path: Path) -> Path | None:
    """.labファイルのパスから対応する音声ファイルのパスを取得"""
    stem = lab_file_path.stem

    # LibriTTSの命名規則: {speaker_id}_{chapter_id}_{utterance_id}_{segment_id}
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"無効なステム形式: {stem}")

    speaker_id = parts[0]
    chapter_id = parts[1]

    # LibriTTSのroot directory
    libritts_root = Path("/tmp/datasets/LibriTTS_clean_data/LibriTTS")

    audio_path = libritts_root / "dev-clean" / speaker_id / chapter_id / f"{stem}.wav"

    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    return audio_path


def extract_audio_segment(audio_path: Path, start_time: float, end_time: float) -> str:
    """音声ファイルから指定時間範囲を切り出して一時ファイルとして保存"""
    try:
        audio, sr = librosa.load(str(audio_path), sr=None)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        requested_length = end_sample - start_sample

        if start_sample >= len(audio) or requested_length <= 0:
            raise ValueError(f"無効な時間範囲: {start_time} - {end_time}")

        # 実際に取得できる範囲を計算
        actual_start = max(0, start_sample)
        actual_end = min(len(audio), end_sample)

        # 音声を切り出し
        if actual_start < actual_end:
            audio_segment = audio[actual_start:actual_end]
        else:
            audio_segment = np.array([], dtype=audio.dtype)

        # パディングが必要な場合は追加
        if len(audio_segment) < requested_length:
            padding_length = requested_length - len(audio_segment)
            padding = np.zeros(padding_length, dtype=audio.dtype)
            audio_segment = np.concatenate([audio_segment, padding])

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_segment, sr)

        return temp_file.name

    except Exception as e:
        print(f"音声切り出しエラー: {e}")
        raise


def main():
    config_path = Path("hiho_check/config.yaml")

    with config_path.open() as f:
        config_dict = yaml.safe_load(f)

    config = Config.from_dict(config_dict)

    try:
        datasets = create_dataset(config.dataset)
        datasets.train[0]

    except Exception:
        pass


def create_gradio_interface():
    try:
        config_path = Path("hiho_check/config.yaml")
        with config_path.open() as f:
            config_dict = yaml.safe_load(f)
        config = Config.from_dict(config_dict)
        datasets = create_dataset(config.dataset)

        def visualize_sample(sample_index, time_start=0, time_end=2):
            try:
                sample = datasets.train[sample_index]
                from hiho_pytorch_base.data.data import preprocess
                from hiho_pytorch_base.dataset import get_datas

                datas = get_datas(config.dataset.train)
                lazy_data = datas[sample_index]

                try:
                    audio_path = get_audio_path_from_lab(lazy_data.lab_path)
                except (FileNotFoundError, ValueError):
                    audio_path = None

                input_data = lazy_data.generate()
                output_data = preprocess(input_data, is_eval=True)
                f0_values = output_data.f0.detach().numpy()
                volume_values = output_data.volume.detach().numpy()
                vowel_f0_means_calculated = output_data.vowel_f0_means.detach().numpy()
                stress_values = output_data.stress.detach().numpy()
                vowel_indices = output_data.vowel_index.detach().numpy()

                f0_rate = input_data.f0_data.rate
                phonemes = ArpaPhoneme.load_julius_list(
                    lazy_data.lab_path, verify=False
                )
                fig, ax1 = plt.subplots(1, 1, figsize=(24, 6))

                f0_time = np.arange(len(f0_values)) / f0_rate
                volume_time = np.arange(len(volume_values)) / f0_rate
                f0_values_display = f0_values.copy()
                f0_values_display[f0_values == 0] = np.nan

                ax1.plot(f0_time, f0_values_display, "b-", linewidth=3, label="F0")
                ax1.set_xlabel("時間 (秒)", fontsize=20)
                ax1.set_ylabel("F0 (Hz)", fontsize=20, color="b")
                ax1.tick_params(axis="y", labelcolor="b")

                # volumeデータの可視化を追加
                ax2 = ax1.twinx()
                ax2.plot(
                    volume_time,
                    volume_values,
                    "r-",
                    linewidth=2,
                    label="Volume",
                    alpha=0.7,
                )
                ax2.set_ylabel("Volume (dB)", fontsize=20, color="r")
                ax2.tick_params(axis="y", labelcolor="r")

                ax1.set_xlim(time_start, time_end)
                ax2.set_xlim(time_start, time_end)

                y_min, y_max = ax1.get_ylim()

                cmap = plt.cm.tab20  # type: ignore
                max_phoneme_id = len(ArpaPhoneme.phoneme_list)

                vowel_stress_map = {}
                for i, vowel_idx in enumerate(vowel_indices):
                    if i < len(stress_values):
                        vowel_stress_map[vowel_idx] = stress_values[i]

                for phoneme_idx, phoneme in enumerate(phonemes):
                    # 表示範囲内の音素のみ処理
                    if phoneme.end >= time_start and phoneme.start <= time_end:
                        # 音素IDに基づいて色を決定
                        color = cmap(phoneme.phoneme_id / max_phoneme_id)
                        ax1.axvspan(phoneme.start, phoneme.end, alpha=0.3, color=color)
                        # 音素ラベルを中央に配置
                        mid_time = (phoneme.start + phoneme.end) / 2
                        if time_start <= mid_time <= time_end:
                            ax1.text(
                                mid_time,
                                y_max - (y_max - y_min) * 0.1,
                                phoneme.phoneme,
                                ha="center",
                                va="top",
                                fontsize=18,
                                rotation=0,
                                bbox=dict(
                                    boxstyle="round,pad=0.3", facecolor=color, alpha=0.8
                                ),
                            )

                            # 母音の場合はアクセント値も表示
                            if (
                                ArpaPhoneme.is_vowel(phoneme.phoneme)
                                and phoneme_idx in vowel_stress_map
                            ):
                                stress_value = vowel_stress_map[phoneme_idx]
                                if stress_value > 0:  # 0の場合は何も表示しない
                                    stress_display = "*" * stress_value
                                    ax1.text(
                                        mid_time,
                                        y_max - (y_max - y_min) * 0.05,
                                        stress_display,
                                        ha="center",
                                        va="top",
                                        fontsize=28,
                                        fontweight="bold",
                                        color="black",
                                    )

                # 両方の軸の凡例を組み合わせて表示
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    fontsize=18,
                    loc="lower right",
                    ncol=len(labels1 + labels2),
                )

                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis="both", which="major", labelsize=18)
                ax2.tick_params(axis="both", which="major", labelsize=18)

                try:
                    vowel_idx = 0
                    plotted_vowels = 0

                    for phoneme in phonemes:
                        if ArpaPhoneme.is_vowel(phoneme.phoneme):
                            # 表示範囲内の母音のみ処理
                            if phoneme.end >= time_start and phoneme.start <= time_end:
                                mid_time = (phoneme.start + phoneme.end) / 2
                                if (
                                    vowel_idx < len(vowel_f0_means_calculated)
                                    and vowel_f0_means_calculated[vowel_idx] > 0
                                ):
                                    # 重心F0を点でプロット
                                    ax1.scatter(
                                        mid_time,
                                        vowel_f0_means_calculated[vowel_idx],
                                        s=200,
                                        c="red",
                                        marker="o",
                                        edgecolors="black",
                                        linewidth=2,
                                        zorder=10,
                                        label="Vowel F0 Centroid"
                                        if plotted_vowels == 0
                                        else "",
                                    )
                                    # 重心F0値をテキストで表示
                                    ax1.annotate(
                                        f"{vowel_f0_means_calculated[vowel_idx]:.1f}Hz",
                                        xy=(
                                            float(mid_time),
                                            float(vowel_f0_means_calculated[vowel_idx]),
                                        ),
                                        xytext=(5, 10),
                                        textcoords="offset points",
                                        fontsize=14,
                                        fontweight="bold",
                                        bbox=dict(
                                            boxstyle="round,pad=0.3",
                                            facecolor="yellow",
                                            alpha=0.8,
                                        ),
                                        arrowprops=dict(
                                            arrowstyle="->",
                                            connectionstyle="arc3,rad=0",
                                        ),
                                    )
                                    plotted_vowels += 1
                            vowel_idx += 1

                    # 母音F0重心をプロットした後にレジェンドを更新
                    if plotted_vowels > 0:
                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(
                            lines1 + lines2,
                            labels1 + labels2,
                            fontsize=18,
                            loc="lower right",
                            ncol=len(labels1 + labels2),
                        )

                except Exception:
                    pass

                plt.tight_layout()

                if audio_path:
                    try:
                        audio_for_gradio = extract_audio_segment(
                            audio_path, time_start, time_end
                        )
                    except Exception as e:
                        print(f"音声切り出しに失敗: {e}")
                        audio_for_gradio = None
                else:
                    audio_for_gradio = None

                print("情報テキストの作成...")
                phoneme_info_list = []
                for i, p in enumerate(phonemes):
                    info_line = f"  {p.phoneme}: {p.start:.3f}-{p.end:.3f}s"
                    if ArpaPhoneme.is_vowel(p.phoneme) and i in vowel_stress_map:
                        info_line += f" [stress: {vowel_stress_map[i]}]"
                    phoneme_info_list.append(info_line)
                phoneme_info = "\n".join(phoneme_info_list)
                max_time = (
                    max(phonemes[-1].end, f0_time[-1], volume_time[-1])
                    if phonemes
                    else max(f0_time[-1], volume_time[-1])
                )
                # 基本統計
                vowel_count = len(vowel_indices)
                stress_count = {0: 0, 1: 0, 2: 0}
                for stress in stress_values:
                    if stress in stress_count:
                        stress_count[stress] += 1

                # F0統計
                valid_f0 = f0_values[f0_values > 0]
                f0_detection_rate = (
                    len(valid_f0) / len(f0_values) * 100 if len(f0_values) > 0 else 0
                )
                f0_stats = {
                    "min": float(valid_f0.min()) if len(valid_f0) > 0 else 0,
                    "max": float(valid_f0.max()) if len(valid_f0) > 0 else 0,
                    "mean": float(valid_f0.mean()) if len(valid_f0) > 0 else 0,
                    "std": float(valid_f0.std()) if len(valid_f0) > 0 else 0,
                }

                # 母音F0重心統計
                valid_vowel_f0 = vowel_f0_means_calculated[
                    vowel_f0_means_calculated > 0
                ]
                vowel_f0_stats = {
                    "count": len(valid_vowel_f0),
                    "min": float(valid_vowel_f0.min())
                    if len(valid_vowel_f0) > 0
                    else 0,
                    "max": float(valid_vowel_f0.max())
                    if len(valid_vowel_f0) > 0
                    else 0,
                    "mean": float(valid_vowel_f0.mean())
                    if len(valid_vowel_f0) > 0
                    else 0,
                }

                # 音素継続時間統計
                phoneme_durations = output_data.phoneme_duration.detach().numpy()
                duration_stats = {
                    "mean": float(phoneme_durations.mean()),
                    "min": float(phoneme_durations.min()),
                    "max": float(phoneme_durations.max()),
                    "total": float(phoneme_durations.sum()),
                }

                info_text = f"""
                基本情報:
                - 音素数: {len(phonemes)} (母音: {vowel_count})
                - アクセント統計: stress0={stress_count[0]}, stress1={stress_count[1]}, stress2={stress_count[2]}
                - 話者ID: {int(output_data.speaker_id)}
                - 音声ファイル: {audio_path.name if audio_path else "見つからない"}

                時間・フレーム情報:
                - F0フレーム数: {len(f0_values)} (検出率: {f0_detection_rate:.1f}%)
                - Volumeフレーム数: {len(volume_values)}
                - サンプリングレート: {f0_rate:.2f} Hz
                - 総時間: {f0_time[-1]:.3f} 秒 (最大: {max_time:.3f} 秒)
                - 表示範囲: {time_start:.3f} - {time_end:.3f} 秒 ({time_end - time_start:.3f} 秒間)

                F0統計 (有効値のみ):
                - 範囲: {f0_stats["min"]:.1f} - {f0_stats["max"]:.1f} Hz
                - 平均: {f0_stats["mean"]:.1f} ± {f0_stats["std"]:.1f} Hz

                母音F0重心統計:
                - 有効母音数: {vowel_f0_stats["count"]}/{vowel_count}
                - 範囲: {vowel_f0_stats["min"]:.1f} - {vowel_f0_stats["max"]:.1f} Hz
                - 平均: {vowel_f0_stats["mean"]:.1f} Hz

                音素継続時間統計:
                - 平均: {duration_stats["mean"]:.3f} 秒
                - 範囲: {duration_stats["min"]:.3f} - {duration_stats["max"]:.3f} 秒
                - 合計: {duration_stats["total"]:.3f} 秒

                Volume統計:
                - 範囲: {float(output_data.volume.min()):.2f} - {float(output_data.volume.max()):.2f} dB
                - 平均: {float(output_data.volume.mean()):.2f} dB

                音素区間:
                {phoneme_info}
                """

                print(f"サンプル {sample_index} の可視化完了")
                return fig, info_text, audio_for_gradio

            except Exception as e:
                import traceback

                error_msg = f"エラーが発生しました: {str(e)}"
                error_trace = traceback.format_exc()
                print(f"サンプル {sample_index} の可視化中にエラー: {error_msg}")
                print(f"エラーのトレースバック:\n{error_trace}")

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(
                    0.5,
                    0.5,
                    f"エラー: {str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return (
                    fig,
                    f"エラーが発生しました: {error_msg}\n\nトレースバック:\n{error_trace}",
                    None,  # 音声ファイルなし
                )

        # Gradioインターフェースの作成
        print("Gradioインターフェースの設定中...")

        def move_time_window(sample_index, time_start, time_end, direction):
            window_size = time_end - time_start
            if direction == "left":
                new_start = max(0, time_start - window_size * 1.0)
            else:  # right
                new_start = time_start + window_size * 1.0
            new_end = new_start + window_size
            # 時間値のみを更新（プロットは時間値変更イベントで更新される）
            return new_start, new_end

        with gr.Blocks() as interface:
            with gr.Row():
                sample_slider = gr.Slider(
                    minimum=0,
                    maximum=len(datasets.train) - 1,
                    value=0,
                    step=1,
                    label="サンプルインデックス",
                    scale=2,
                )

            with gr.Row():
                time_start = gr.Number(value=0, label="開始時間 (秒)", scale=1)
                time_end = gr.Number(value=2, label="終了時間 (秒)", scale=1)
                left_btn = gr.Button("← 左へ", scale=1)
                right_btn = gr.Button("右へ →", scale=1)

            with gr.Row():
                audio_output = gr.Audio(label="表示範囲の音声再生", type="filepath")

            with gr.Row():
                plot_output = gr.Plot(label="データ可視化")

            with gr.Row():
                info_output = gr.Textbox(label="サンプル情報", lines=8)

            def update_visualization(sample_index, time_start, time_end):
                return visualize_sample(sample_index, time_start, time_end)

            def update_sample_and_reset_time(sample_index):
                plot, text, audio = visualize_sample(sample_index, 0, 2)
                return plot, text, audio, 0, 2

            sample_slider.change(
                fn=update_sample_and_reset_time,
                inputs=[sample_slider],
                outputs=[plot_output, info_output, audio_output, time_start, time_end],
            )

            time_start.change(
                fn=update_visualization,
                inputs=[sample_slider, time_start, time_end],
                outputs=[plot_output, info_output, audio_output],
            )

            time_end.change(
                fn=update_visualization,
                inputs=[sample_slider, time_start, time_end],
                outputs=[plot_output, info_output, audio_output],
            )

            left_btn.click(
                fn=lambda s, ts, te: move_time_window(s, ts, te, "left"),
                inputs=[sample_slider, time_start, time_end],
                outputs=[time_start, time_end],
            )

            right_btn.click(
                fn=lambda s, ts, te: move_time_window(s, ts, te, "right"),
                inputs=[sample_slider, time_start, time_end],
                outputs=[time_start, time_end],
            )

            interface.load(
                fn=update_visualization,
                inputs=[
                    gr.Number(value=0, visible=False),
                    gr.Number(value=0, visible=False),
                    gr.Number(value=2, visible=False),
                ],
                outputs=[plot_output, info_output, audio_output],
            )

        return interface

    except Exception:
        raise


demo = create_gradio_interface()


if __name__ == "__main__":
    try:
        interface = demo
        interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
    except Exception:
        raise
