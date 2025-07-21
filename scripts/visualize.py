"""可視化"""

import argparse
from pathlib import Path

import gradio as gr
import japanize_matplotlib  # noqa: F401 日本語フォントに必須
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.dataset import DatasetType, create_dataset


def visualize(config_path: Path, dataset_type: DatasetType):
    """指定されたデータセットをGradio UIで可視化する - 状態中心設計"""
    with config_path.open() as f:
        config = Config.from_dict(yaml.safe_load(f))
    dataset_collection = create_dataset(config.dataset)

    initial_dataset = dataset_collection.get(dataset_type)
    initial_max_index = len(initial_dataset) - 1

    def create_plots(index: int, dataset_type_selected: DatasetType):
        """プロットを作成する"""
        dataset = dataset_collection.get(dataset_type_selected)
        output_data = dataset[index]

        # 固定長特徴ベクトルのプロット
        feature_vector_data = output_data.feature_vector.cpu().numpy().flatten()
        plt.figure(figsize=(10, 4))
        plt.plot(feature_vector_data)
        plt.title("固定長特徴ベクトル")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True)
        feature_vector_plot = plt.gcf()
        plt.close()

        # 可変長特徴データのプロット
        feature_variable_data = output_data.feature_variable.cpu().numpy().flatten()
        plt.figure(figsize=(10, 4))
        plt.plot(feature_variable_data)
        plt.title("可変長特徴データ")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True)
        feature_variable_plot = plt.gcf()
        plt.close()

        return (feature_vector_plot, feature_variable_plot)

    def get_data_info(index: int, dataset_type_selected: DatasetType):
        """データ情報を取得"""
        dataset = dataset_collection.get(dataset_type_selected)
        output_data = dataset[index]
        lazy_data = dataset.datas[index]

        target_vector_df = pd.DataFrame(
            output_data.target_vector.cpu().numpy().reshape(1, -1)
        )
        target_scalar_str = f"{output_data.target_scalar.item():.6f}"
        speaker_id_str = f"{output_data.speaker_id.item()}"

        details = f"""
データセットタイプ: {dataset_type_selected}
Index: {index}

設定ファイル: {config_path}

固定長特徴ベクトル
パス: {lazy_data.feature_vector_path}
shape: {tuple(output_data.feature_vector.shape)}

可変長特徴データ
パス: {lazy_data.feature_variable_path}
shape: {tuple(output_data.feature_variable.shape)}

サンプリングデータ
パス: {lazy_data.target_vector_path}
shape: {tuple(output_data.target_vector.shape)}

回帰ターゲット
パス: {lazy_data.target_scalar_path}
shape: {tuple(output_data.target_scalar.shape)}

話者ID: {output_data.speaker_id.item()}
"""

        return (
            target_vector_df,
            target_scalar_str,
            speaker_id_str,
            details,
        )

    with gr.Blocks() as demo:
        # 状態
        current_index = gr.State(0)
        current_dataset_type = gr.State(dataset_type)

        # UI コンポーネント
        with gr.Row():
            dataset_type_dropdown = gr.Dropdown(
                choices=list(DatasetType),
                value=dataset_type,
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

        @gr.render(inputs=[current_index, current_dataset_type])
        def render_content(index: int, dataset_type: DatasetType):
            plots = create_plots(index, dataset_type)
            data_info = get_data_info(index, dataset_type)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 固定長特徴ベクトル")
                    gr.Plot(value=plots[0], label="feature_vector")

                with gr.Column():
                    gr.Markdown("### 可変長特徴データ")
                    gr.Plot(value=plots[1], label="feature_variable")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### サンプリングデータ")
                    gr.DataFrame(value=data_info[0], label="target_vector")

                with gr.Column():
                    gr.Markdown("### その他の値")
                    gr.Textbox(
                        value=data_info[1], label="回帰ターゲット", interactive=False
                    )
                    gr.Textbox(value=data_info[2], label="話者ID", interactive=False)

            gr.Markdown("### 詳細情報")
            gr.Textbox(
                value=data_info[3],
                label="詳細情報",
                lines=20,
                max_lines=30,
                interactive=False,
            )

        # 状態変更によるUI同期
        def sync_slider_from_state(index: int, dataset_type: DatasetType):
            dataset = dataset_collection.get(dataset_type)
            max_index = len(dataset) - 1

            return (
                index,  # index_slider value
                gr.update(maximum=max_index),  # index_slider max
            )

        current_index.change(
            sync_slider_from_state,
            inputs=[current_index, current_dataset_type],
            outputs=[index_slider, index_slider],
        )

        current_dataset_type.change(
            sync_slider_from_state,
            inputs=[current_index, current_dataset_type],
            outputs=[index_slider, index_slider],
        )

        # UI操作から状態への更新
        index_slider.change(
            lambda new_index: new_index, inputs=[index_slider], outputs=[current_index]
        )

        dataset_type_dropdown.change(
            lambda new_type: (0, new_type),
            inputs=[dataset_type_dropdown],
            outputs=[current_index, current_dataset_type],
        )

        # 初期化
        demo.load(
            lambda: (0, dataset_type),
            outputs=[current_index, current_dataset_type],
        )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセットのビジュアライゼーション")
    parser.add_argument("config_path", type=Path, help="設定ファイルのパス")
    parser.add_argument(
        "--dataset_type", type=DatasetType, required=True, help="データセットタイプ"
    )

    args = parser.parse_args()
    visualize(config_path=args.config_path, dataset_type=args.dataset_type)
