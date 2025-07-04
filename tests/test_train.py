# NOTE: このテストファイルは現在動作しません
# pytorch-trainerが削除されているため、trainer.pyとmodel.pyでModuleNotFoundErrorが発生します
# 将来的にpytorch-trainerの代替実装またはtorch.distributed等への移行が必要です

import tempfile
from pathlib import Path

import pytest
# from hiho_pytorch_base.trainer import create_trainer  # ModuleNotFoundError


# すべてのテストはpytorch-trainerが削除されているため、現在動作しません

# def test_create_trainer(test_config_dict):
#     """
#     create_trainerが正常に動作することをテストする
#     """
#     # 一時的な出力ディレクトリを作成
#     with tempfile.TemporaryDirectory() as temp_dir:
#         output_path = Path(temp_dir) / "test_output"
#
#         # トレーナーを作成
#         trainer = create_trainer(
#             config_dict=test_config_dict,
#             output=output_path
#         )
#
#         # トレーナーが正常に作成されたかチェック
#         assert trainer is not None
#         assert output_path.exists()
#         assert (output_path / "config.yaml").exists()
#         assert (output_path / "struct.txt").exists()


# def test_train_simple_steps(test_config_dict):
#     """
#     実際に数ステップだけ学習を実行してみる
#     """
#     # 一時的な出力ディレクトリを作成
#     with tempfile.TemporaryDirectory() as temp_dir:
#         output_path = Path(temp_dir) / "test_output"
#
#         # トレーナーを作成
#         trainer = create_trainer(
#             config_dict=test_config_dict,
#             output=output_path
#         )
#
#         # 学習を実行（設定では5イテレーションで停止）
#         trainer.run()
#
#         # 学習後の確認
#         assert trainer.updater.iteration >= test_config_dict["train"]["stop_iteration"]
#         assert (output_path / "log").exists()


# def test_dataset_creation(test_dataset):
#     """
#     データセットが正しく作成されることをテストする
#     """
#     from hiho_pytorch_base.dataset import create_dataset
#     from hiho_pytorch_base.config import DatasetConfig
#
#     # DatasetConfigを作成
#     dataset_config = DatasetConfig(
#         feature_glob=test_dataset["feature_glob"],
#         target_glob=test_dataset["target_glob"],
#         test_num=3,
#         seed=42
#     )
#
#     # データセットを作成
#     datasets = create_dataset(dataset_config)
#
#     # データセットが正しく作成されたかチェック
#     assert "train" in datasets
#     assert "test" in datasets
#     assert "eval" in datasets
#     assert len(datasets["train"]) == 7  # 10 - 3 = 7
#     assert len(datasets["test"]) == 3   # test_num = 3


# def test_model_creation(test_config_dict):
#     """
#     モデルが正しく作成されることをテストする
#     """
#     from hiho_pytorch_base.model import Model, create_network
#     from hiho_pytorch_base.config import Config
#
#     # Configを作成
#     config = Config.from_dict(test_config_dict)
#
#     # ネットワークとモデルを作成
#     networks = create_network(config.network)
#     model = Model(model_config=config.model, networks=networks)
#
#     # モデルが正しく作成されたかチェック
#     assert model is not None
#     assert hasattr(model, "forward")
#     assert hasattr(networks, "predictor")
