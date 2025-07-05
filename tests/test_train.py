import tempfile
from pathlib import Path

import pytest
import yaml

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.dataset import create_dataset
from hiho_pytorch_base.model import Model
from hiho_pytorch_base.network.predictor import create_predictor
from train import train


def test_dataset_creation(test_config_dataclass):
    """
    データセットが正しく作成されることをテストする
    """
    # データセットを作成
    datasets = create_dataset(test_config_dataclass.dataset)

    assert "train" in datasets
    assert "test" in datasets
    assert "eval" in datasets
    assert len(datasets["train"]) == 50
    assert len(datasets["test"]) == 50
    assert len(datasets["eval"]) == 50
    assert len(datasets["valid"]) == 100


def test_model_creation(test_config_dataclass):
    """
    モデルが正しく作成されることをテストする
    """
    predictor = create_predictor(test_config_dataclass.network)
    model = Model(model_config=test_config_dataclass.model, predictor=predictor)

    assert model is not None
    assert hasattr(model, "forward")
    assert predictor is not None


def test_train_simple_epochs(test_config_dict):
    """
    実際に数エポックだけ学習を実行してみる
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_output"
        config_path = Path(temp_dir) / "test_config.yaml"

        with open(config_path, "w") as f:
            yaml.dump(test_config_dict, f)

        train(config_path, output_path)

        assert output_path.exists()
        assert (output_path / "config.yaml").exists()
        assert (output_path / "snapshot.pth").exists()

        # Predictorモデルファイルが作成されているか確認
        predictor_files = list(output_path.glob("predictor_*.pth"))
        assert len(predictor_files) > 0

        # TensorBoardログファイルが作成されているか確認
        tensorboard_files = list(output_path.glob("events.out.tfevents.*"))
        assert len(tensorboard_files) > 0


def test_config_loading(test_config_dict):
    """
    設定ファイルが正しく読み込まれることをテストする
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config_dict, f)
        config_path = Path(f.name)

    try:
        with config_path.open() as f:
            loaded_config_dict = yaml.safe_load(f)

        config = Config.from_dict(loaded_config_dict)

        # 設定が正しく読み込まれたかチェック
        assert config.train.batch_size == test_config_dict["train"]["batch_size"]
        assert config.train.stop_epoch == test_config_dict["train"]["stop_epoch"]
        assert config.network.input_size == test_config_dict["network"]["input_size"]
        assert config.network.hidden_size == test_config_dict["network"]["hidden_size"]
        assert config.network.output_size == test_config_dict["network"]["output_size"]

    finally:
        config_path.unlink()


def test_data_loading(test_dataset):
    """
    生成されたテストデータが正しく読み込まれることをテストする
    """
    # pathlistファイルが存在することを確認
    assert test_dataset["train_feature_pathlist"].exists()
    assert test_dataset["train_target_pathlist"].exists()
    assert test_dataset["valid_feature_pathlist"].exists()
    assert test_dataset["valid_target_pathlist"].exists()

    # pathlistファイルの内容を確認
    with open(test_dataset["train_feature_pathlist"]) as f:
        train_features = f.read().strip().split("\n")

    with open(test_dataset["train_target_pathlist"]) as f:
        train_targets = f.read().strip().split("\n")

    # 100個のファイルが記録されているか確認
    assert len(train_features) == 100
    assert len(train_targets) == 100

    # ファイルが実際に存在するか確認
    data_dir = test_dataset["data_dir"]

    assert (data_dir / train_features[0]).exists()
    assert (data_dir / train_targets[0]).exists()
