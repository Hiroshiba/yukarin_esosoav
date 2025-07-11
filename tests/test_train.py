"""学習システムの統合テスト"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.dataset import create_dataset
from hiho_pytorch_base.model import Model
from hiho_pytorch_base.network.predictor import create_predictor
from scripts.generate import generate
from tests.generate_test_data import create_pathlist_files, generate_multi_type_data
from train import train


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """テスト環境の設定を行う"""
    os.environ["WANDB_MODE"] = "disabled"  # W&Bを無効化
    return


@pytest.fixture(scope="session")
def test_data_dir():
    """テストデータディレクトリのパスを返す"""
    return Path(__file__).parent / "data" / "test_data"


@pytest.fixture(scope="session")
def test_paths(test_data_dir):
    """新しいマルチタイプテストデータセットパスを提供"""
    # 新しいマルチタイプデータを生成
    generate_multi_type_data(
        data_dir=test_data_dir,
        num_samples=300,
        feature_shape=(16,),
        num_classes=3,
        seed=42,
    )

    # pathlistファイルを生成
    pathlist_files = create_pathlist_files(
        data_dir=test_data_dir,
        base_dir=test_data_dir,
        train_count=200,
        valid_count=100,
        seed=42,
    )

    return {
        "data_dir": test_data_dir,
        "train_feature_vector_pathlist": pathlist_files[
            "train_feature_vector_pathlist"
        ],
        "train_feature_variable_pathlist": pathlist_files[
            "train_feature_variable_pathlist"
        ],
        "train_target_vector_pathlist": pathlist_files["train_target_vector_pathlist"],
        "train_target_scalar_pathlist": pathlist_files["train_target_scalar_pathlist"],
        "valid_feature_vector_pathlist": pathlist_files[
            "valid_feature_vector_pathlist"
        ],
        "valid_feature_variable_pathlist": pathlist_files[
            "valid_feature_variable_pathlist"
        ],
        "valid_target_vector_pathlist": pathlist_files["valid_target_vector_pathlist"],
        "valid_target_scalar_pathlist": pathlist_files["valid_target_scalar_pathlist"],
    }


@pytest.fixture(scope="session")
def train_config(test_paths):
    """テスト用のPydantic設定オブジェクトを返す"""
    config_path = Path(__file__).parent / "data" / "train_config.yaml"

    with config_path.open() as f:
        config_dict = yaml.safe_load(f)

    config = Config.from_dict(config_dict)

    # テスト用のパスに更新
    config.dataset.train.feature_vector_pathlist_path = test_paths[
        "train_feature_vector_pathlist"
    ]
    config.dataset.train.feature_variable_pathlist_path = test_paths[
        "train_feature_variable_pathlist"
    ]
    config.dataset.train.target_vector_pathlist_path = test_paths[
        "train_target_vector_pathlist"
    ]
    config.dataset.train.target_scalar_pathlist_path = test_paths[
        "train_target_scalar_pathlist"
    ]
    config.dataset.train.root_dir = test_paths["data_dir"]
    config.dataset.valid.feature_vector_pathlist_path = test_paths[
        "valid_feature_vector_pathlist"
    ]
    config.dataset.valid.feature_variable_pathlist_path = test_paths[
        "valid_feature_variable_pathlist"
    ]
    config.dataset.valid.target_vector_pathlist_path = test_paths[
        "valid_target_vector_pathlist"
    ]
    config.dataset.valid.target_scalar_pathlist_path = test_paths[
        "valid_target_scalar_pathlist"
    ]
    config.dataset.valid.root_dir = test_paths["data_dir"]

    return config


@pytest.fixture(scope="session")
def trained_model_dir(train_config, tmp_path_factory):
    """セッション単位で一度だけ学習を実行し、学習済みモデルディレクトリを返す"""
    temp_dir = tmp_path_factory.mktemp("trained_model")
    output_path = temp_dir / "test_output"
    config_path = temp_dir / "test_config.yaml"

    with open(config_path, "w") as f:
        yaml.dump(train_config.to_dict(), f)

    train(config_path, output_path)

    return output_path


def test_dataset_creation(train_config):
    """データセットが正しく作成されることをテストする"""
    # データセットを作成
    datasets = create_dataset(train_config.dataset)

    assert datasets.train is not None
    assert datasets.test is not None
    assert datasets.eval is not None
    assert len(datasets.train) == 100
    assert len(datasets.test) == 100
    assert len(datasets.eval) == 100
    assert len(datasets.valid) == 100


def test_model_creation(train_config):
    """モデルが正しく作成されることをテストする"""
    predictor = create_predictor(train_config.network)
    model = Model(model_config=train_config.model, predictor=predictor)

    assert model is not None
    assert hasattr(model, "forward")
    assert predictor is not None


def test_train_simple_epochs(trained_model_dir):
    """実際に数エポックだけ学習を実行してみる"""
    assert trained_model_dir.exists()
    assert (trained_model_dir / "config.yaml").exists()
    assert (trained_model_dir / "snapshot.pth").exists()

    predictor_files = list(trained_model_dir.glob("predictor_*.pth"))
    assert len(predictor_files) > 0

    tensorboard_files = list(trained_model_dir.glob("events.out.tfevents.*"))
    assert len(tensorboard_files) > 0


def test_generate_with_trained_model(trained_model_dir, tmp_path):
    """学習済みモデルを使用した推論テスト"""
    generate_output_path = tmp_path / "generate_output"

    generate(
        model_dir=trained_model_dir,
        model_iteration=None,
        model_config=None,
        output_dir=generate_output_path,
        use_gpu=False,
    )

    assert generate_output_path.exists()
    assert (generate_output_path / "arguments.yaml").exists()


def test_config_loading(train_config):
    """設定ファイルが正しく読み込まれることをテストする"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(train_config.to_dict(), f)
        config_path = Path(f.name)

    try:
        with config_path.open() as f:
            loaded_config_dict = yaml.safe_load(f)

        config = Config.from_dict(loaded_config_dict)

        assert config.train.batch_size == train_config.train.batch_size
        assert config.train.stop_epoch == train_config.train.stop_epoch
        assert (
            config.network.feature_vector_size
            == train_config.network.feature_vector_size
        )
        assert config.network.hidden_size == train_config.network.hidden_size
        assert (
            config.network.target_vector_size == train_config.network.target_vector_size
        )

    finally:
        config_path.unlink()


def test_data_loading(test_paths):
    """生成されたテストデータが正しく読み込まれることをテストする"""
    # pathlistファイルが存在することを確認
    assert test_paths["train_feature_vector_pathlist"].exists()
    assert test_paths["train_feature_variable_pathlist"].exists()
    assert test_paths["train_target_vector_pathlist"].exists()
    assert test_paths["train_target_scalar_pathlist"].exists()
    assert test_paths["valid_feature_vector_pathlist"].exists()
    assert test_paths["valid_feature_variable_pathlist"].exists()
    assert test_paths["valid_target_vector_pathlist"].exists()
    assert test_paths["valid_target_scalar_pathlist"].exists()

    # pathlistファイルの内容を確認
    with open(test_paths["train_feature_vector_pathlist"]) as f:
        train_features = f.read().strip().split("\n")

    with open(test_paths["train_target_vector_pathlist"]) as f:
        train_targets = f.read().strip().split("\n")

    # 200個のファイルが記録されているか確認（dataset.pyで100個ずつtrain/testに分割される）
    assert len(train_features) == 200
    assert len(train_targets) == 200

    # ファイルが実際に存在するか確認（新しいディレクトリ構造）
    data_dir = test_paths["data_dir"]

    assert (data_dir / "feature_vector" / train_features[0]).exists()
    assert (data_dir / "target_vector" / train_targets[0]).exists()
