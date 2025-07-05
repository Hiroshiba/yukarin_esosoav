import os
from pathlib import Path

import pytest

from tests.generate_test_data import ensure_test_data_exists


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
def test_feature_dir(test_data_dir):
    """テスト用特徴量ディレクトリのパスを返す"""
    return test_data_dir / "feature-npy"


@pytest.fixture(scope="session")
def test_target_dir(test_data_dir):
    """テスト用ターゲットディレクトリのパスを返す"""
    return test_data_dir / "target-npy"


@pytest.fixture(scope="session")
def test_dataset(test_data_dir, test_feature_dir, test_target_dir):
    """
    テストデータセットを自動生成し、fixtureとして提供する
    テストデータが存在しない場合は自動生成する（300個：train100、test100、valid100）
    """
    ensure_test_data_exists(
        feature_dir=test_feature_dir,
        target_dir=test_target_dir,
        num_samples=300,
        train_count=100,
        test_count=100,
        valid_count=100,
        feature_shape=(16,),
        num_classes=3,
        seed=42,
    )

    return {
        "data_dir": test_data_dir,
        "feature_dir": test_feature_dir,
        "target_dir": test_target_dir,
        "train_feature_pathlist": test_data_dir / "train_feature_pathlist.txt",
        "train_target_pathlist": test_data_dir / "train_target_pathlist.txt",
        "test_feature_pathlist": test_data_dir / "test_feature_pathlist.txt",
        "test_target_pathlist": test_data_dir / "test_target_pathlist.txt",
        "valid_feature_pathlist": test_data_dir / "valid_feature_pathlist.txt",
        "valid_target_pathlist": test_data_dir / "valid_target_pathlist.txt",
    }


@pytest.fixture(scope="session")
def test_config_dict(test_dataset):
    """テスト用の設定辞書を返す（新しいpathlist方式）"""
    return {
        "dataset": {
            "train_file": {
                "feature_pathlist_path": str(test_dataset["train_feature_pathlist"]),
                "target_pathlist_path": str(test_dataset["train_target_pathlist"]),
                "root_dir": str(test_dataset["feature_dir"].parent),
            },
            "valid_file": {
                "feature_pathlist_path": str(test_dataset["valid_feature_pathlist"]),
                "target_pathlist_path": str(test_dataset["valid_target_pathlist"]),
                "root_dir": str(test_dataset["feature_dir"].parent),
            },
            "test_num": 50,
            "eval_times_num": 1,
            "seed": 42,
        },
        "network": {
            "input_size": 16,
            "hidden_size": 32,
            "output_size": 3,
        },
        "model": {},
        "train": {
            "batch_size": 10,
            "eval_batch_size": 10,
            "log_epoch": 1,
            "eval_epoch": 2,
            "snapshot_epoch": 4,
            "stop_epoch": 3,
            "model_save_num": 2,
            "optimizer": {"name": "adam", "lr": 0.001},
            "scheduler": None,
            "num_processes": 1,
            "use_gpu": False,
            "use_amp": False,
        },
        "project": {"name": "test_project", "category": "test"},
    }


@pytest.fixture(scope="session")
def test_config_dataclass(test_config_dict):
    """テスト用の設定をdataclassオブジェクトとして返す"""
    from hiho_pytorch_base.config import Config

    config = Config.from_dict(test_config_dict)
    return config
