import pytest
from pathlib import Path
from tests.generate_test_data import ensure_test_data_exists


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
def test_dataset(test_feature_dir, test_target_dir):
    """
    テストデータセットを自動生成し、fixtureとして提供する
    テストデータが存在しない場合は自動生成する
    """
    ensure_test_data_exists(
        feature_dir=test_feature_dir,
        target_dir=test_target_dir,
        num_samples=10,
        feature_shape=(64,),
        num_classes=3,
        seed=42
    )
    
    return {
        "feature_dir": test_feature_dir,
        "target_dir": test_target_dir,
        "feature_glob": str(test_feature_dir / "*.npy"),
        "target_glob": str(test_target_dir / "*.npy"),
    }


@pytest.fixture(scope="session")
def test_config_dict(test_dataset):
    """テスト用の設定辞書を返す"""
    return {
        "dataset": {
            "feature_glob": test_dataset["feature_glob"],
            "target_glob": test_dataset["target_glob"],
            "test_num": 3,
            "seed": 42
        },
        "network": {},
        "model": {},
        "train": {
            "batch_size": 2,
            "eval_batch_size": 2,
            "log_iteration": 1,
            "eval_iteration": 5,
            "snapshot_iteration": 10,
            "stop_iteration": 5,
            "optimizer": {
                "name": "adam",
                "lr": 0.001
            },
            "use_gpu": False,  # テストではCPUを使用
            "use_amp": False,
            "use_multithread": False
        },
        "project": {
            "name": "test_project",
            "category": "test"
        }
    }