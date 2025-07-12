"""pytestの共通設定と自動テストデータ生成"""

import os
from pathlib import Path

import pytest

from hiho_pytorch_base.config import Config
from tests.test_utils import create_train_config, setup_data


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """テスト環境のセットアップ"""
    os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture(scope="session")
def base_config_path() -> Path:
    """ベース設定ファイルのパスを返す"""
    return Path(__file__).parent / "data" / "base_config.yaml"


@pytest.fixture(scope="session")
def train_config(base_config_path: Path) -> Config:
    """学習テスト用設定を作成"""
    data_dir = Path(__file__).parent / "data" / "data"
    return create_train_config(base_config_path, data_dir)


@pytest.fixture(scope="session", autouse=True)
def setup_data_dir(train_config: Config) -> None:
    """データディレクトリのセットアップ"""
    setup_data(train_config)


@pytest.fixture(scope="session")
def train_output_dir() -> Path:
    """学習結果ディレクトリのパスを返す"""
    output_dir = Path(__file__).parent / "data" / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
