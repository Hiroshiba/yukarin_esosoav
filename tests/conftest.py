"""pytestの共通設定と自動テストデータ生成"""

import os
from pathlib import Path

import pytest

from hiho_pytorch_base.config import Config
from tests.test_utils import setup_data_and_config


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """テスト環境のセットアップ"""
    os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture(scope="session")
def base_config_path() -> Path:
    """ベース設定ファイルのパス"""
    return Path(__file__).parent / "data" / "base_config.yaml"


@pytest.fixture(scope="session", autouse=True)
def data_and_config(base_config_path: Path) -> Config:
    """データディレクトリと学習テスト用の設定のセットアップ"""
    data_dir = Path(__file__).parent / "data" / "data"
    return setup_data_and_config(base_config_path, data_dir)


@pytest.fixture(scope="session")
def train_config(data_and_config: Config) -> Config:
    """学習テスト用設定"""
    return data_and_config


@pytest.fixture(scope="session")
def train_output_dir() -> Path:
    """学習結果ディレクトリのパス"""
    output_dir = Path(__file__).parent / "data" / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
