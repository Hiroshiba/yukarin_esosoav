"""設定クラスのテスト"""

from pathlib import Path

import yaml
from yaml import SafeLoader

from hiho_pytorch_base.config import Config


def test_from_dict(base_config_path: Path):
    """辞書から設定オブジェクトの作成をテスト"""
    with base_config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    Config.from_dict(d)


def test_to_dict(base_config_path: Path):
    """設定オブジェクトの辞書変換をテスト"""
    with base_config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    Config.from_dict(d).to_dict()


def test_equal_base_config_and_reconstructed(base_config_path: Path):
    """設定の往復変換で同等性が保たれることをテスト"""
    with base_config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    base = Config.from_dict(d)
    base_re = Config.from_dict(base.to_dict())
    assert base == base_re
