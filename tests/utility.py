"""テスト用ユーティリティ関数"""

from pathlib import Path


def get_data_directory() -> Path:
    """テストデータディレクトリのパスを取得"""
    return Path(__file__).parent.relative_to(Path.cwd()) / "data"
