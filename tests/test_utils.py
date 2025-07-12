"""テストユーティリティ - シンプルなテストデータ生成"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from hiho_pytorch_base.config import Config


def create_train_config(base_config_path: Path, data_dir: Path) -> Config:
    """学習テスト用の設定を作成"""
    with base_config_path.open() as f:
        config_dict = yaml.safe_load(f)

    config = Config.from_dict(config_dict)
    assert config.dataset.valid is not None

    config.dataset.train.root_dir = data_dir / "train"
    config.dataset.train.feature_vector_pathlist_path = (
        data_dir / "train_feature_vector_pathlist.txt"
    )
    config.dataset.train.feature_variable_pathlist_path = (
        data_dir / "train_feature_variable_pathlist.txt"
    )
    config.dataset.train.target_vector_pathlist_path = (
        data_dir / "train_target_vector_pathlist.txt"
    )
    config.dataset.train.target_scalar_pathlist_path = (
        data_dir / "train_target_scalar_pathlist.txt"
    )

    config.dataset.valid.root_dir = data_dir / "valid"
    config.dataset.valid.feature_vector_pathlist_path = (
        data_dir / "valid_feature_vector_pathlist.txt"
    )
    config.dataset.valid.feature_variable_pathlist_path = (
        data_dir / "valid_feature_variable_pathlist.txt"
    )
    config.dataset.valid.target_vector_pathlist_path = (
        data_dir / "valid_target_vector_pathlist.txt"
    )
    config.dataset.valid.target_scalar_pathlist_path = (
        data_dir / "valid_target_scalar_pathlist.txt"
    )

    return config


def setup_data(config: Config) -> None:
    """テストデータをセットアップ"""
    if config.dataset.train.root_dir.exists():
        return

    assert config.dataset.valid is not None

    def setup_data_type(
        pathlist_path: Path,
        data_type: str,
        sample_indices: np.ndarray,
        root_dir: Path,
        extension: str,
        generator_func: Callable[[Path], None],
    ) -> None:
        """データタイプのディレクトリとpathlistを準備し、ジェネレーター関数でファイルを生成"""
        data_dir = root_dir / data_type
        data_dir.mkdir(parents=True, exist_ok=True)

        relative_paths = [f"{data_type}/{index}.{extension}" for index in sample_indices]
        file_paths = [root_dir / relative_path for relative_path in relative_paths]

        pathlist_path.write_text("\n".join(relative_paths) + "\n")

        for file_path in file_paths:
            generator_func(file_path)

    for config_split, count in zip(
        [config.dataset.train, config.dataset.valid],
        [100, 50],
        strict=True,
    ):
        sample_indices = np.arange(count)

        # 固定長特徴ベクトル
        def generate_feature_vector(file_path: Path) -> None:
            feature_vector = np.random.randn(config.network.feature_vector_size).astype(
                np.float32
            )
            np.save(file_path, feature_vector)

        setup_data_type(
            config_split.feature_vector_pathlist_path,
            "feature_vector",
            sample_indices,
            config_split.root_dir,
            "npy",
            generate_feature_vector,
        )

        # 可変長特徴データ
        def generate_feature_variable(file_path: Path) -> None:
            variable_length = int(np.random.randint(5, 15))
            feature_variable = np.random.randn(
                variable_length, config.network.feature_variable_size
            ).astype(np.float32)
            np.save(file_path, feature_variable)

        setup_data_type(
            config_split.feature_variable_pathlist_path,
            "feature_variable",
            sample_indices,
            config_split.root_dir,
            "npy",
            generate_feature_variable,
        )

        # クラス分類
        def generate_target_vector(file_path: Path) -> None:
            target_class = np.random.randint(
                0, config.network.target_vector_size, dtype=np.int64
            )
            np.save(file_path, target_class)

        setup_data_type(
            config_split.target_vector_pathlist_path,
            "target_vector",
            sample_indices,
            config_split.root_dir,
            "npy",
            generate_target_vector,
        )

        # 回帰ターゲット
        def generate_target_scalar(file_path: Path) -> None:
            target_class = np.random.randint(
                0, config.network.target_vector_size, dtype=np.int64
            )
            target_scalar = float(target_class) + np.random.randn() * 0.1
            np.save(file_path, target_scalar)

        setup_data_type(
            config_split.target_scalar_pathlist_path,
            "target_scalar",
            sample_indices,
            config_split.root_dir,
            "npy",
            generate_target_scalar,
        )
