"""テストユーティリティ - シンプルなテストデータ生成"""

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

    for config_split, count in zip(
        [config.dataset.train, config.dataset.valid],
        [100, 50],
        strict=True,
    ):
        sample_indices = np.arange(count)

        # 固定長特徴ベクトル
        feature_vector_dir = config_split.root_dir / "feature_vector"
        feature_vector_dir.mkdir(parents=True, exist_ok=True)
        for idx in sample_indices:
            feature_vector = np.random.randn(config.network.feature_vector_size).astype(
                np.float32
            )
            np.save(feature_vector_dir / f"{idx}.npy", feature_vector)
        config_split.feature_vector_pathlist_path.write_text(
            "\n".join(f"{idx}.npy" for idx in sample_indices) + "\n"
        )

        # 可変長特徴データ
        feature_variable_dir = config_split.root_dir / "feature_variable"
        feature_variable_dir.mkdir(parents=True, exist_ok=True)
        for idx in sample_indices:
            variable_length = np.random.randint(5, 15)
            feature_variable = np.random.randn(
                variable_length, config.network.feature_variable_size
            ).astype(np.float32)
            np.save(feature_variable_dir / f"{idx}.npy", feature_variable)
        config_split.feature_variable_pathlist_path.write_text(
            "\n".join(f"{idx}.npy" for idx in sample_indices) + "\n"
        )

        # クラス分類
        target_vector_dir = config_split.root_dir / "target_vector"
        target_vector_dir.mkdir(parents=True, exist_ok=True)
        for idx in sample_indices:
            target_class = np.random.randint(
                0, config.network.target_vector_size, dtype=np.int64
            )
            np.save(target_vector_dir / f"{idx}.npy", target_class)
        config_split.target_vector_pathlist_path.write_text(
            "\n".join(f"{idx}.npy" for idx in sample_indices) + "\n"
        )

        # 回帰ターゲット
        target_scalar_dir = config_split.root_dir / "target_scalar"
        target_scalar_dir.mkdir(parents=True, exist_ok=True)
        for idx in sample_indices:
            target_class = np.random.randint(
                0, config.network.target_vector_size, dtype=np.int64
            )
            target_scalar = float(target_class) + np.random.randn() * 0.1
            np.save(target_scalar_dir / f"{idx}.npy", target_scalar)
        config_split.target_scalar_pathlist_path.write_text(
            "\n".join(f"{idx}.npy" for idx in sample_indices) + "\n"
        )
