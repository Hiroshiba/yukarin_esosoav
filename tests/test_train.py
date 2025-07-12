"""
学習システムのシンプルなテスト。

学習は重いので１回だけ行えば良いように設計されている。
"""

import shutil
from pathlib import Path

import pytest
import yaml

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.dataset import create_dataset
from hiho_pytorch_base.model import Model
from hiho_pytorch_base.network.predictor import create_predictor
from scripts.generate import generate
from train import train


def pytest_collection_modifyitems(items):
    """end-to-endテストの実行順序を制御（train → generate）"""
    e2e_train = [i for i in items if "e2e_train" in i.name]
    e2e_generate = [i for i in items if "e2e_generate" in i.name]
    others = [i for i in items if "e2e" not in i.name]

    # 実行順序: その他のテスト → e2e_train → e2e_generate
    items[:] = others + e2e_train + e2e_generate


def test_dataset_creation(train_config: Config) -> None:
    """データセットの作成テスト"""
    datasets = create_dataset(train_config.dataset)

    assert datasets.train is not None
    assert datasets.test is not None
    assert datasets.eval is not None
    assert datasets.valid is not None


def test_model_creation(train_config: Config) -> None:
    """モデルの作成テスト"""
    predictor = create_predictor(train_config.network)
    model = Model(model_config=train_config.model, predictor=predictor)

    assert model is not None
    assert hasattr(model, "forward")


def test_e2e_train(train_config: Config, train_output_dir: Path) -> None:
    """学習の統合テスト - 学習実行と結果ファイルの生成確認"""
    output_dir = train_output_dir / "trained_model"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    config_path = train_output_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(train_config.to_dict(), f)

    train(config_path, output_dir)

    assert output_dir.exists()
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "snapshot.pth").exists()

    predictor_files = list(output_dir.glob("predictor_*.pth"))
    assert len(predictor_files) > 0


def test_e2e_generate(train_output_dir: Path, tmp_path: Path) -> None:
    """推論の統合テスト - 学習済みモデルを使用した推論実行"""
    trained_model_dir = train_output_dir / "trained_model"
    if not trained_model_dir.exists():
        pytest.fail("train test not completed yet")

    generate_output_dir = tmp_path / "generate_output"

    generate(
        model_dir=trained_model_dir,
        predictor_iteration=None,
        config_path=None,
        predictor_path=None,
        output_dir=generate_output_dir,
        use_gpu=False,
    )

    assert generate_output_dir.exists()
    assert (generate_output_dir / "arguments.yaml").exists()
