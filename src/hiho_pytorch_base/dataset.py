"""データセット処理モジュール"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import random

import numpy
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_convert

from hiho_pytorch_base.config import DatasetConfig, DatasetFileConfig


@dataclass
class DatasetInput:
    """データセットの入力データ構造"""

    feature_vector: numpy.ndarray
    feature_variable: numpy.ndarray
    target_vector: numpy.ndarray
    target_scalar: float


@dataclass
class LazyDatasetInput:
    """遅延読み込み対応のデータセット入力"""

    feature_vector_path: Path
    feature_variable_path: Path
    target_vector_path: Path
    target_scalar_path: Path

    def generate(self):
        """ファイルからデータを読み込んでDatasetInputを生成"""
        return DatasetInput(
            feature_vector=numpy.load(self.feature_vector_path, allow_pickle=True),
            feature_variable=numpy.load(self.feature_variable_path, allow_pickle=True),
            target_vector=numpy.load(self.target_vector_path, allow_pickle=True),
            target_scalar=float(numpy.load(self.target_scalar_path, allow_pickle=True)),
        )


@dataclass
class DatasetOutput:
    """データセットの出力データ構造"""

    feature_vector: Tensor
    feature_variable: Tensor
    target_vector: Tensor
    target_scalar: Tensor


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    feature_vector: Tensor
    feature_variable: list[Tensor]
    target_vector: Tensor
    target_scalar: Tensor


def preprocess(d: DatasetInput) -> DatasetOutput:
    """前処理関数"""
    variable_scalar = numpy.mean(d.feature_variable)
    enhanced_feature = d.feature_vector + variable_scalar

    return DatasetOutput(
        feature_vector=torch.from_numpy(enhanced_feature).float(),
        feature_variable=torch.from_numpy(d.feature_variable).float(),
        target_vector=torch.from_numpy(d.target_vector).long(),
        target_scalar=torch.tensor(d.target_scalar).float(),
    )


def _load_pathlist(pathlist_path: Path, root_dir: Path) -> dict[str, Path]:
    """pathlistファイルを読み込み、stemをキー、パスを値とする辞書を返す"""
    path_list = [root_dir / p for p in pathlist_path.read_text().splitlines()]
    return {p.stem: p for p in path_list}


def get_data_paths(root_dir: Path, *pathlist_configs: tuple[str, Path]) -> tuple[list[str], dict[str, dict[str, Path]]]:
    """複数のpathlistファイルからデータパスマッピングを作成し、ファイル数の整合性をチェックする
    
    Args:
        root_dir: データファイルのルートディレクトリ
        *pathlist_configs: (データタイプ名, pathlistファイルパス) のタプル
        
    Returns:
        fn_list: ソート済みのstemリスト
        path_mappings: データタイプ名 -> {stem: Path} の辞書
    """
    if not pathlist_configs:
        raise ValueError("少なくとも1つのpathlist設定が必要です")
    
    path_mappings: dict[str, dict[str, Path]] = {}
    
    # 最初のpathlistをベースにstemリストを作成
    first_data_type, first_pathlist_path = pathlist_configs[0]
    first_paths = _load_pathlist(first_pathlist_path, root_dir / first_data_type)
    fn_list = sorted(first_paths.keys())
    assert len(fn_list) > 0, f"ファイルが存在しません: {first_pathlist_path}"
    
    path_mappings[first_data_type] = first_paths
    
    # 残りのpathlistが同じstemセットを持つかチェック
    for data_type, pathlist_path in pathlist_configs[1:]:
        paths = _load_pathlist(pathlist_path, root_dir / data_type)
        assert set(fn_list) == set(paths.keys()), (
            f"ファイルが一致しません: {data_type} (expected: {len(fn_list)}, got: {len(paths)})"
        )
        path_mappings[data_type] = paths
        
    return fn_list, path_mappings


def get_datas(config: DatasetFileConfig) -> list[LazyDatasetInput]:
    """データを取得"""
    fn_list, path_mappings = get_data_paths(
        config.root_dir,
        ("feature_vector", config.feature_vector_pathlist_path),
        ("feature_variable", config.feature_variable_pathlist_path),
        ("target_vector", config.target_vector_pathlist_path),
        ("target_scalar", config.target_scalar_pathlist_path),
    )

    datas = [
        LazyDatasetInput(
            feature_vector_path=path_mappings["feature_vector"][fn],
            feature_variable_path=path_mappings["feature_variable"][fn],
            target_vector_path=path_mappings["target_vector"][fn],
            target_scalar_path=path_mappings["target_scalar"][fn],
        )
        for fn in fn_list
    ]
    return datas


class FeatureTargetDataset(Dataset):
    """特徴量とターゲットを扱うPyTorchデータセット"""

    def __init__(
        self,
        datas: Sequence[DatasetInput | LazyDatasetInput],
    ):
        self.datas = datas

    def __len__(self):
        """データセットのサイズを返す"""
        return len(self.datas)

    def __getitem__(self, i):
        """指定されたインデックスのデータを前処理して返す"""
        data = self.datas[i]
        if isinstance(data, LazyDatasetInput):
            data = data.generate()

        return default_convert(preprocess(data))


def create_dataset(config: DatasetConfig) -> dict[str, Dataset | None]:
    """データセットを作成"""
    # TODO: accent_estimatorのようにHDF5に対応させ、docs/にドキュメントを書く
    datas = get_datas(config.train)

    if config.seed is not None:
        random.Random(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]

    def dataset_wrapper(
        datas: Sequence[DatasetInput | LazyDatasetInput], is_eval: bool
    ) -> Dataset:
        dataset = FeatureTargetDataset(datas=datas)
        if is_eval:
            dataset = ConcatDataset([dataset] * config.eval_times_num)
        return dataset

    # TODO: この出力もdataclassにする
    return {
        "train": dataset_wrapper(trains, is_eval=False),
        "test": dataset_wrapper(tests, is_eval=False),
        "eval": dataset_wrapper(tests, is_eval=True),
        "valid": (
            dataset_wrapper(get_datas(config.valid), is_eval=True)
            if config.valid is not None
            else None
        ),
    }
