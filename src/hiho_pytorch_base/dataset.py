from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_convert

from hiho_pytorch_base.config import DatasetConfig, DatasetFileConfig


@dataclass
class DatasetInput:
    feature_vector: numpy.ndarray
    feature_variable: numpy.ndarray
    target_vector: numpy.ndarray
    target_scalar: float


@dataclass
class LazyDatasetInput:
    feature_vector_path: Path
    feature_variable_path: Path
    target_vector_path: Path
    target_scalar_path: Path

    def generate(self):
        return DatasetInput(
            feature_vector=numpy.load(self.feature_vector_path, allow_pickle=True),
            feature_variable=numpy.load(self.feature_variable_path, allow_pickle=True),
            target_vector=numpy.load(self.target_vector_path, allow_pickle=True),
            target_scalar=float(numpy.load(self.target_scalar_path, allow_pickle=True)),
        )


@dataclass
class DatasetOutput:
    feature_vector: Tensor
    feature_variable: Tensor
    target_vector: Tensor
    target_scalar: Tensor


@dataclass
class BatchOutput:
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


def _load_pathlist(path: Path, root_dir: Path) -> dict[str, Path]:
    """pathlistファイルから辞書を作成"""
    path_list = [root_dir / p for p in path.read_text().splitlines()]
    return {p.stem: p for p in path_list}


def get_datas(config: DatasetFileConfig) -> list[LazyDatasetInput]:
    """データを取得"""
    # TODO: 過去の３つのプロジェクトに合わせ、ファイル数をassertするべき

    feature_vector_paths = _load_pathlist(config.feature_pathlist_path, config.root_dir / "feature_vector")
    fn_list = sorted(feature_vector_paths.keys())
    assert len(fn_list) > 0

    target_vector_paths = _load_pathlist(config.target_pathlist_path, config.root_dir / "target_vector")
    
    target_fn_list = sorted(target_vector_paths.keys())
    assert set(fn_list) == set(target_fn_list), f"Feature files: {set(fn_list)}, Target files: {set(target_fn_list)}"
    
    feature_variable_paths = {fn: config.root_dir / "feature_variable" / f"{fn}.npy" for fn in fn_list}
    target_scalar_paths = {fn: config.root_dir / "target_scalar" / f"{fn}.npy" for fn in fn_list}

    datas = [
        LazyDatasetInput(
            feature_vector_path=feature_vector_paths[fn],
            feature_variable_path=feature_variable_paths[fn],
            target_vector_path=target_vector_paths[fn],
            target_scalar_path=target_scalar_paths[fn],
        )
        for fn in fn_list
    ]
    return datas


class FeatureTargetDataset(Dataset):
    def __init__(
        self,
        datas: Sequence[DatasetInput | LazyDatasetInput],
    ):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        if isinstance(data, LazyDatasetInput):
            data = data.generate()

        return default_convert(preprocess(data))


def create_dataset(config: DatasetConfig) -> dict[str, Dataset | None]:
    """データセットを作成"""
    datas = get_datas(config.train_file)

    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(datas)  # type: ignore

    tests, trains = datas[: config.test_num], datas[config.test_num :]

    def dataset_wrapper(
        datas: Sequence[DatasetInput | LazyDatasetInput], is_eval: bool
    ) -> Dataset:
        dataset = FeatureTargetDataset(datas=datas)
        if is_eval:
            dataset = ConcatDataset([dataset] * config.eval_times_num)
        return dataset

    valid_dataset = None
    if config.valid_file is not None:
        valids = get_datas(config.valid_file)
        valid_dataset = dataset_wrapper(valids, is_eval=True)

    # TODO: この出力もdataclassにする
    return {
        "train": dataset_wrapper(trains, is_eval=False),
        "test": dataset_wrapper(tests, is_eval=False),
        "eval": dataset_wrapper(tests, is_eval=True),
        "valid": valid_dataset,
    }
