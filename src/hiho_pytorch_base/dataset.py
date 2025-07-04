from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_convert

from hiho_pytorch_base.config import DatasetConfig, DatasetFileConfig


@dataclass
class DatasetInput:
    feature: numpy.ndarray
    target: numpy.ndarray


@dataclass
class LazyDatasetInput:
    feature_path: Path
    target_path: Path

    def generate(self):
        return DatasetInput(
            feature=numpy.load(self.feature_path, allow_pickle=True),
            target=numpy.load(self.target_path, allow_pickle=True),
        )


class DatasetOutput(TypedDict):
    feature: Tensor
    target: Tensor


def preprocess(d: DatasetInput) -> DatasetOutput:
    """前処理関数"""
    output_data = DatasetOutput(
        feature=torch.from_numpy(d.feature).float(),
        target=torch.from_numpy(d.target).long(),
    )
    return output_data


def _load_pathlist(path: Path, root_dir: Path) -> dict[str, Path]:
    """pathlistファイルから辞書を作成"""
    path_list = [root_dir / p for p in path.read_text().splitlines()]
    return {p.stem: p for p in path_list}


def get_datas(config: DatasetFileConfig) -> list[LazyDatasetInput]:
    """データを取得"""
    feature_paths = _load_pathlist(config.feature_pathlist_path, config.root_dir)
    fn_list = sorted(feature_paths.keys())
    assert len(fn_list) > 0

    target_paths = _load_pathlist(config.target_pathlist_path, config.root_dir)
    assert set(fn_list) == set(target_paths.keys())

    datas = [
        LazyDatasetInput(
            feature_path=feature_paths[fn],
            target_path=target_paths[fn],
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


def create_dataset(config: DatasetConfig):
    """データセットを作成"""
    datas = get_datas(config.train_file)

    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(datas)  # type: ignore

    tests, trains = datas[: config.test_num], datas[config.test_num :]

    def dataset_wrapper(datas, is_eval: bool):
        dataset = FeatureTargetDataset(datas=datas)
        if is_eval:
            dataset = ConcatDataset([dataset] * config.eval_times_num)
        return dataset

    # バリデーションデータセット
    valid_dataset = None
    if config.valid_file is not None:
        valids = get_datas(config.valid_file)
        valid_dataset = dataset_wrapper(valids, is_eval=True)

    return {
        "train": dataset_wrapper(trains, is_eval=False),
        "test": dataset_wrapper(tests, is_eval=False),
        "eval": dataset_wrapper(tests, is_eval=True),
        "valid": valid_dataset,
    }
