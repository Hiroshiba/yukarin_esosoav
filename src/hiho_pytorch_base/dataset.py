"""データセットモジュール"""

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import assert_never

from pydantic import TypeAdapter
from torch.utils.data import Dataset as BaseDataset
from upath import UPath

from hiho_pytorch_base.config import DataFileConfig, DatasetConfig
from hiho_pytorch_base.data.data import (
    InputData,
    OutputData,
    preprocess,
)
from hiho_pytorch_base.data.phoneme import ArpaPhoneme
from hiho_pytorch_base.data.sampling_data import SamplingData
from hiho_pytorch_base.utility.upath_utility import to_local_path


@dataclass
class LazyInputData:
    """遅延読み込み対応の入力データ構造"""

    f0_path: UPath
    volume_path: UPath
    lab_path: UPath
    silence_path: UPath
    spec_path: UPath  # NOTE: 対数メルスペクトログラム
    wave_path: UPath
    speaker_id: int

    def fetch(self) -> InputData:
        """ファイルからデータを読み込んでDatasetInputを生成"""
        return InputData(
            phonemes=ArpaPhoneme.load_julius_list(to_local_path(self.lab_path)),
            f0_data=SamplingData.load(to_local_path(self.f0_path)),
            volume_data=SamplingData.load(to_local_path(self.volume_path)),
            silence_data=SamplingData.load(to_local_path(self.silence_path)),
            spec_data=SamplingData.load(to_local_path(self.spec_path)),
            wave_data=SamplingData.load(to_local_path(self.wave_path)),
            speaker_id=self.speaker_id,
        )


def prefetch_datas(datas: list[LazyInputData], num_prefetch: int) -> None:
    """データセットを前もって読み込む"""
    if num_prefetch <= 0:
        return

    # TODO: これだとメインがエラーで落ちてもスレッドの完了を待ってしまうので、threading.Thread(daemon=True)に変えたい
    with ThreadPoolExecutor(max_workers=num_prefetch) as executor:
        for data in datas:
            executor.submit(data.fetch)


class Dataset(BaseDataset[OutputData]):
    """メインのデータセット"""

    def __init__(
        self,
        datas: list[LazyInputData],
        config: DatasetConfig,
        is_eval: bool,
    ):
        self.datas = datas
        self.config = config
        self.is_eval = is_eval

    def __len__(self):
        """データセットのサイズ"""
        return len(self.datas)

    def __getitem__(self, i: int) -> OutputData:
        """指定されたインデックスのデータを前処理して返す"""
        try:
            return preprocess(
                self.datas[i].fetch(),
                prepost_silence_frame_length=self.config.prepost_silence_frame_length,
                max_frame_length=self.config.max_frame_length,
                wave_frame_length=self.config.wave_frame_length,
                is_eval=self.is_eval,
            )
        except Exception as e:
            raise RuntimeError(
                f"データ処理に失敗しました: index={i} data={self.datas[i]}"
            ) from e


class DatasetType(str, Enum):
    """データセットタイプ"""

    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"
    VALID = "valid"


@dataclass
class DatasetCollection:
    """データセットコレクション"""

    train: Dataset
    """重みの更新に用いる"""

    test: Dataset
    """trainと同じドメインでモデルの過適合確認に用いる"""

    eval: Dataset | None
    """testと同じデータを評価に用いる"""

    valid: Dataset | None
    """trainやtestと異なり、評価専用に用いる"""

    def get(self, type: DatasetType) -> Dataset:
        """指定されたタイプのデータセットを返す"""
        match type:
            case DatasetType.TRAIN:
                return self.train
            case DatasetType.TEST:
                return self.test
            case DatasetType.EVAL:
                if self.eval is None:
                    raise ValueError("evalデータセットが設定されていません")
                return self.eval
            case DatasetType.VALID:
                if self.valid is None:
                    raise ValueError("validデータセットが設定されていません")
                return self.valid
            case _:
                assert_never(type)


PathMap = dict[str, UPath]
"""パスマップ。stemをキー、パスを値とする辞書型"""


def _load_pathlist(pathlist_path: UPath, root_dir: UPath) -> PathMap:
    """pathlistファイルを読み込みんでパスマップを返す。"""
    path_list = [root_dir / p for p in pathlist_path.read_text().splitlines()]
    return {p.stem: p for p in path_list}


def get_data_paths(
    root_dir: UPath | None, pathlist_paths: list[UPath]
) -> tuple[list[str], list[PathMap]]:
    """複数のpathlistファイルからstemリストとパスマップを返す。整合性も確認する。"""
    if len(pathlist_paths) == 0:
        raise ValueError("少なくとも1つのpathlist設定が必要です")

    if root_dir is None:
        root_dir = UPath(".")

    path_mappings: list[PathMap] = []

    # 最初のpathlistをベースにstemリストを作成
    first_pathlist_path = pathlist_paths[0]
    first_paths = _load_pathlist(first_pathlist_path, root_dir)
    fn_list = list(first_paths.keys())
    assert len(fn_list) > 0, f"ファイルが存在しません: {first_pathlist_path}"

    path_mappings.append(first_paths)

    # 残りのpathlistが同じstemリストを持つかチェック
    for pathlist_path in pathlist_paths[1:]:
        paths = _load_pathlist(pathlist_path, root_dir)
        assert fn_list == list(paths.keys()), (
            f"ファイルが一致しません: {pathlist_path} (expected: {len(fn_list)}, got: {len(paths)})"
        )
        path_mappings.append(paths)

    return fn_list, path_mappings


def get_datas(config: DataFileConfig) -> list[LazyInputData]:
    """データを取得"""
    (
        fn_list,
        (
            f0_pathmappings,
            volume_pathmappings,
            lab_pathmappings,
            silence_pathmappings,
            spec_pathmappings,
            wave_pathmappings,
        ),
    ) = get_data_paths(
        config.root_dir,
        [
            config.f0_pathlist_path,
            config.volume_pathlist_path,
            config.lab_pathlist_path,
            config.silence_pathlist_path,
            config.spec_pathlist_path,
            config.wave_pathlist_path,
        ],
    )

    fn_each_speaker = TypeAdapter(dict[str, list[str]]).validate_json(
        config.speaker_dict_path.read_text()
    )
    speaker_ids = {
        fn: speaker_id
        for speaker_id, fns in enumerate(fn_each_speaker.values())
        for fn in fns
    }

    datas = [
        LazyInputData(
            f0_path=f0_pathmappings[fn],
            volume_path=volume_pathmappings[fn],
            lab_path=lab_pathmappings[fn],
            silence_path=silence_pathmappings[fn],
            spec_path=spec_pathmappings[fn],
            wave_path=wave_pathmappings[fn],
            speaker_id=speaker_ids[fn],
        )
        for fn in fn_list
    ]
    return datas


def create_dataset(config: DatasetConfig) -> DatasetCollection:
    """データセットを作成"""
    # TODO: accent_estimatorのようにHDF5に対応させ、docs/にドキュメントを書く
    datas = get_datas(config.train)

    if config.seed is not None:
        random.Random(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]
    if config.train_num is not None:
        trains = trains[: config.train_num]

    def _wrapper(datas: list[LazyInputData], is_eval: bool) -> Dataset:
        if is_eval:
            datas = datas * config.eval_times_num
        dataset = Dataset(datas=datas, config=config, is_eval=is_eval)
        return dataset

    return DatasetCollection(
        train=_wrapper(trains, is_eval=False),
        test=_wrapper(tests, is_eval=False),
        eval=(_wrapper(tests, is_eval=True) if config.eval_for_test else None),
        valid=(
            _wrapper(get_datas(config.valid), is_eval=True)
            if config.valid is not None
            else None
        ),
    )
