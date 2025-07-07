"""機械学習プロジェクトの設定管理モジュール"""

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from hiho_pytorch_base.utility.git_utility import get_branch_name, get_commit_id


class DatasetFileConfig(BaseModel):
    """データセットファイルパスの設定"""

    feature_vector_pathlist_path: Path
    feature_variable_pathlist_path: Path
    target_vector_pathlist_path: Path
    target_scalar_pathlist_path: Path
    root_dir: Path


class DatasetConfig(BaseModel):
    """データセット全体の設定"""

    train: DatasetFileConfig
    valid: DatasetFileConfig | None = None
    test_num: int
    eval_times_num: int = 1
    seed: int = 0


class NetworkConfig(BaseModel):
    """ニューラルネットワーク構造の設定"""

    input_size: int
    hidden_size: int
    output_size: int


class ModelConfig(BaseModel):
    """モデル固有の設定"""

    pass


class TrainConfig(BaseModel):
    """学習パラメータの設定"""

    batch_size: int
    eval_batch_size: int
    log_epoch: int
    eval_epoch: int
    snapshot_epoch: int
    stop_epoch: int
    model_save_num: int
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None = None
    weight_initializer: str | None = None
    pretrained_predictor_path: Path | None = None
    num_processes: int = 0
    use_gpu: bool = True
    use_amp: bool = True


class ProjectConfig(BaseModel):
    """プロジェクト情報の設定"""

    name: str
    tags: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None


class Config(BaseModel):
    """機械学習プロジェクトの全設定"""

    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """辞書から設定オブジェクトを作成"""
        backward_compatible(d)
        return cls.model_validate(d)

    def to_dict(self) -> dict[str, Any]:
        """設定オブジェクトを辞書に変換"""
        return self.model_dump(mode="json")

    def add_git_info(self) -> None:
        """Git情報をプロジェクトタグに追加"""
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: dict[str, Any]) -> None:
    """設定の後方互換性を保つための変換"""
    pass
