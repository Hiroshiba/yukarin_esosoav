from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from hiho_pytorch_base.utility.git_utility import get_branch_name, get_commit_id


class DatasetFileConfig(BaseModel):
    feature_pathlist_path: Path
    target_pathlist_path: Path
    root_dir: Path


class DatasetConfig(BaseModel):
    train_file: DatasetFileConfig
    test_num: int
    valid_file: DatasetFileConfig | None = None
    eval_times_num: int = 1
    seed: int = 0


class NetworkConfig(BaseModel):
    input_size: int
    hidden_size: int
    output_size: int


class ModelConfig(BaseModel):
    pass


class TrainConfig(BaseModel):
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
    num_processes: int = 4
    use_gpu: bool = True
    use_amp: bool = True


class ProjectConfig(BaseModel):
    name: str
    tags: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None


class Config(BaseModel):
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        backward_compatible(d)
        return cls.model_validate(d)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def add_git_info(self) -> None:
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: dict[str, Any]) -> None:
    pass
