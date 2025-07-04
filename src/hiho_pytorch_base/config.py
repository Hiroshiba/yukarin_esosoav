from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from hiho_pytorch_base.utility import dataclass_utility
from hiho_pytorch_base.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    feature_glob: str
    target_glob: str
    test_num: int
    eval_times_num: int = 1
    seed: int = 0


@dataclass
class NetworkConfig:
    pass


@dataclass
class ModelConfig:
    pass


@dataclass
class TrainConfig:
    batch_size: int
    eval_batch_size: int | None
    log_iteration: int
    eval_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: dict[str, Any]
    weight_initializer: str | None = None
    num_processes: int | None = None
    use_gpu: bool = True
    use_amp: bool = False
    use_multithread: bool = False


@dataclass
class ProjectConfig:
    name: str
    tags: dict[str, Any] = field(default_factory=dict)
    category: str | None = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: dict[str, Any]):
    pass
