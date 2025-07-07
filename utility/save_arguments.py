"""関数の引数をYAMLファイルに保存するユーティリティ"""

import inspect
from collections.abc import Callable
from pathlib import Path, PosixPath, WindowsPath
from typing import Any

import yaml


def _path_represent(dumper, data):
    return dumper.represent_str(str(data))


yaml.SafeDumper.add_representer(PosixPath, _path_represent)
yaml.SafeDumper.add_representer(WindowsPath, _path_represent)


def save_arguments(path: Path, target_function: Callable, arguments: dict[str, Any]):
    """対象関数の引数のみを抽出してYAMLファイルに保存"""
    args = inspect.getfullargspec(target_function).args
    obj = {k: v for k, v in arguments.items() if k in args}

    with path.open(mode="w") as f:
        yaml.safe_dump(obj, f)
