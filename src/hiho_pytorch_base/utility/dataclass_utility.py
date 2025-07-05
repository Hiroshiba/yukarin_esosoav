import dataclasses
import types
from pathlib import Path
from typing import Any, get_args, get_origin


def convert_to_dict(data: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(data):
        data = dataclasses.asdict(data)
    for key, val in data.items():
        if isinstance(val, Path):
            data[key] = str(val)
        if isinstance(val, dict):
            data[key] = convert_to_dict(val)
    return data


def _get_non_none_type(field_type: Any) -> Any:
    """X | Y 記法から、None以外の型を取得する"""
    if get_origin(field_type) is types.UnionType:
        args = get_args(field_type)
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            return non_none_types[0]
    return field_type


def convert_from_dict(cls: Any, data: dict[str, Any]) -> Any:
    if data is None:
        data = {}

    converted_data = {}
    for key, val in data.items():
        field_type = cls.__dataclass_fields__[key].type
        actual_type = _get_non_none_type(field_type)
        if val is None:
            converted_data[key] = None
        elif actual_type == Path:
            converted_data[key] = Path(val)
        elif dataclasses.is_dataclass(actual_type):
            converted_data[key] = convert_from_dict(actual_type, val)
        else:
            converted_data[key] = val

    return cls(**converted_data)
