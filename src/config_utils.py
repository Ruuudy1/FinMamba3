"""Shared configuration helpers for LOB pretraining."""

from __future__ import annotations

import argparse
import ast
from typing import Sequence

import torch


class DotDict(dict):
    """Dictionary with dot notation access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(f"'DotDict' object has no attribute '{item}'") from exc
        if isinstance(value, dict):
            value = DotDict(value)
            self[item] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def update_or_create(self, key_path, value):
        keys = key_path.split(".")
        d = self
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = DotDict()
            d = d[key]
        d[keys[-1]] = value


def _dtype_mapper(dtype_value):
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtype_map[str(dtype_value)]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"Unknown torch dtype {dtype_value!r}; expected one of {sorted(dtype_map)}"
        ) from exc


def _bool_mapper(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ["true", "1", "yes", "y", "on"]


def parse_args_and_update_config(
    config,
    prefix: str = "",
    argv: Sequence[str] | None = None,
):
    """Register dotted CLI overrides for every scalar config leaf."""

    parser = argparse.ArgumentParser(allow_abbrev=False)

    def add_arguments(node, current_prefix=""):
        for key, value in node.items():
            arg_name = f"--{current_prefix}{key}"
            if isinstance(value, dict):
                add_arguments(value, current_prefix + key + ".")
            elif isinstance(value, bool):
                parser.add_argument(arg_name, type=_bool_mapper, default=value)
            elif key == "dtype":
                default = _dtype_mapper(value)
                parser.add_argument(arg_name, type=_dtype_mapper, default=default)
            elif isinstance(value, (list, dict)):
                parser.add_argument(
                    arg_name,
                    type=lambda x: ast.literal_eval(x),
                    default=value,
                )
            else:
                parser.add_argument(arg_name, type=type(value), default=value)

    def update_dict(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    add_arguments(config, prefix)
    args = parser.parse_args(argv)

    for arg_key, arg_value in vars(args).items():
        if arg_value is not None:
            update_dict(config, arg_key.split("."), arg_value)

    return config
