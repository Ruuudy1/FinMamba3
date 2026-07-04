"""Standalone in-place build for the native C++ backtester engine (Phase 2 accelerator).

This is intentionally NOT part of ``pip install -e .`` so the package installs with no compiler; the
native path is opt-in. Build it explicitly with:

    python -m finmamba3.backtester.engine_cpp.build

It compiles ``_engine.cpp`` into ``_engine.<abi>.pyd`` next to this file via pybind11 and the local C++
toolchain (MSVC on Windows, gcc in the CUDA Docker image).
"""
# region imports
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Distribution
# endregion


def build() -> int:
    here = Path(__file__).resolve().parent
    extension = Pybind11Extension("_engine", [str(here / "_engine.cpp")], cxx_std=17)
    distribution = Distribution({"name": "finmamba3_engine_cpp", "ext_modules": [extension]})
    command = build_ext(distribution)
    command.ensure_finalized()
    # Write the built module straight into this package directory (no inplace path juggling).
    command.build_lib = str(here)
    command.build_temp = str(here / "_build_tmp")
    command.run()
    built = list(here.glob("_engine*.pyd")) + list(here.glob("_engine*.so"))
    print(f"built: {[str(path.name) for path in built]}")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(build())
