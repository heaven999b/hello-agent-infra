"""练习加载器：默认加载模块目录下的 exercise.py；设置 INFRA_SOLUTION=1 时加载 solution.py。

用法（在 test_exercise.py 里）：
    from codesign_sim.testing import load_exercise
    ex = load_exercise(__file__)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load_exercise(test_file: str, name: str = "exercise") -> ModuleType:
    here = Path(test_file).resolve().parent
    use_solution = os.environ.get("INFRA_SOLUTION") == "1"
    filename = "solution.py" if use_solution and name == "exercise" else f"{name}.py"
    path = here / filename
    mod_name = f"_module_{here.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
