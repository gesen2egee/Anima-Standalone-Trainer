import importlib.util
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAME_PATH = PROJECT_ROOT / "library" / "came.py"


def load_came_class():
    spec = importlib.util.spec_from_file_location("standalone_came", CAME_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CAME


def test_came_has_only_standalone_imports():
    lines = CAME_PATH.read_text(encoding="utf-8").splitlines()
    imports = [
        line
        for line in lines
        if line.startswith("import ") or line.startswith("from ")
    ]

    assert imports == [
        "import math",
        "from typing import Tuple",
        "import torch",
    ]


def test_came_can_be_imported_from_library_package():
    from library.came import CAME

    assert CAME.__module__ == "library.came"


def test_came_accepts_use_8bit_arg_and_keeps_state_quantized():
    CAME = load_came_class()
    param = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
    optimizer = CAME(
        [param],
        lr=1e-3,
        weight_decay=0.0,
        use_magma=False,
        use_kahan=False,
        use_8bit=True,
        min_8bit_size=1,
    )

    param.grad = torch.ones_like(param)
    optimizer.step()

    state = optimizer.state[param]
    assert optimizer.param_groups[0]["use_8bit"] is True
    assert state["exp_avg"].dtype == torch.uint8
    assert "exp_avg_8bit_scale" in state

    param.grad = torch.full_like(param, 0.5)
    optimizer.step()

    assert torch.isfinite(param).all()
    assert optimizer.state[param]["exp_avg"].dtype == torch.uint8
