from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_runner(script: str, workspace_config: str | None = None):
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if workspace_config is not None:
        environment["CUBLAS_WORKSPACE_CONFIG"] = workspace_config
    environment["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )


def test_runner_import_configures_deterministic_cublas_workspace():
    result = _import_runner(
        "import os; "
        "import validation_v2.experiments.runner; "
        "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ":4096:8"


def test_runner_import_preserves_supported_cublas_workspace():
    result = _import_runner(
        "import os; "
        "import validation_v2.experiments.runner; "
        "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])",
        workspace_config=":16:8",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ":16:8"


def test_runner_import_rejects_unsupported_cublas_workspace():
    result = _import_runner(
        "import validation_v2.experiments.runner",
        workspace_config=":invalid",
    )

    assert result.returncode != 0
    assert "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8" in result.stderr


def test_set_seed_allows_deterministic_cuda_linear_when_available():
    result = _import_runner(
        "import os; "
        "from validation_v2.experiments import runner; "
        "import torch; "
        "runner._set_seed(2026); "
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); "
        "layer = torch.nn.Linear(4, 2).to(device); "
        "inputs = torch.randn(3, 4, device=device, requires_grad=True); "
        "layer(inputs).sum().backward(); "
        "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ":4096:8"
