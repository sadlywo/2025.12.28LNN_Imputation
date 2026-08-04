import subprocess

import pytest


def test_launcher_help_lists_complete_lifecycle():
    probe = subprocess.run(["bash", "--version"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("a functional Bash runtime is not installed")
    result = subprocess.run(
        ["bash", "scripts/run_modern_imputation_matpool.sh", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    for command in ("prepare", "start", "status", "logs", "resume", "package-results"):
        assert command in result.stdout
