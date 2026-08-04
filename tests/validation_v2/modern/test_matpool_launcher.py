import subprocess
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
LAUNCHER = REPOSITORY / "scripts" / "run_modern_imputation_matpool.sh"
SSSD_REQUIREMENTS = REPOSITORY / "requirements-modern-sssd.txt"
PACKAGER = REPOSITORY / "scripts" / "package_modern_experiment.ps1"


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


def test_sssd_environment_uses_4090_compatible_torch_and_legacy_pip():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    requirements = SSSD_REQUIREMENTS.read_text(encoding="utf-8")

    assert '"pip==24.0"' in launcher
    assert "https://download.pytorch.org/whl/cu118" in launcher
    assert '"torch==2.0.1"' in launcher
    assert not any(line.startswith("torch==") for line in requirements.splitlines())


def test_detached_session_invokes_non_executable_launcher_through_bash():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "bash scripts/run_modern_imputation_matpool.sh _run" in launcher


def test_packaged_sssd_snapshot_has_a_commit_marker_checked_by_prepare():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    packager = PACKAGER.read_text(encoding="utf-8")

    assert ".pinned-commit" in launcher
    assert ".pinned-commit" in packager
