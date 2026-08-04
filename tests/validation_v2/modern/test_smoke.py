from pathlib import Path

from validation_v2.modern.cli import main


def test_cli_plan_emits_reference_and_modern_tasks(tmp_path: Path):
    result = main(
        [
            "plan",
            "--config",
            "configs/validation_v2/modern_smoke.yaml",
            "--output",
            str(tmp_path),
        ]
    )
    assert result == 0
    text = (tmp_path / "campaign-plan.json").read_text(encoding="utf-8")
    assert '"hybrid"' in text and '"brits"' in text and '"sssd"' in text
