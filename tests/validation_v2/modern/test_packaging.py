from pathlib import Path

from validation_v2.modern.cli import package_result_tree


def test_summary_excludes_checkpoints_and_samples(tmp_path: Path):
    source = tmp_path / "campaign"
    (source / "summary").mkdir(parents=True)
    (source / "summary" / "summary.csv").write_text("x\n", encoding="utf-8")
    (source / "checkpoints").mkdir()
    (source / "checkpoints" / "model.pt").write_bytes(b"weights")
    (source / "samples").mkdir()
    (source / "samples" / "samples.npz").write_bytes(b"samples")
    manifest = package_result_tree(source, tmp_path / "result", mode="summary")
    assert "summary/summary.csv" in manifest["files"]
    assert all(
        "checkpoints/" not in name and "samples/" not in name
        for name in manifest["files"]
    )
