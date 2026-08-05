import hashlib
from pathlib import Path

from scripts.download_external_datasets import (
    DATASETS,
    DatasetSpec,
    resolve_selection,
    validate_file,
)


def test_dataset_manifest_matches_published_archives():
    assert DATASETS["euroc-room1"].size == 6_042_263_426
    assert DATASETS["euroc-room1"].md5 == "5ce06b405827e453a82523d3ca9c2fd0"
    assert DATASETS["idol-building1"].size == 580_418_073
    assert DATASETS["idol-building1"].md5 == "4e676daf04e1b3db2646af34f68cd487"
    assert DATASETS["idol-building2"].size == 281_164_824
    assert DATASETS["idol-building3"].size == 259_367_888


def test_resolve_selection_expands_dataset_groups():
    assert resolve_selection(["euroc"]) == [DATASETS["euroc-room1"]]
    assert resolve_selection(["idol"]) == [
        DATASETS["idol-building1"],
        DATASETS["idol-building2"],
        DATASETS["idol-building3"],
    ]
    assert resolve_selection(["euroc", "idol-building1"]) == [
        DATASETS["euroc-room1"],
        DATASETS["idol-building1"],
    ]


def test_validate_file_checks_size_and_md5(tmp_path: Path):
    payload = b"external-dataset-test"
    archive = tmp_path / "archive.zip"
    archive.write_bytes(payload)
    spec = DatasetSpec(
        key="test",
        group="test",
        filename=archive.name,
        url="https://example.invalid/archive.zip",
        size=len(payload),
        md5=hashlib.md5(payload).hexdigest(),
    )

    assert validate_file(archive, spec)

    archive.write_bytes(payload + b"corrupt")
    assert not validate_file(archive, spec)
