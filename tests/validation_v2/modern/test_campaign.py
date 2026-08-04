from pathlib import Path

import pytest

from validation_v2.modern.campaign import claim_task, complete_task, pending_tasks


def test_completed_hash_matching_task_is_not_pending(tmp_path: Path):
    task = {"task_id": "a" * 64, "model": "brits", "seed": 2026}
    claim_task(tmp_path, task)
    complete_task(tmp_path, task, {"checkpoint_sha256": "b" * 64})
    assert pending_tasks(tmp_path, [task]) == ()


def test_inconsistent_completed_task_is_rejected(tmp_path: Path):
    task = {"task_id": "a" * 64, "model": "brits", "seed": 2026}
    claim_task(tmp_path, task)
    complete_task(tmp_path, task, {"checkpoint_sha256": "b" * 64})
    (tmp_path / task["task_id"] / "completed.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="inconsistent completed task"):
        pending_tasks(tmp_path, [task])
