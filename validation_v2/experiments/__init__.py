"""Reproducible experiment provenance and aggregation."""

from .evaluate import evaluate_test_once
from .matrix import enumerate_matrix
from .provenance import collect_provenance, run_id, write_run_manifest
from .runner import run_matrix, run_smoke
from .summarize import summarize_runs
from .train import resume_run, select_best_checkpoint, train_one_run

__all__ = [
    "collect_provenance",
    "enumerate_matrix",
    "evaluate_test_once",
    "resume_run",
    "run_id",
    "run_matrix",
    "run_smoke",
    "select_best_checkpoint",
    "summarize_runs",
    "train_one_run",
    "write_run_manifest",
]
