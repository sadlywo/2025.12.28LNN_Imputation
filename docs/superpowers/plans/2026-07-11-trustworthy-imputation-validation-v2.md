# Trustworthy IMU Imputation Validation V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a leakage-free, time-correct, provenance-tracked IMU imputation validation pipeline with local smoke tests and a server-ready five-seed experiment matrix.

**Architecture:** Add an isolated validation_v2 package rather than modifying legacy experiments. Separate raw recording loading, split/scaler state, masking/features, model timing, objectives, trajectory evaluation, experiment orchestration, and artifact validation so each scientific assumption is independently testable.

**Tech Stack:** Python 3.9, PyTorch 2.3.1, ncps/CfC, NumPy, pandas, SciPy Rotation/Slerp, PyYAML, pytest, conda environment pinn_imu.

---

## File Map

New production files:

- validation_v2/types.py: immutable recording, split, scaler, mask, and run metadata.
- validation_v2/config.py: typed YAML configuration.
- validation_v2/data/{oxiod,splits,normalization,masking,features,windows}.py.
- validation_v2/models/{bilnn,bilstm,hybrid,baselines}.py.
- validation_v2/objectives/{reconstruction,kinematic}.py.
- validation_v2/evaluation/{synchronization,trajectory,reconstruction,statistics}.py.
- validation_v2/experiments/{provenance,train,evaluate,matrix,summarize,validate_artifacts}.py.
- validation_v2/cli.py.
- configs/validation_v2/{smoke,server_full}.yaml.
- requirements-validation-v2.txt.
- docs/validation_v2_server_runbook.md.

New tests:

- tests/validation_v2/test_config.py.
- tests/validation_v2/test_oxiod.py.
- tests/validation_v2/test_splits_and_scaler.py.
- tests/validation_v2/test_masking_and_features.py.
- tests/validation_v2/test_models.py.
- tests/validation_v2/test_objectives.py.
- tests/validation_v2/test_synchronization_and_trajectory.py.
- tests/validation_v2/test_provenance_and_statistics.py.
- tests/validation_v2/test_train_evaluate.py.
- tests/validation_v2/test_cli_smoke.py.
- tests/validation_v2/test_server_handoff.py.

## Task 1: Establish package, typed configuration, and test harness

**Files:**

- Create: validation_v2/__init__.py
- Create: validation_v2/types.py
- Create: validation_v2/config.py
- Create: tests/validation_v2/test_config.py
- Create: requirements-validation-v2.txt
- Create: .gitignore

- [ ] **Step 1: Write the failing configuration test**

    from pathlib import Path
    from validation_v2.config import load_config

    def test_config_rejects_test_tuning(tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text("selection_split: test\nseeds: [2026]\n", encoding="utf-8")
        try:
            load_config(path)
        except ValueError as exc:
            assert "selection_split must be validation" in str(exc)
        else:
            raise AssertionError("test-based selection must be rejected")

- [ ] **Step 2: Run RED**

Run:

    conda run -n pinn_imu python -m pytest tests/validation_v2/test_config.py -q

Expected: collection fails because validation_v2.config does not exist.

- [ ] **Step 3: Implement typed config loading**

    @dataclass(frozen=True)
    class ExperimentConfig:
        data_root: Path
        output_root: Path
        selection_split: str
        seeds: tuple[int, ...]
        seq_len: int
        batch_size: int
        epochs: int

    def load_config(path: Path) -> ExperimentConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("selection_split", "validation") != "validation":
            raise ValueError("selection_split must be validation")
        seeds = tuple(int(seed) for seed in raw["seeds"])
        if not seeds:
            raise ValueError("at least one seed is required")
        return ExperimentConfig(
            data_root=Path(raw["data_root"]),
            output_root=Path(raw["output_root"]),
            selection_split="validation",
            seeds=seeds,
            seq_len=int(raw["seq_len"]),
            batch_size=int(raw["batch_size"]),
            epochs=int(raw["epochs"]),
        )

- [ ] **Step 4: Add dependencies and ignores**

requirements-validation-v2.txt contains:

    numpy==1.26.4
    pandas==2.3.3
    scipy==1.13.1
    PyYAML==6.0.3
    pytest==8.4.2
    torch==2.3.1
    ncps==1.0.1

The local PyTorch build is 2.3.1+cu121; CUDA installation remains an environment-level server concern rather than a pip local-version suffix. .gitignore contains .worktrees/, results/validation_v2/, __pycache__/, and .pytest_cache/.

- [ ] **Step 5: Run GREEN and commit**

    conda run -n pinn_imu python -m pytest tests/validation_v2/test_config.py -q
    git add .gitignore requirements-validation-v2.txt validation_v2 tests/validation_v2/test_config.py
    git commit -m "test: establish validation v2 configuration contract"

## Task 2: Load OxIOD recordings with explicit time units

**Files:**

- Create: validation_v2/data/__init__.py
- Create: validation_v2/data/oxiod.py
- Test: tests/validation_v2/test_oxiod.py

- [ ] **Step 1: Write failing timestamp and overlap tests**

    def test_vicon_nanoseconds_are_converted_to_seconds(tmp_path):
        imu, vi = write_synthetic_pair(
            tmp_path,
            imu_t0=1_496_760_699.22,
            vi_t0_ns=1_496_760_699_220_000_000,
        )
        recording = load_recording(imu, vi)
        assert recording.vicon_time_s[0] == pytest.approx(recording.imu_time_s[0])
        assert np.all(np.diff(recording.vicon_time_s) > 0)

    def test_loader_rejects_empty_time_overlap(tmp_path):
        imu, vi = write_nonoverlapping_pair(tmp_path)
        with pytest.raises(ValueError, match="no IMU/Vicon overlap"):
            load_recording(imu, vi)

- [ ] **Step 2: Run RED**

    conda run -n pinn_imu python -m pytest tests/validation_v2/test_oxiod.py -q

Expected: import failure for validation_v2.data.oxiod.

- [ ] **Step 3: Implement raw parsing and unit validation**

    def _time_to_seconds(values: np.ndarray, stream: str) -> np.ndarray:
        median = float(np.median(np.abs(values)))
        if stream == "vicon" and median > 1e15:
            values = values / 1e9
        if not np.all(np.diff(values) > 0):
            raise ValueError(f"{stream} timestamps must be strictly increasing")
        return values.astype(np.float64)

    def overlapping_interval(imu_t, vicon_t):
        start = max(float(imu_t[0]), float(vicon_t[0]))
        end = min(float(imu_t[-1]), float(vicon_t[-1]))
        if start >= end:
            raise ValueError("no IMU/Vicon overlap")
        return start, end

Parse Vicon quaternion columns as x, y, z, w, matching Oxford Dataset/ReadMe.txt.

- [ ] **Step 4: Add a real-file regression test**

Load handbag-1/imu1.csv and vi1.csv. Assert all associated query times are inside the Vicon range and interpolated positions contain more than one unique row.

- [ ] **Step 5: Run GREEN and commit**

    conda run -n pinn_imu python -m pytest tests/validation_v2/test_oxiod.py -q
    git add validation_v2/data tests/validation_v2/test_oxiod.py
    git commit -m "fix: normalize OxIOD timestamp units"

## Task 3: Create disjoint manifests and train-only normalization

**Files:**

- Create: validation_v2/data/splits.py
- Create: validation_v2/data/normalization.py
- Test: tests/validation_v2/test_splits_and_scaler.py

- [ ] **Step 1: Write failing split and scaler tests**

    def test_manifest_has_no_recording_overlap(recording_index):
        manifest = stratified_file_split(recording_index, seed=2026)
        train = set(manifest.query("split == 'train'").recording_id)
        val = set(manifest.query("split == 'validation'").recording_id)
        test = set(manifest.query("split == 'test'").recording_id)
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

    def test_scaler_refuses_non_train_recordings(train_recording, test_recording):
        with pytest.raises(ValueError, match="fit accepts train recordings only"):
            RobustTrainScaler.fit(
                [train_recording, test_recording],
                allowed_ids={train_recording.id},
            )

- [ ] **Step 2: Run RED**

Run pytest on tests/validation_v2/test_splits_and_scaler.py.

- [ ] **Step 3: Implement deterministic manifests**

Implement stratified_file_split and leave_one_scenario_out. Each row stores recording_id, scenario, paths, split, and file SHA256.

- [ ] **Step 4: Implement train-only robust scaling**

    @dataclass(frozen=True)
    class RobustTrainScaler:
        center: np.ndarray
        scale: np.ndarray
        training_ids: tuple[str, ...]

        @classmethod
        def fit(cls, recordings, allowed_ids):
            ids = {recording.id for recording in recordings}
            if not ids <= set(allowed_ids):
                raise ValueError("fit accepts train recordings only")
            values = np.concatenate([r.imu_six for r in recordings], axis=0)
            center = np.median(values, axis=0)
            mad = np.median(np.abs(values - center), axis=0)
            return cls(center, np.maximum(1.4826 * mad, 1e-6), tuple(sorted(ids)))

- [ ] **Step 5: Run GREEN and commit**

Commit with message fix: enforce split isolation and train-only scaling.

## Task 4: Generate missingness and leakage-safe features

**Files:**

- Create: validation_v2/data/masking.py
- Create: validation_v2/data/features.py
- Create: validation_v2/data/windows.py
- Test: tests/validation_v2/test_masking_and_features.py

- [ ] **Step 1: Write the failing leakage-invariance test**

    def test_hidden_targets_cannot_change_model_input():
        target_a = torch.arange(60, dtype=torch.float32).reshape(10, 6)
        target_b = target_a.clone()
        mask = torch.ones_like(target_a)
        mask[3:7, 2:5] = 0
        target_b[mask == 0] += 10_000
        input_a = build_features(target_a, mask, torch.full((10,), 0.01))
        input_b = build_features(target_b, mask, torch.full((10,), 0.01))
        torch.testing.assert_close(input_a.values, input_b.values, rtol=0, atol=0)

- [ ] **Step 2: Write realized-rate tests**

    def test_channel_mask_reports_discrete_realized_rate():
        result = channel_outage(torch.ones(100, 6), requested_fraction=0.30, seed=7)
        assert result.masked_channels == 1
        assert result.realized_fraction == pytest.approx(1 / 6)

- [ ] **Step 3: Run RED**

Run pytest on the masking/features test file.

- [ ] **Step 4: Implement observed-only features**

    def build_features(target, mask, dt):
        observed = target * mask
        delta = torch.zeros_like(target)
        valid_delta = mask[1:] * mask[:-1]
        delta[1:] = (observed[1:] - observed[:-1]) * valid_delta
        values = torch.cat(
            [observed, mask, dt[:, None], delta, valid_delta],
            dim=-1,
        )
        return FeatureBatch(values=values, dt=dt, mask=mask)

Do not add window mean, variance, energy, or difference energy.

- [ ] **Step 5: Run GREEN and commit**

Commit with message fix: remove target-derived imputation features.

## Task 5: Implement explicit-time models and observed preservation

**Files:**

- Create: validation_v2/models/__init__.py
- Create: validation_v2/models/bilnn.py
- Create: validation_v2/models/bilstm.py
- Create: validation_v2/models/hybrid.py
- Create: validation_v2/models/baselines.py
- Test: tests/validation_v2/test_models.py

- [ ] **Step 1: Write failing completion and gate tests**

    def test_complete_signal_preserves_observed_values():
        observed = torch.tensor([[[1.0], [0.0], [3.0]]])
        mask = torch.tensor([[[1.0], [0.0], [1.0]]])
        pred = torch.tensor([[[9.0], [2.0], [9.0]]])
        completed = complete_signal(observed, mask, pred)
        torch.testing.assert_close(completed, torch.tensor([[[1.0], [2.0], [3.0]]]))

    @pytest.mark.parametrize("gate,expected", [(0.0, 4.0), (0.5, 3.0), (1.0, 2.0)])
    def test_fixed_gate_has_declared_lnn_weight(gate, expected):
        result = fuse(torch.tensor(2.0), torch.tensor(4.0), torch.tensor(gate))
        assert result.item() == expected

- [ ] **Step 2: Write a spying-CfC timing test**

Inject a fake CfC module that records timespans. Assert both branches receive non-None, positive, correctly aligned interval tensors.

- [ ] **Step 3: Run RED**

Run pytest on tests/validation_v2/test_models.py.

- [ ] **Step 4: Implement minimal model APIs**

    def complete_signal(observed, mask, prediction):
        return mask * observed + (1.0 - mask) * prediction

    def fuse(lnn_prediction, lstm_prediction, lnn_gate):
        return lnn_gate * lnn_prediction + (1.0 - lnn_gate) * lstm_prediction

BidirectionalCfC.forward(features, forward_dt, reverse_dt) calls each CfC with the timespans keyword. HybridImputer.forward_components returns both branch predictions, gate, raw prediction, and completed signal.

- [ ] **Step 5: Implement diagnostic baselines**

Add linear interpolation, LOCF, single branches, equal averaging, and fixed gates 0, 0.5, and 1 under the same observed-information budget.

- [ ] **Step 6: Run GREEN and commit**

Commit with message feat: add explicit-time validation models.

## Task 6: Implement trustworthy reconstruction and trajectory math

**Files:**

- Create: validation_v2/objectives/reconstruction.py
- Create: validation_v2/objectives/kinematic.py
- Create: validation_v2/evaluation/synchronization.py
- Create: validation_v2/evaluation/trajectory.py
- Create: validation_v2/evaluation/reconstruction.py
- Test: tests/validation_v2/test_objectives.py
- Test: tests/validation_v2/test_synchronization_and_trajectory.py

- [ ] **Step 1: Write missing-only loss test**

    def test_reconstruction_loss_ignores_observed_errors():
        pred = torch.tensor([[[99.0], [2.0]]])
        target = torch.tensor([[[1.0], [4.0]]])
        mask = torch.tensor([[[1.0], [0.0]]])
        assert missing_mse(pred, target, mask).item() == pytest.approx(4.0)

- [ ] **Step 2: Write synthetic trajectory tests**

    def test_constant_world_acceleration_matches_analytic_solution():
        dt = np.full(101, 0.01)
        acc_world = np.zeros((101, 3))
        acc_world[:, 0] = 2.0
        trajectory = integrate_acceleration(
            acc_world,
            dt,
            p0=np.zeros(3),
            v0=np.zeros(3),
        )
        assert trajectory.position[-1, 0] == pytest.approx(1.0, rel=2e-2)

    def test_body_rotation_maps_acceleration_to_world_axis():
        body_acc = np.array([[1.0, 0.0, 0.0]])
        attitude = Rotation.from_euler("z", [90], degrees=True)
        world_acc = rotate_body_to_world(body_acc, attitude)
        np.testing.assert_allclose(world_acc, [[0.0, 1.0, 0.0]], atol=1e-7)

- [ ] **Step 3: Run RED**

Run both test files.

- [ ] **Step 4: Implement synchronization and SLERP**

Use scipy Rotation.from_quat with x,y,z,w order and Slerp. Reject query timestamps outside the source range.

- [ ] **Step 5: Implement measured-attitude full-record diagnostics**

Use configurable Euler order xyz and mapping body_to_reference. Rotate gravity-removed user_acc times 9.81 into the reference frame. Initialize once per recording. Compute ATE-RMSE, interval RPE/RTE, endpoint drift, velocity RMSE, and delta versus complete IMU.

- [ ] **Step 6: Keep kinematic loss gated**

The API requires denormalized world acceleration, true seconds, frame metadata, and velocity/displacement labels. It raises when units or frame metadata are absent.

- [ ] **Step 7: Run GREEN and commit**

Commit with message fix: rebuild trajectory evaluation in physical coordinates.

## Task 7: Add per-record statistics and immutable provenance

**Files:**

- Create: validation_v2/evaluation/statistics.py
- Create: validation_v2/experiments/provenance.py
- Create: validation_v2/experiments/summarize.py
- Test: tests/validation_v2/test_provenance_and_statistics.py

- [ ] **Step 1: Write failing completeness tests**

    def test_summary_rejects_missing_seed(tmp_path):
        write_fake_run(tmp_path, seed=2026)
        with pytest.raises(ValueError, match="missing required seeds: 2027"):
            summarize_runs(tmp_path, required_seeds=(2026, 2027))

    def test_run_id_changes_when_config_changes():
        assert run_id({"seq_len": 30}, seed=1) != run_id({"seq_len": 50}, seed=1)

- [ ] **Step 2: Run RED**

Run the test file.

- [ ] **Step 3: Implement content-addressed manifests**

run_id is the first 16 hex characters of SHA256 over canonical resolved config, seed, split hash, scaler hash, git commit, and dirty-state digest. Record packages with importlib.metadata.version.

- [ ] **Step 4: Implement paired summaries**

Aggregate by independent recording and seed. Produce mean/SD, median/IQR, paired bootstrap 95% CI, paired mean difference, and rank-biserial effect. Refuse incomplete seed-record matrices.

- [ ] **Step 5: Run GREEN and commit**

Commit with message feat: add immutable experiment provenance.

## Task 8: Add frozen train/evaluate orchestration

**Files:**

- Create: validation_v2/experiments/train.py
- Create: validation_v2/experiments/evaluate.py
- Create: validation_v2/experiments/matrix.py
- Test: tests/validation_v2/test_train_evaluate.py

- [ ] **Step 1: Write failing one-time-test tests**

Use distinguishable train, validation, and test sentinel metrics. Assert checkpoint selection accepts validation history only and evaluate_test_once raises on a second call for the same run ID.

- [ ] **Step 2: Run RED**

Run pytest on tests/validation_v2/test_train_evaluate.py.

- [ ] **Step 3: Implement one-run training**

Save train/validation metrics each epoch. Select by validation missing RMSE. Do not construct the test loader until the checkpoint hash is frozen.

- [ ] **Step 4: Implement frozen evaluation**

Write per-record normalized and physical metrics. Full-record trajectory diagnostics use completed signals. Every output row includes run ID, seed, recording ID, scenario, topology, requested and realized missingness, model, and checkpoint hash.

- [ ] **Step 5: Implement deterministic matrix enumeration**

List every model by seed by mask topology by rate by split protocol combination before execution. Resume only when resolved config and checkpoint hashes match.

- [ ] **Step 6: Run GREEN and commit**

Commit with message feat: orchestrate frozen validation runs.

## Task 9: Add CLI, smoke config, and server matrix

**Files:**

- Create: validation_v2/cli.py
- Create: configs/validation_v2/smoke.yaml
- Create: configs/validation_v2/server_full.yaml
- Test: tests/validation_v2/test_cli_smoke.py

- [ ] **Step 1: Write failing deterministic dry-run test**

    def test_server_matrix_dry_run_is_deterministic():
        first = run_cli(["matrix", "--config", str(SERVER), "--dry-run"])
        second = run_cli(["matrix", "--config", str(SERVER), "--dry-run"])
        assert first.returncode == second.returncode == 0
        assert first.stdout == second.stdout
        assert "2026,2027,2028,2029,2030" in first.stdout

- [ ] **Step 2: Run RED**

Run pytest on tests/validation_v2/test_cli_smoke.py.

- [ ] **Step 3: Add smoke config**

Use two training recordings, one validation recording, one test recording, seed 2026, one epoch, batch size 4, sequence length 30, random 30% missingness, and models linear, bilstm, hybrid.

- [ ] **Step 4: Add server config**

Use seeds 2026 through 2030, strict file and scenario-holdout protocols, rates 0.1 through 0.4, point/block/channel plus separate irregular-timestamp cases, mandatory diagnostic baselines, and reconstruction-only primary training. Kinematic training is a separately named ablation.

- [ ] **Step 5: Implement commands**

    python -m validation_v2.cli smoke --config PATH
    python -m validation_v2.cli matrix --config PATH --dry-run
    python -m validation_v2.cli matrix --config PATH
    python -m validation_v2.cli summarize --root PATH

- [ ] **Step 6: Run GREEN and local smoke**

    conda run -n pinn_imu python -m pytest tests/validation_v2 -q
    conda run -n pinn_imu python -m validation_v2.cli matrix --config configs/validation_v2/server_full.yaml --dry-run
    conda run -n pinn_imu python -m validation_v2.cli smoke --config configs/validation_v2/smoke.yaml

Expected: tests pass; dry-run is stable; smoke creates a manifest, checkpoint, per-record metrics, and summary.

- [ ] **Step 7: Commit**

Commit with message feat: add local and server validation entrypoints.

## Task 10: Validate server handoff and write runbook

**Files:**

- Create: validation_v2/experiments/validate_artifacts.py
- Create: tests/validation_v2/test_server_handoff.py
- Create: docs/validation_v2_server_runbook.md

- [ ] **Step 1: Write failing artifact-validator test**

Create one complete fake run and another missing checkpoint_sha256. Assert the complete run passes and the incomplete run fails naming checkpoint_sha256.

- [ ] **Step 2: Run RED**

Run pytest on tests/validation_v2/test_server_handoff.py.

- [ ] **Step 3: Implement artifact validation**

Validate required files, schema fields, hashes, seed and recording completeness, finite metrics, and one-time test evaluation. Exit nonzero for incomplete runs.

- [ ] **Step 4: Write exact server runbook**

Include environment verification, repository commit check, dataset override, dry-run, full execution, safe resume, artifact validation, summarization, output layout, batch-size-only memory fallback, and packaging results for return to the local machine.

- [ ] **Step 5: Run final verification**

    conda run -n pinn_imu python -m pytest tests/validation_v2 -q
    conda run -n pinn_imu python -m validation_v2.cli matrix --config configs/validation_v2/server_full.yaml --dry-run
    conda run -n pinn_imu python -m validation_v2.experiments.validate_artifacts --root results/validation_v2/smoke
    git status --short

Expected: zero test failures, deterministic matrix, valid smoke artifacts, and no generated results staged.

- [ ] **Step 6: Commit**

Commit with message docs: add validation v2 server runbook.

## Plan Self-Review Checklist

- [ ] Every design requirement maps to at least one task.
- [ ] All production behavior begins with a failing test.
- [ ] No task reads legacy result CSVs or reuses legacy checkpoints.
- [ ] Measured attitude remains evaluation-only.
- [ ] Kinematic training remains a separately named ablation.
- [ ] Model selection is validation-only and final test evaluation is one-time.
- [ ] Local and server commands use pinn_imu.
- [ ] Generated artifacts remain outside commits.
