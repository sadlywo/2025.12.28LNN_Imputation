# Three-dataset joint training protocol

The formal main-table configuration jointly trains one shared imputation model
with OxIOD, EuRoC MAV Vicon Room 1/2, and IDOL Building 1/2/3.

## Frozen contract

- Discover recordings through the three registered `DatasetAdapter` instances.
- Split each dataset independently and deterministically at recording-file level
  with seed 2026 and requested ratios 80/10/10. Tiny strata keep at least one
  validation and one test recording, so no recording or source file crosses a
  split and every dataset occurs in train, validation, and test.
- Fit one `RobustTrainScaler` per dataset using that dataset's training records
  only. This is required because OxIOD acceleration is expressed in G with
  gravity-compensated semantics, whereas EuRoC and IDOL use m/s² specific force.
- Allocate one third of the bounded train/validation window budget to each
  dataset and interleave the resulting windows. The shared model therefore sees
  all three normalized domains without IDOL dominating merely because it has
  more recordings.
- Train at 30% point missingness. Evaluate the frozen checkpoint on point,
  block, and channel missingness at 10%, 20%, 30%, and 40%, plus the declared
  20% interval-jitter / 30% point-missing irregular-time case.
- Report both pooled metrics and the explicit `dataset` field in the split
  manifest so per-dataset aggregation remains possible.

The joint benchmark currently uses `reconstruction_only`. Joint
`physics_informed` optimization fails closed because the three datasets do not
share the same acceleration semantics; per-record physical and trajectory
evaluation still uses the correct adapter semantics.

## Configurations

- `configs/validation_v2/joint_three_dataset_smoke.yaml`: bounded integration
  test for the shared BiLNN data path.
- `configs/validation_v2/server_full.yaml`: five-seed reference-model campaign.
- `configs/validation_v2/modern_tuning.yaml`: BRITS/SAITS tuning on the joint
  validation data.
- `configs/validation_v2/modern_stage_a.yaml`: joint reference + BRITS/SAITS
  main-table campaign. CSDI and SSSD remain deferred.
