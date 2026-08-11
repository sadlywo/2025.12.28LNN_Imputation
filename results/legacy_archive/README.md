# Legacy result archive index

This directory is the boundary between pre-refactor results and the new
physics-loss campaign. Existing result directories are intentionally left at
their original paths so historical scripts, figures, and checksums remain
valid. They are read-only inputs for comparison and must not receive new runs.

The corresponding historical code is archived under
`legacy/pre_validation_v2/`.

The archived legacy scopes are:

- `baseline_Imputation_Method`, `recon_only_method_comparison`
- `bidirectional_lnn_residual`, `bidirectional_lnn_residual_tra`
- `hybrid_lnn_lstm`, `hybrid_multirate`
- `loss_functions_experiment`, `consistency_weight_ablation`
- `physical_head_experiment_1_OnlyRecon`, `physical_head_experiment_2_AdapativeLoss`
- `physical_loss_function_weights`, `physical_loss_function_weights_stable`
- all pre-existing model, sequence-length, hidden-size, drift, trajectory, and visualization ablations

All refactored outputs start under `results/physics_loss_refactor/v1/`. No
legacy checkpoint is treated as a refactored model checkpoint.
