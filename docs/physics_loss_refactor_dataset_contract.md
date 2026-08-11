# Physics-refactor dataset adapter contract

The training and objective layers consume `validation_v2.types.Recording` and
must not branch on a dataset name. A EuRoC MAV or IDOL subset adapter should:

1. Implement `DatasetAdapter` from `validation_v2.data.adapters`.
2. Declare gyro unit, acceleration unit/semantics, quaternion order, rotation
   mapping, and time unit in `DatasetSemantics`.
3. Discover paired IMU/reference sources without mixing recordings.
4. Convert timestamps to strictly increasing seconds and positions to metres.
5. Return six channels ordered as body-frame gyro xyz followed by body-frame
   acceleration xyz.
6. Preserve the reference quaternion order in metadata and return it through
   the canonical `Recording` fields.
7. Register once with `register_dataset_adapter(adapter)` and set
   `dataset_name` in the experiment YAML.

Implemented adapter names are `euroc_mav`, `idol`, and `oxiod`. EuRoC source
wxyz quaternions and IDOL source wxyz quaternions are converted to the
canonical xyzw order. The IDOL adapter deliberately uses the Stencil IMU,
because its published frame matches the ground-truth frame; it does not mix
iPhone acceleration with Stencil pose.

Before a new adapter enables a non-zero physics weight, it must run the clean
mechanization diagnostic for that dataset. Dataset documentation alone is not
enough to mark an IMU/reference extrinsic or rotation direction as validated.

EuRoC MAV and IDOL Stencil acceleration are loaded as raw specific force in
`m/s^2`. This is distinct from the OxIOD `user_acc` convention. Both adapters
remain frame-diagnostic-only: registration and successful parsing do not by
themselves authorize a non-zero physics weight.
