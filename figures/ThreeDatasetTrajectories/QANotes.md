# Figure QA notes

- Core conclusion: one pipeline spans smartphone, MAV, and pedestrian motion domains.
- Archetype: quantitative trajectory grid (2 representatives × 3 datasets).
- Final size: 183 × 116 mm (double-column).
- Backend: Python/matplotlib only.
- Primary export: SVG with editable text; PDF, 600-dpi TIFF, and 300-dpi PNG included.
- Data integrity: native coordinates; no smoothing, registration, or spatial rescaling.
- Colour integrity: shared viridis scale, 0–1.498 m/s; values above the pooled 98.5th percentile are colour-clipped only.
- Rendering: at most 2600 deterministic points per trajectory; source indices are recorded.
- Interpretation limit: representative dataset coverage, not an imputation-performance comparison.
- Statistics: none; this figure contains representative trajectories rather than inferential estimates.
