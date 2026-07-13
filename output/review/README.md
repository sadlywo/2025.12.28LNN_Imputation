# Manuscript Review Package

## Editorial Outcome

**Reject — Resubmit Encouraged**

The current evidence chain must be rebuilt before manuscript revision. The main blockers are target leakage, invalid Vicon/IMU synchronization, unresolved coordinate-frame and trajectory mechanics, mismatch between the published physics loss and the implementation, and result tables that are not uniquely traceable to saved artifacts.

## Review Files

1. [Phase 0 reviewer configuration](phase0_reviewer_configuration.md)
2. [EIC review](phase1/eic_review.md)
3. [Reviewer 1 — Methodology & Reproducibility](phase1/reviewer1_methodology.md)
4. [Reviewer 2 — Inertial Navigation & Physical Modeling](phase1/reviewer2_domain.md)
5. [Reviewer 3 — Continuous-Time Models & Edge Deployment](phase1/reviewer3_perspective.md)
6. [Devil’s Advocate stress test](phase1/devils_advocate.md)
7. [Editorial decision and executable revision roadmap](editorial_decision_and_roadmap.md)

## Recommended Execution Order

`P1-1/2/3/4/5 → P1-6 → P1-7/8/9/10 → P1-11 → P2 evidence strengthening → P3 publication QA`

Do not begin prose-only manuscript rewriting until the P1 validity checks pass and the core experiments have been rerun.

## Scope Note

The review process was read-only with respect to the submitted manuscript and repository source code. Only independent review documents were created under `output/review/`.
