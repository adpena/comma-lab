# BEV staticity v2 — absolute-trajectory control and n600 verdict

**Verdict:** `MEASURED_D0_PASS_D1_D2_NEGATIVE_D3_BLOCKED` for the exact
G1-calibrated PoseNet absolute chart. The repaired bottom-connected hood control passes at n64 with
MEASURED p50 **0.0 scorer px** and **0.922991** of samples within the registered 1 px floor. The
load-bearing n600 rerun also passes D0, and the canonical singleton f0 sidecar resolves all three v1
batch-geometry mismatches: **0/600** cached-f1 label mismatches.

After that gate opened, Road and Lane did not become static. Their n600 MEASURED p50 ruling
residuals are **39.0226 px** and **47.1192 px**, with only **0.043093** and **0.043713** of samples
within 1 px. Movable is independently non-static (p50 **10.7628 px**, fraction **0.178138**), as
expected. D3 is therefore blocked and emits no static-ground coefficients, xi B-spline packet, byte
ratio, or score claim.

Axis is `[macOS-CPU advisory]`; deterministic CPU Torch/NumPy seed 1234; `score_claim=false`;
`promotion_eligible=false`; pointer `0.1910828242 [contest-CPU]` UNMOVED. MAIN landing review is
required.

## What v2 corrected

- Canonical SegNet custody now scores `gt_f0[t]` and `gt_f1[t]` as independent singleton calls. The
  f1 result must equal the hash-pinned cache before its f0 result is admitted to the sidecar.
- The missing transition `gt_f1[t-1] -> gt_f0[t]` is scored directly by the frozen CPU PoseNet; the
  v1 nearest-target proxy and misuse of already-relative targets as an absolute path are removed.
- Translation-first twists use the hash-pinned G1 calibration and `tac.lie`. With `A_f1[0]=I`, v2
  constructs
  `A_f0[0]=A_f1[0] exp(xi_within[0])^-1`,
  `A_f0[t]=A_f1[t-1] exp(xi_cross[t])`, and
  `A_f1[t]=A_f0[t] exp(xi_within[t])`.
- The largest 4-connected MyCar component touching the bottom edge is the hood control. f0 hood
  transform/closure uses `A_f0`; f1 Road/Lane/Movable transport uses `A_f1`.
- n64 executes only D0. Road/Lane, worldsheet, and D3 are unreachable until its exact-source receipt
  passes every preregistered check.

## MEASURED D0 gate

| scale | singleton f1/cache mismatches | hood p50 / p95 (px) | fraction <=1 px | max world-to-ego closure | verdict |
|---|---:|---:|---:|---:|---|
| n64 | 0 / 64 | 0.0 / 6.0 | 0.922991 | 4.441e-16 m | **PASS** |
| n600 | 0 / 600 | 0.0 / 11.0 | 0.913057 | 3.580e-15 m | **PASS** |

All finite-SE(3), homogeneous-row, inverse-compose, cross-phase, and within-phase closure checks are
exact or below `1e-9`. Both receipts reproduced byte-identically on a stage-resume rerun.

## MEASURED n600 D1 / D2 rows

| stratum | shallow / excluded-deep | p50 / p90 residual (px) | fraction <=1 px | events / static segments | disposition |
|---|---:|---:|---:|---:|---|
| Road | 722,312 / 789,922 | 39.0226 / 180.5095 | 0.043093 | 13 / 7 | non-static; D1/D2 hold false |
| Lane | 448,715 / 373,347 | 47.1192 / 186.7025 | 0.043713 | 2 / 3 | non-static; D1/D2 hold false |
| Movable | 96,828 / 94,478 | 10.7628 / 57.5552 | 0.178138 | 32 / 25 | expected non-static control passes |

D2 uses the directrix-plus-absolute-xi ruling residual and refuses a raw numerical Gaussian-K claim.
The measured developable fractions at the 1 px floor are the same `0.043093` Road and `0.043713`
Lane fractions above. Those are negative for this chart at this registered floor.

## D3 and routing

- **D3:** `NO_ROAD_OR_LANE_STRATUM_PASSED_HOOD_GATED_D1_D2`. No xi B-spline or static-ground
  coefficient/event packet was produced.
- **Routing:** U1/U2, U5, and P0 activation are false; dispatch authority is false.
- **Verdict scope:** this falsifies the static Road/Lane collapse premise for the exact
  G1-calibrated, `s_t=-0.00143`, `s_r=0`, `pitch=-0.05` PoseNet absolute chart. It does not falsify
  true absolute ego GT, a different independently admitted calibration, or other BEV/worldsheet
  families.
- **Reactivation:** require a new custodied absolute-motion source or calibration that preserves the
  same D0 pass and raises both Road and Lane to p50 <=1 px and fraction >=0.5 at n600 before D3.

## Triality and apparatus

- **DSL:** typed research-only CLI; SSD preflight; exact n64 receipt gate before n600; no invented
  trainer flag or launch authority.
- **DAG:** sibling `bev_staticity_v2_absolute_trajectory_DAG_FEED_20260721T183219Z.md` records the
  D0 pass, D1/D2 negative, D3 blocker, and reactivation edge.
- **Equations:** the phase-consistent cross-then-within SE(3) chain above; subpixel boundary law
  `t=M_p/(M_p+M_q)`; openpilot IPM; directrix/ruling residual in scorer-pixel equivalents.
- **Resumability:** 600 singleton-label stages, 600 dual-pose stages, 600 measurement stages, and 38
  chunk receipts are preserved. The rebuildable 275 MiB SSD evidence tree has source hashes and
  atomic stage records; no local bulk scratch exists.
- **Pointer delta:** exactly zero.

## Evidence and git proof

- SSD root: `/Volumes/VertigoDataTier/pact/evidence/bev_staticity_v2_20260721/canonical_v1`.
- Frozen cache SHA-256:
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
- n64 receipt file SHA-256:
  `94a7d7b5635e04d5da6f22e1d4f2e5b8d170a9dc95923e3835b9421aedb8bbba`;
  internal canonical SHA-256:
  `4312762657767f230b9de2989f6836dd7a7002e8312227805cdd00b7ef2c634f`.
- n600 receipt file SHA-256:
  `c3ec847ba5ca43246f01af12f7bd650b14aba2784eb1878c29c16f8a4469ab96`;
  internal canonical SHA-256:
  `567f0afb9c8982fe78ab6897dc486f93d71fb9503468ce0e5f9d1a204b747d1d`.
- n600 singleton manifest SHA-256:
  `7614b540d9bd926d258ade9d8dc53598f24ed7a7bc2c41477b70af22b1b27c75`.
- Implementation base SHA: `97662e8a64575368cc0801a5e1b6b002b73f3218`.
- Sealed implementation commit: `ba1c4500a72d79ee353b0c7654886caa62437f40`.
- Canonical blocking outcome: probe
  `bev_staticity_v2_absolute_trajectory_n600_20260721`, verdict `KILL` with the exact-chart-only
  scope and three explicit reactivation criteria in `.omx/state/probe_outcomes.jsonl`.
- Verification: ruff, py_compile, diff check, and **13 focused tests passed**; two review-tracker
  passes were recorded for each Python file.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; v7.5/v8 SPECs; `PROGRAM.md`; craft handoff; `reports/latest.md`;
lane/subagent registries; latest Codex/Claude findings and design memos; canonical blocking-probe
ledger; `gt_n600.npz`; G1 receipt and LawRef calibration; frozen upstream SegNet/PoseNet;
`tac.lie`; `clip_profile`; `lane_sdf_component`; `ego_xi_trajectory`; the v1 memo/receipt/DAG/reuse
manifest; live arm inbox and broadcast directives.
