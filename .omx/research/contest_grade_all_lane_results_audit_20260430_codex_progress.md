# Codex Progress Ledger - Contest-Grade Audit

Adjacent source of truth:
`contest_grade_all_lane_results_audit_20260430.md`

Date: 2026-04-30

## Scope

This ledger tracks changes that affect evidence grading, artifact custody,
manifest compliance, and exact-eval readiness. It is intentionally strict:
implementation readiness is not score evidence.

## Landed

1. `.nrv` archive members are now explicitly allowed by
   `experiments/contest_auth_eval.py`.
   - Security posture is unchanged: unknown suffixes, forbidden housekeeping
     files, and zip-slip paths still fail.
   - Focused tests pass in `src/tac/tests/test_runtime_guards_pass_3.py`.

2. OWV3 decode can be reached through the contest inflate loader.
   - Magic: `b"OWV3"`.
   - Decode module: `tac.owv3_sensitivity_weighted:decode_owv3_archive`.
   - Sensitivity computation is compress-time only and is not required at
     inflate.

3. Neighboring dispatch regression checks passed:
   - OWV2 inflate dispatch still works.
   - IMPS registry entry still resolves.

## Verification Run

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_sensitivity_map.py \
  src/tac/tests/test_owv3_sensitivity_weighted.py \
  src/tac/tests/test_runtime_guards_pass_3.py \
  src/tac/tests/test_contest_auth_eval.py -q

34 passed

.venv/bin/python -m pytest \
  src/tac/tests/test_owv2_renderer_archive_inflate.py::test_owv2_archive_inflate_renderer_dispatch \
  src/tac/tests/test_imps_renderer_archive.py::test_imps_registered_in_codec_magic_registry_synthetic -q

2 passed
```

## Grade Impact

- Grade A table now includes the harvested PFP16 exact CUDA archive eval.
- Lane G v3 PFP16 is the verified frontier at recomputed
  `1.0440481283330025`, with display `final_score=1.04`.
- OWV3 remains below Grade A until:
  - exact stack archive exists,
  - SHA-256 is recorded,
  - CUDA sensitivity artifact provenance exists,
  - `contest_auth_eval.py -> inflate.sh -> upstream/evaluate.py` completes,
  - score recomputes over 600 samples.
- Lane 12 remains empirical-only until full CUDA `.nrv` archive scoring lands.

## Next Audit Actions

1. Fix the PFP16 remote provenance/adjudication parser bug class that wrote
   `contest_cuda_score=100.0` and `hard_kill_triggered=true`; trust or
   recompute from `contest_auth_eval.json`. The current adjudicator now emits
   scoped regression fields for future runs.
2. Attach A++ evidence for PFP16 on T4/equivalent hardware if it becomes the
   submission candidate.
3. Recover or rebuild the exact OWV2 diagnostic archive, or keep it Grade B.
4. Build OWV3 archive with deterministic zip and provenance.
5. Verify Lane 12 payload closure in a clean contest inflate environment.
6. Record manifest, SHA, `contest_auth_eval.json`, `provenance.json`, and logs
   together for every candidate.

---

## Update - 2026-04-30 Later

Audit-relevant implementation changes:

- `src/tac/submission_archive.py` now recognizes `masks.nrv` via
  `masks_nrv`, `RENDERER_NRV_MANIFEST`, and
  `RENDERER_NRV_COMPACT_MANIFEST`.
- Canonical archive construction now writes deterministic ZIP entries with
  fixed timestamp, fixed permissions, and manifest order instead of iterating a
  set with source mtimes.
- `masks.nrv` validation checks minimum size and `NRV1` magic.
- `experiments/build_lane_g_v3_owv3_stack.py` emits deterministic archive and
  provenance, but it is not Grade A until exact CUDA eval.
- `experiments/convert_fisher_to_owv3_sensitivity_map.py` emits tagged
  sensitivity artifacts with source hashes and authoritative/non-authoritative
  labels.

Grade impact:

- PFP16 exact CUDA eval has landed under
  `experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z`.
- Authoritative fields from `contest_auth_eval.json`:
  `final_score=1.04`,
  `score_recomputed_from_components=1.0440481283330025`,
  `avg_posenet_dist=0.0034602`, `avg_segnet_dist=0.0040083`,
  `archive_size_bytes=686635`, archive SHA-256
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Hardware is NVIDIA GeForce RTX 4090 with `gpu_t4_match=false`, so PFP16 is
  Grade A score-grade but not A++.
- `remote_provenance.json` currently has invalid parser/adjudication fields
  `contest_cuda_score=100.0`, `hard_kill_triggered=true`, and
  `lane_status=HARD_KILL_REGRESSION`. These are superseded by
  `contest_auth_eval.json`; the current adjudicator fix has landed.

- OWV3 remains implementation-smoke only until a CUDA sensitivity artifact and
  exact archive eval exist.
- Lane 12 remains empirical-only until full CUDA `.nrv` archive eval exists,
  despite canonical manifest support now being unblocked.

---

## Update - 2026-04-30 Auth-Eval Parser Fix Landed

The parser/adjudication fix has landed after the PFP16 harvest:

- `scripts/adjudicate_contest_auth_eval.py` is now the canonical remote
  adjudicator for exact archive evals that need provenance updates.
- It reads `score_recomputed_from_components` from `contest_auth_eval.json`;
  `final_score` is retained only as rounded display.
- It validates the evaluated archive SHA-256 and byte count against the JSON
  provenance and rejects non-CUDA evidence.
- `remote_lane_pfp16_stack.sh` no longer can parse the formula coefficient
  `100` from the human-readable report line.
- `remote_lane_omega_w_v2_stack.sh` and `remote_lane_8_multipass.sh` were
  refactored off the same fragile parser.
- Sweep/adjacent scripts no longer fall back to `grep -Eo '{.*}'` over logs;
  they require `contest_auth_eval.json`.
- Strict preflight now forbids remote-lane auth-eval score regexes and
  last-JSON-object log scraping.

Verification:

- `src/tac/tests/test_remote_auth_eval_hardening.py`: 6 passed.
- Modified shell scripts passed `bash -n`.
- `py_compile` passed for the new adjudicator, launcher wrapper, and preflight.

---

## Update - 2026-04-30 Contest-Grade Guardrail Expansion

No new lane is promoted to Grade A in this update. The material audit change is
that dispatch/evaluation harnesses now have stronger self-protection against
false evidence and duplicate spend.

New guardrails:

- `scripts/launch_lane_with_retry.py`:
  - single-flight lock per logical lane label,
  - live Vast same-prefix detection,
  - signal/timeout cleanup of child process groups,
  - fail-closed `UNKNOWN_EXISTING_LABEL_PREFIX` state.
- `src/tac/preflight.py`:
  - `check_launch_retry_wrapper_singleflight_and_signal_safe`.
- `src/tac/tests/test_remote_auth_eval_hardening.py`:
  - expanded to 9 tests including duplicate-prefix refusal and timeout cleanup.

SegMap clone dispatch evidence:

- First fresh attempt `35906564` failed NVDEC at setup and auto-destroyed.
- Second attempt `35906669` launched successfully.
- Remote setup proof:
  - `SETUP_COMPLETE` in `/workspace/setup.log`.
  - heartbeat present under
    `/workspace/pact/lane_sa_segmap_clone_results/heartbeat.log`.
  - Stage 2 training reached in `run.log`.

Grade impact:

- SegMap clone remains ungraded until it produces `contest_auth_eval.json`
  through exact archive CUDA eval.
- H-V3 remains ungraded. It is dispatched as
  `35907873` / `lane_h_v3_joint_halfframe_2026-04-30_codex_a4`, but the last
  checkpoint only proved setup activity and lightweight NVDEC pre-probe; no
  archive eval exists.
- Q-FAITHFUL remains gated/high-risk because the current design path includes
  KL-distill-like machinery.

Validation:

- `src/tac/tests/test_remote_auth_eval_hardening.py`: 9 passed.
- `py_compile`: clean for retry launcher, preflight, adjudicator.
- `check_launch_retry_wrapper_singleflight_and_signal_safe`: 0 violations.
- `check_remote_lane_auth_eval_json_adjudication`: 0 violations.
- `git diff --check`: clean.

---

## Update - 2026-04-30T16:16Z Evidence Ledger Addendum

New evaluated artifact:

- Lane: Lane 12 NeRV mask replacement.
- Evidence path:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json`.
- Exact path: archive CUDA eval through `experiments/contest_auth_eval.py` with
  `submissions/robust_current/inflate.sh` and upstream evaluator, 600 samples.
- Hardware: RTX 4090, CUDA, `gpu_t4_match=false`.
- Score: `score_recomputed_from_components=26.03719330455429`,
  `final_score=26.04`.
- Components: PoseNet `49.77849960`, SegNet `0.03528685`, rate contribution
  `0.19741250`, archive `296478` bytes.
- Archive SHA-256 from nested provenance:
  `864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97`.
- Grade: exact-CUDA regression review. Not Grade A. Not promotable. Retires
  the measured implementation/config only.

Evidence policy clarifications:

- PFP16 exact CUDA evidence remains Grade A score-grade only until the new
  Lightning/T4 path produces `gpu_t4_match=true`.
- HM-S is live but uses `variant=kl_distill`; any result is high-risk and must
  be treated as forensic until KL collapse risk is cleared by exact evidence.
- Lane 19, SA, and H-V3 are live/training but have no lane-local
  `contest_auth_eval.json`; no result claim is allowed yet.
- OWV3/Fisher has no artifact after two Vast NVDEC failures; no result claim is
  allowed.

Audit status:

- No new positive Grade A/A++ lane result was added in this update.
- One negative Grade-A-quality exact-regression evidence item was added for
  Lane 12.

---

## Update - 2026-04-30T16:25Z PFP16 Promoted To A++ Evidence Grade

Lane: PFP16 stack / Lane G V3 PFP16.

Evidence path:
`experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json`.

Contest-grade facts:

- Eval chain:
  `contest_auth_eval.py --device cuda -> submissions/robust_current/inflate.sh -> upstream/evaluate.py`.
- Samples: `600`.
- Hardware: Lightning AI Tesla T4.
- `gpu_t4_match=true`.
- Archive bytes: `686635`.
- Archive SHA-256:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Rounded final score: `1.04`.
- Recomputed score: `1.043987524793892`.
- PoseNet: `0.00346442`.
- SegNet: `0.00400656`.
- Rate: `0.01828808`.

Audit grade: A++.

This supersedes the earlier RTX 4090 Grade A score-grade run for hardware
provenance, while matching the same exact archive SHA and byte count.

---

## Update - 2026-04-30T16:45Z OWV3 Smoke Evidence And Source-Doc Sync

New non-promotable OWV3/Fisher smoke evidence:

- Modal label: `lane_g_v3_owv3_fisher_smoke_20260430_codex`.
- Recovered artifacts:
  `experiments/results/lane_lane_g_v3_owv3_fisher_smoke_20260430_codex_modal/`.
- Fisher/sensitivity artifacts exist for the smoke:
  `hessian_per_weight.pt`, `owv3_sensitivity_map.pt`, metadata JSONs.
- Build provenance:
  `lane_g_v3_owv3_fisher_stack_results/build_provenance.json`.
- Archive SHA:
  `710cba0c7c490b13db8b0aee897dd0f33cb8b66a6ed229466bf0d1aea392f5a3`.
- Archive bytes: `912971`, which is `+218897` vs Lane G v3.
- Renderer bloat: ASYM `296776` bytes -> OWV3 `572250` bytes.
- No exact eval was run (`RUN_CONTEST_EVAL=0`; Modal remains advisory).

Audit grade: `empirical` suspicious negative smoke. It cannot promote or kill
the OWV3 method family. It does require encoder overhead/config review before
another promotion attempt.

Source docs synchronized in this pass:

- `contest_grade_all_lane_results_audit_20260430.md`
- `shannon_floor_execution_readiness_20260430.md`
- `shannon_floor_paper_rigor_writeup_blueprint_20260430.md`
- `grand_council_paradigm_shift_to_shannon_floor_20260430.md`

Policy addendum: PFP16 A++ is the controlling frontier; Lane 12 NeRV
`jsonfix40` is exact-negative for that measured implementation only; Dykstra
sub-0.30 byte ceiling is `450545` bytes and remains a necessary bound, not a
sufficiency claim.
