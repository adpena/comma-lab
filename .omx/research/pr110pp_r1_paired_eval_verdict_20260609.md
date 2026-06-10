# PR110++ R1 — paired contest-CPU eval verdict (frontier vs strict vs R1-PLUS)

**Subagent:** `pr110pp_r1_paired_eval_20260609` (crash-resume continuation) · UTC 2026-06-09/10.
**Lane:** `lane_pr110pp_r2_nonmps_mode_table_20260609` (R2 substrate) + per-eval CPU lanes.
**Operator authorization:** PRE-AUTHORIZED ("all approved on operator end") — the three paired
contest-CPU evals are explicitly operator-sanctioned; this is R1, the operator-dispatched half of R2's
READY_FOR_R1 packet.
**Axis discipline:** results are `[contest-CPU]` ONLY if `contest_auth_eval.py` itself stamps
`evidence_grade=contest-CPU` on the Modal Linux-x86_64 container (1:1 with the contest GHA CPU runner).
CUDA is NOT touched; this single-axis CPU run carries an explicit single-axis waiver because the
contest leaderboard ranks by CPU and CUDA is a SEPARATE promotion gate per CLAUDE.md "Submission auth
eval — BOTH CPU AND CUDA". No PROMOTION is minted here (promotion requires the paired CPU+CUDA axis on
the same bytes); a sub-0.19199 R1-PLUS is a FRONTIER-CANDIDATE flag, not a promotion.

---

## 0. The 5 Modal infra bugs this resume cleared (the dispatch never reached a score before)

Every contest-CPU Modal eval since 2026-05-31 (commit `826cc63ab`) had been failing silently (0 scored
rows). The predecessor fixed bugs 1-4; this resume found + fixed the 5th (the one that actually blocked a
score) and added structural protection:

1. (`0f3a742a3`) `include_source=False` regression — entrypoint module `modal_auth_eval_cpu` not
   re-added → remote `ModuleNotFoundError`. Fix: `add_local_python_source(entrypoint)`.
2. (same) `tac` not importable on remote → `No module named tac`. Fix: add `tac` to python source.
3. (same) subprocess `No module named tac` → prepend resolved tac parent to subprocess `PYTHONPATH`.
4. (`dba23cd8d`) redundant whole-`src` lazy mount poisoned the sibling `upstream` `copy=False` mount.
   Fix: removed the whole-`src` mount.
5. (`4b4d10f68` — THE one that blocked a score) the `ignore_generated_mount_path` callable did
   `path.lstat()` and **fail-closed-to-True on `OSError`**. Modal's `add_local_dir(..., ignore=callable)`
   passes paths RELATIVE to the mount SOURCE root (`evaluate.py` for `add_local_dir("upstream", ...)`),
   resolved against CWD; those relative names do not resolve in CWD → `lstat` raised `FileNotFoundError`
   → the helper returned True → the **entire `upstream` mount landed empty** and `contest_auth_eval.py`
   crashed in 0.11s with `--upstream-dir missing evaluate.py`. Fix: KEEP on unresolvable path (let
   Modal's own enumeration with real paths decide); `.git` dir-name exclusion still extincts the
   fsmonitor socket; a RESOLVED socket is still excluded. +2 regression tests. Sister `copy=True`
   hardening on runtime-critical mounts (`61f6dfd74`, CPU+CUDA wrappers).

**MVP-first validation (CLAUDE.md):** a ~$0.001 one-shot Modal probe
(`experiments/results/pr110pp_r1_paired_eval_20260609/probe_eval_image_mounts.py`) confirmed
`upstream/evaluate.py` + `modules.py` + `frame_utils.py` + `public_test_video_names.txt` + `videos/`
(count=1) all land on the container BEFORE the three full 600-sample dispatches — saving a 6th failed
dispatch.

---

## 1. The three dispatches (post-fix, detached, concurrent)

| eval | archive sha256 | bytes | selector B | switches | call_id |
|---|---|---:|---:|---:|---|
| frontier baseline (control) | `b7106c9b…8997c8c` | 178,493 | 220 | 0 | `fc-01KTQH2NBYYDTRV2PMX4PYF5EZ` |
| strict candidate | `5facf0fb…4fa55d` | 178,493 | 220 | 2 | `fc-01KTQH5589V9023X0BPATHX32V` |
| R1-PLUS candidate | `6c8059a7…d2d49c1` | 178,520 | 247 | 64 | `fc-01KTQH67DXQDNGAFZY0TZAEZ83` |

All three: `upstream/evaluate.py --device cpu`, 600 samples, Linux x86_64 Modal container, byte-identical
inflate runtime (the candidates change only archive bytes; runtime tree identical). Frontier baseline is
re-measured on the SAME Modal host/evaluate.py SHA as the candidates → apples-to-apples per CLAUDE.md.

---

## 2. PRE-REGISTERED falsifiable predictions (recorded BEFORE scores land)

These are committed before harvest; do not edit after results land (append a verdict section instead).

- **Strict candidate**: `ΔS_total` (strict − frontier) within **±0.0002** of 0. Rationale: d_seg
  provably 0 (frame-0 transforms, SegNet scores last frame only — verified on all 67 sampled pairs),
  rate unchanged (220→220 selector bytes), only 2 pairs switched to exact-macOS-CPU argmin-pose. The
  pose delta is tiny (linear pose-avg −1.14e-5) so the score move is near-zero either direction.
- **R1-PLUS candidate**: `ΔS_total` (r1plus − frontier) **< 0** IF the exact-macOS-CPU per-mode pose
  ordering transfers to Linux-x86_64 CPU. Rationale: 64 pairs switched to argmin-pose (sample pose
  reduction 8.27e-2), rate cost +1.80e-5 (negligible), d_seg still 0. The R2 cross-check proved the MPS
  substrate mis-ranks 95.5% of pairs; the exact-CPU table is the corrected substrate, so the incumbent
  (MPS-chosen) selector is per-pair pose-suboptimal and the argmin-switch should reduce pose.
- **KILL criterion (both candidates)**: `ΔS_total ≥ +0.0005` falsifies "macOS-CPU per-mode pose ordering
  transfers to Linux-x86_64 CPU" (host FP drift reversed the ordering). Re-route: score per-mode pose
  directly on the contest-CPU host before any further selector candidate.
- **FRONTIER-CANDIDATE flag**: if R1-PLUS final_score < `0.19199` [contest-CPU] (the archived PR110++
  frontier on the CPU axis), FLAG (do not promote) — CUDA pairing + `pre_submission_compliance_check.py
  --contest-final` still required per CLAUDE.md before any submission.

---

## 3. RESULTS — all 3 evals PASSED, harvested, ingested (exact contest-CPU)

All three: `passed=true`, `evidence_grade=contest-CPU`, `score_axis=contest_cpu`, `n_samples=600`,
Linux x86_64 Modal container, `upstream/evaluate.py --device cpu`, byte-identical inflate runtime.
Scores are `score_recomputed_from_components` (exact). Harvested within session per HARVEST-OR-LOSE.

| eval | final_score [contest-CPU] | d_seg | d_pose | bytes | call_id | elapsed |
|---|---:|---:|---:|---:|---|---:|
| **frontier baseline** | **0.19198534** | 5.5979e-04 | 2.943e-05 | 178,493 | `fc-01KTQH2NBYYDTRV2PMX4PYF5EZ` | 291s |
| strict candidate | 0.19348492 | 5.5979e-04 | 3.480e-05 | 178,493 | `fc-01KTQH5589V9023X0BPATHX32V` | 275s |
| R1-PLUS candidate | 0.21374416 | 5.5979e-04 | 1.5129e-04 | 178,520 | `fc-01KTQH67DXQDNGAFZY0TZAEZ83` | 246s |

### Verdict vs the pre-registered predictions

- **Apples-to-apples VALIDATED:** the fresh frontier baseline `0.19198534` matches the archived
  `0.19199` within `4.66e-6` AND matches the canonical frontier pointer score EXACTLY. The Modal CPU
  path (post 5-bug fix) is a faithful contest-CPU axis.
- **d_seg IDENTICAL across all three** (`5.5979e-04`) — the SegNet-blindness mechanism R2 verified held
  exactly on the contest-CPU host: frame-0 transforms cannot move the last-frame SegNet argmax. So 100%
  of every score difference is the **pose term moving the WRONG way**.
- **strict ΔS = +0.00149958** — predicted ±0.0002; **KILL** (≥+0.0005 fired, 7.5× over threshold). Pose
  rose 2.943e-05 → 3.480e-05.
- **R1-PLUS ΔS = +0.02175882** — predicted <0; **KILL** (43× over threshold; massively falsified). Pose
  rose 2.943e-05 → 1.5129e-04 (5.1× the frontier pose).
- **NO frontier-candidate:** both candidates score WORSE than 0.19199. Verdict `above_frontier` (worse).

### Mechanism — the KILL is a clean falsification, not an implementation bug

The pre-registered KILL criterion fired EXACTLY as designed: **the exact-macOS-CPU per-mode pose
ordering does NOT transfer to Linux-x86_64 CPU.** R2's per-mode pose table (the substrate that chose the
2 and 64 argmin-pose switches) was built on macOS-CPU. Host floating-point drift between macOS-arm-CPU
and Linux-x86_64-CPU reversed the per-pair argmin-pose ordering at the frontier operating point
(pose ~1e-5..1e-3, where the √(10·d_pose) marginal is largest and tiny FP perturbations flip the per-pair
best mode). Every "improvement" switch chosen on macOS-CPU was, on the contest-CPU host, a per-pair
pose REGRESSION. The more switches, the worse: strict (2 switches) +0.00150; R1-PLUS (64 switches)
+0.02176 — monotonic in switch count, the signature of a systematically mis-ranked substrate.

This is the SAME class as R2's own falsifiable finding: R2 proved the MPS substrate mis-ranks 95.5% of
pairs vs macOS-CPU; R1 now proves macOS-CPU mis-ranks vs Linux-x86_64-CPU at this operating point. The
per-mode pose lever is real but the ONLY substrate that can choose it correctly is the contest-CPU host
itself.

### Routing (DEFER-pending, NOT kill of the paradigm)

Per CLAUDE.md "Forbidden premature KILL": the per-pair-selector PARADIGM is intact (d_seg=0 SegNet
blindness is real; a per-pair pose lever exists). What is FALSIFIED is the **macOS-CPU substrate as the
selector chooser**. Reactivation criterion: score the 16-mode per-pair pose table DIRECTLY on the
Linux-x86_64 contest-CPU host (one Modal CPU pass that renders + scores all 16 modes per pair, B=16),
then choose argmin-pose on THAT table. Only switches that lower pose on the contest-CPU host are
admissible. Until then, the incumbent FECa selector (whatever its provenance) is the operating point —
do not switch pairs on a non-contest-CPU substrate. The R2 mode-table cluster should consume the
contest-CPU per-mode table, NOT the macOS-CPU or MPS one.

### Ingested rows (first exact contest-CPU ActionEffect rows in PR110++ history)

`tools/ingest_exact_eval_to_candidate.py` minted the typed V3 rows for all three (authority_tier
`contest_cpu`, metric_family `exact_evaluate`, score_roadmap_update_eligible True, promotion_eligible
False — promotion still requires the paired CUDA axis):
- `frontier_baseline_cpu/candidate_action_evaluation_pr110pp_r1_frontier_baseline_contest_cpu.v1.json`
- `strict_candidate_cpu/candidate_action_evaluation_pr110pp_r1_strict_candidate_contest_cpu.v1.json`
  (verdict above_frontier, route build_authority_trace, seg_flat pose_worsened)
- `r1plus_candidate_cpu/candidate_action_evaluation_pr110pp_r1_r1plus_candidate_contest_cpu.v1.json`
  (verdict above_frontier, route build_authority_trace, seg_flat pose_worsened)

### Ingest-tool fix landed (sister of the Modal fixes)

`tools/ingest_exact_eval_to_candidate.py` could not read the canonical contest_auth_eval.json: it read
`archive_bytes` (nested under `b2`) but the canonical schema stores `archive_size_bytes` at top level,
the archive sha under `provenance.archive_sha256`, and signals success via `score_claim_valid` +
contest-axis `evidence_grade` (no `pipeline_works`/`b2`). Extended `_extract_distortions` (top-level
`archive_size_bytes`), candidate-sha extraction (`provenance.archive_sha256`), `_bridge_ok` (canonical
success signal), and `_metric_family` (drop brittle schema-string literal; `ran_evaluate_py`+full-fields
suffices). +1 regression test (`test_ingest_canonical_contest_auth_eval_json_schema`).

---

## 4. Wire-in (Catalog #125)

- Hook #5 continual-learning: each exact row ingested via `tools/ingest_exact_eval_to_candidate.py` →
  the FIRST exact contest-axis `tac.action_effect.v1` rows in PR110++ history; reseeds the V3 ΔS-judge.
- Hook #2 Pareto: confirms/refutes R2's finding that the FECa coder resistance (not the 220-byte floor)
  is the binding constraint, and that +27 bytes buys the distortion-optimal selection.
- Hook #6 probe-disambiguator: resolves "does the exact-CPU per-mode pose table transfer Linux↔macOS?"
- Hooks #1/#3/#4: actionable once a real contest-CPU win lands (selector becomes a bit-allocator input).
