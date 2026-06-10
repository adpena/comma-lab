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

## 3. RESULTS (appended after harvest)

<!-- HARVEST PENDING — daemon experiments/results/pr110pp_r1_paired_eval_20260609/harvest_daemon.py -->

---

## 4. Wire-in (Catalog #125)

- Hook #5 continual-learning: each exact row ingested via `tools/ingest_exact_eval_to_candidate.py` →
  the FIRST exact contest-axis `tac.action_effect.v1` rows in PR110++ history; reseeds the V3 ΔS-judge.
- Hook #2 Pareto: confirms/refutes R2's finding that the FECa coder resistance (not the 220-byte floor)
  is the binding constraint, and that +27 bytes buys the distortion-optimal selection.
- Hook #6 probe-disambiguator: resolves "does the exact-CPU per-mode pose table transfer Linux↔macOS?"
- Hooks #1/#3/#4: actionable once a real contest-CPU win lands (selector becomes a bit-allocator input).
