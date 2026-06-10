# CUDA-pairing of the recoded-R3 frontier — the required BOTH-axis pre-submission gate

**Date:** 2026-06-10
**Subagent:** `cuda_pairing_recoded_r3_20260610`
**Lane:** `lane_cuda_pairing_recoded_r3_20260610`
**Mode:** required BOTH-axis pre-submission gate (CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")
**Operator authorization:** Modal spend pre-approved (<$1 of remaining budget).
**Inputs:** `.omx/research/leapfrog_pr112_absorb_recode_verdict_20260610.md` (the recoded archive +
the CPU axis) + `.omx/research/cuda_axis_frontier_eval_verdict_20260610.md` (prior CUDA infra +
the R3-candidate CUDA distortion + the transfer caveat).

---

## 0. HEADLINE — lossless recode is axis-invariant. CUDA confirmed.

The current CPU frontier (recoded-R3, sha `b46897267ded…`, 177,169 B, `[contest-CPU]` 0.19109982,
the leapfrog winner over PR #112) now has its `[contest-CUDA]` axis on the EXACT same sha.

| axis | score (recomputed from components) | d_seg | d_pose | bytes | call_id |
|---|---:|---:|---:|---:|---|
| **`[contest-CPU]`** (Modal Linux x86_64, 600 samples) | **0.19109982** | 0.00055978 | 0.00002942 | 177,169 | `fc-01KTRAYS68F3S0YWFT0CX35HDG` |
| **`[contest-CUDA]`** (Modal T4, 600 samples) | **0.22528084** | 0.00066254 | 0.00016857 | 177,169 | `fc-01KTRCQ6KYPYFNYB83GKYDYKXE` |

Both `passed=true`, both recomputed from components (the rounding-trap discipline: the rounded
`final_score` field said 0.19 / 0.23 respectively — far too coarse to adjudicate; the recomputed
values are exact to 8+ decimals).

### Pre-registered prediction — CONFIRMED to 8 decimals

The recode is a **pure lossless entropy re-code** (byte-identical decoded pixels, proven in the
leapfrog verdict's decode-parity gate). Identical pixels ⇒ identical scorer inputs ⇒ identical
d_seg/d_pose on ANY hardware axis. So the CUDA score was predicted BEFORE harvest as:

> recoded-R3 CUDA = (R3-candidate CUDA distortion) + (recoded rate term)
> = `100·0.00066254 + √(10·0.00016857) + 25·177169/37,545,489`
> = `0.066254 + 0.04105728 + 0.11796956` = **0.22528084**

**Measured = 0.22528084193079728. Prediction confirmed exactly.**

### Lossless-on-CUDA verification (the byte-identity invariant, now proven on the CUDA axis)

| component | recoded-R3 CUDA | R3-candidate CUDA (sha `1ccae18d`, prior eval) | Δ |
|---|---:|---:|---:|
| avg SegNet dist | 0.00066254 | 0.00066254 | **0.0** |
| avg PoseNet dist | 0.00016857 | 0.00016857 | **0.0** |

The d_seg/d_pose deltas vs the R3-candidate's CUDA eval are **exactly zero** — well inside the
<1e-6 flag threshold. The ctx-range/AR recode is byte-identical not just on the CPU-pinned inflate
but on the CUDA-host inflate too. The lossless guarantee is hardware-independent by construction
(IEEE-exact float64 ctx tables + deterministic torch reshape) and is now empirically validated on
the CUDA axis at the distortion level, not just decode-parity. **The win is rate-only on every axis.**

---

## 1. Apples-to-apples + axis tags

- `[contest-CUDA]` = NVIDIA T4, Linux x86_64, exact `archive.zip → inflate.sh → upstream/evaluate.py
  --device cuda`, 600 samples, archive sha `b46897267ded`, runtime-tree sha `12cd60fb…` (validated by
  the modal_auth_eval upload-tree projection; no FATAL = matched).
- `[contest-CPU]` = Modal Linux x86_64, exact `--device cpu`, 600 samples, same sha.
- The two axes are NOT inferred from each other; both are measured on 1:1 contest-compliant hardware.
- The recode rate delta (−1,326 B) lands identically on both axes: −0.00088293.

---

## 2. The rate-only win vs the standing pointers

| comparison | recoded-R3 CUDA 0.22528084 | delta |
|---|---|---:|
| vs R3-candidate CUDA (sha `1ccae18d`, 178,495 B) = 0.22616377 | rate-only improvement | **−0.00088293** |
| vs CUDA pointer (pr106 format0d, sha `9cb989ce`, 186,876 B) = 0.20533003 | does NOT beat | **+0.01995081** |

**No CUDA-frontier improvement.** The recoded-R3 CUDA (0.22528) is +0.020 above the standing CUDA
pointer (0.20533, pr106 format0d). This is the SAME structural result the prior CUDA-axis probe
documented for the R3-candidate: the PR110++ selector family's pose lever is contest-CPU-axis-only;
the CPU→CUDA pose drift (CUDA d_pose 0.00016857 vs CPU d_pose 0.00002942, ~5.7× worse) overwhelms
the byte savings on the CUDA axis. The recode is rate-only and cannot fix that drift — it preserves
the R3 distortion exactly, including the unfavorable CUDA pose.

**Pointer outcome:** NO pointer update.
- CUDA pointer unchanged (0.20533, pr106 family) — recoded-R3 does not beat it.
- CPU pointer already correct (0.19109982 on sha `b46897267ded`, set by the leapfrog verdict).
- `tools/refresh_canonical_frontier.py` NOT invoked to mutate the pointer because no axis improved.

This is the expected and correct result: **the contest leaderboard ranks by the CPU axis**, where
recoded-R3 (0.19109982) is the frontier and beats both PR #112 (0.19112577) and our prior frontier
(0.19198275). The CUDA axis is the secondary submission gate + a mechanism diagnostic.

---

## 3. BOTH-axis status on sha `b46897267ded` (the pre-submission gate)

| | value |
|---|---|
| archive sha256 | `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e` |
| archive bytes | 177,169 |
| `[contest-CPU]` (ranking axis) | **0.19109982** (Modal Linux x86_64, 600 samples) |
| `[contest-CUDA]` (secondary gate) | **0.22528084** (Modal T4, 600 samples) |
| CUDA−CPU gap | +0.03418102 (per-archive empirical; pose-driven) |
| runtime-tree sha256 | `12cd60fb407f117fbc14d55844f6af052f0af25fe104464836a11f5ac137e4cb` |

The recoded-R3 archive now satisfies the CLAUDE.md dual-eval mandate: BOTH axes measured on the
exact same sha, on 1:1 contest-compliant hardware, recomputed from components.

---

## 4. Pre-submission compliance check (`--contest-final --strict`, do NOT submit)

`scripts/pre_submission_compliance_check.py --contest-final --strict --submission-score-axis
contest_cpu` (the leaderboard-ranking axis), with BOTH auth-eval JSONs + archive manifest +
dispatch-claims ledger. Result JSON:
`experiments/results/cuda_pairing_recoded_r3_20260610/compliance_check_both_axis_contest_final.json`.

**Overall: NOT PASSING — 18 of 100 gates fail.** The failures partition into 4 categories:

### Category 2 — REAL submission-runtime findings (genuine engineering blockers)

| gate | finding | disposition |
|---|---|---|
| `submission_runtime_imports_within_allowlist` | `src/codec_ctx.py` imports `constriction`, which is NOT in the compliance allowlist (`brotli, codec, codec_sidecar, frame_selector, model, numpy, src, torch`) | **Dependency-closure class (CLAUDE.md L9, same class as PR #106 `brotli`).** `constriction` is a genuine NEW runtime dependency introduced by absorbing PR #112's entropy coder. BOTH the CUDA and CPU Modal evals PASSED with `constriction` installed (it is a contest-harness base dep per the codec_ctx.py provenance header), so it IS importable in a contest-equivalent runtime — but the static allowlist gate has not been told `constriction` is contest-available. Resolution requires operator confirmation that the contest T4/CPU runner provides `constriction` (or shipping it), then adding it to `RUNTIME_ALLOWED_NON_STDLIB_IMPORT_ROOTS` in `scripts/pre_submission_compliance_check.py`. |
| `submission_runtime_has_no_network_install_or_local_paths` | `src/codec_ctx.py:8` contains a GitHub PR #112 URL in the attribution header | The URL is a pure provenance/attribution comment (mirroring PR #112's own transparency), NOT a network install. The gate flags any `https://github.com/...` string. `codec_ctx.py` is vendored VERBATIM (editing it corrupts provenance per CLAUDE.md "Forbidden in-place edits to public PR intake clones" spirit). Resolution is operator policy: either accept the attribution comment (it carries the MIT license + author credit) or relocate attribution to a sidecar NOTICE file. |

### Category 1 — Adjudication-recording gates (clear once an adjudicated result-review packet lands)

`auth_eval_raw_promotion_policy_blockers_absent` + `contest_cpu_auth_eval_raw_promotion_policy_blockers_absent`.
The blockers (`pre_submission_compliance_check_not_recorded`, `result_review_packet_not_recorded`,
`cpu_leaderboard_reproduction_not_adjudicated`) are the canonical fail-closed markers that a RAW
auth-eval JSON is non-authoritative until an adjudicated result-review packet is recorded. This memo
+ the V3 candidate-action rows are that adjudication; the gate reads a structured `pre_submission_compliance.*`
custody artifact that the operator's final submit step writes.

### Category 4 — Selected-axis dispatch-claim terminal row

`dispatch_claim_successful_exact_eval_terminal_row` wants a terminal claim row marked successful-exact-eval
for the SELECTED axis (`contest_cpu`); the terminal row I wrote is for the CUDA dispatch (this lane).
The CPU-axis successful-exact-eval terminal row lives under the CPU sibling's lane
(`lane_pr110_payload_entropy_recode_20260610`). A submit packet gated on `contest_cpu` should pass
`--expected-lane-id`/`--expected-job-id` for the CPU lane's terminal row.

### Category 3 — PR-template packet content (operator-submit decisions)

`report_mentions_archive_sha256` / `report_mentions_archive_size_bytes` (the scored `report.txt` is
the byte-faithful evaluator output; augmenting it with sha/size would change the modal runtime-tree
hash and break the scored-tree binding — the sha/size belongs in a packet report OUTSIDE the runtime
tree), the 4 `post_deadline_policy_statement_*` gates (the PR-template "competitive or innovative"
answer, via `--competitive-or-innovative-statement[-file]`), `hosted_archive_manifest_supplied` (a
hosted `archive.zip` URL via `--hosted-archive-manifest-json`), the 4 `public_source_*` gates (public
repo link + pinned revision + reproduce command), and the 2 `public_evidence_contest_{cuda,cpu}_label_present`
gates (a public-facing text file with explicit `[contest-CUDA]`/`[contest-CPU]` axis labels, scanned
via `--public-scan-path`). All are content the operator authors at submit time.

### What this subagent fixed (non-destructive, did not pollute the scored runtime tree)

- Built `experiments/results/pr110_payload_entropy_recode_20260610/archive_manifest.json` (member-exact;
  placed OUTSIDE submission_dir so it does not alter the scored runtime-tree sha `12cd60fb`) — cleared
  the `archive_manifest_*` gates.
- Appended a terminal dispatch-claim row binding the exact scored runtime-tree sha `12cd60fb` — cleared
  `dispatch_claim_terminal_runtime_tree_sha_bound`.
- Removed `__pycache__` build artifacts from submission_dir/{src,encoder} (archive sha unchanged; the
  manifest builder already excludes `.pyc`).

---

## 5. One-command-ready submission (operator's call — do NOT submit without operator approval)

The recoded-R3 archive is the CPU-axis frontier and the leapfrog winner. To make the contest-final
gate pass, the operator must supply the Category-2 disposition + Category-3 packet content. The
single command, once those inputs exist:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr110_payload_entropy_recode_20260610/submission_dir \
  --archive experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/archive.zip \
  --auth-eval-json experiments/results/cuda_pairing_recoded_r3_20260610/recoded_cuda_eval/contest_auth_eval.json \
  --contest-cpu-auth-eval-json experiments/results/pr110_payload_entropy_recode_20260610/recoded_cpu_eval/contest_auth_eval.json \
  --archive-manifest-json experiments/results/pr110_payload_entropy_recode_20260610/archive_manifest.json \
  --contest-final --submission-score-axis contest_cpu \
  --expected-archive-sha256 b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e \
  --expected-archive-size-bytes 177169 \
  --expected-runtime-tree-sha256 12cd60fb407f117fbc14d55844f6af052f0af25fe104464836a11f5ac137e4cb \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id lane_pr110_payload_entropy_recode_20260610 \
  --expected-job-id <cpu-axis-terminal-job-id> \
  --competitive-or-innovative-statement-file <pr_template_statement.md> \
  --hosted-archive-manifest-json <hosted_archive_manifest.json> \
  --public-scan-path <pr_body_with_axis_labels.md> \
  --strict
```

Then the actual contest PR (per `docs/submission_template.md`):
`gh pr create --repo commaai/comma_video_compression_challenge ...` with the hosted archive URL.

Attribution credits PR #101 (@SajayR), PR #95 arch, PR #98 channel bias, PR #110 (@adpena = us)
selector/inflate chain, and PR #112 (mattneel) entropy-coder technique.

---

## 6. Routing + wire-in (Catalog #125)

### Verdict

**CONFIRMED: recoded-R3 BOTH-axis measured.** CPU = 0.19109982 (frontier, ranking axis); CUDA =
0.22528084 (secondary gate, does not improve the CUDA pointer). The lossless recode is axis-invariant
(distortion identical to R3-candidate on CUDA). The CPU-axis win is real and submission-eligible
pending packet finalization. No CUDA-frontier improvement (pr106 format0d 0.20533 remains CUDA champion).

### Wire-in (Catalog #125)

- **Hook #5 continual-learning:** CUDA row ingested via `tools/ingest_exact_eval_to_candidate.py`
  (`candidate_action_evaluation_recoded_r3_cuda_pairing.v1.json`); ΔS-vs-CUDA-frontier = +0.01995,
  `pays_rent=False`, verdict `INSPECT_BINDING_CONSTRAINT` (never auto-kills per Forbidden premature KILL).
- **Hook #6 probe-disambiguator:** RESOLVED the probe "does the −1,326 B lossless recode improve the
  CUDA axis?" → YES it lowers CUDA by exactly −0.00088293 (rate-only, axis-invariant) but does NOT
  beat the CUDA pointer (still +0.020 above pr106). The recode is a CPU-frontier rate win; the CUDA
  axis remains pose-drift-bound.
- **Hook #2 Pareto:** confirms the CUDA-axis binding constraint is pose distortion under CUDA kernels
  (CUDA d_pose 5.7× the CPU d_pose), not bytes — the same drift the prior CUDA probe documented.
- **Hook #1 sensitivity / #3 bit-allocator / #4 autopilot-dispatch:** N/A — this is a measurement +
  pre-submission-gate landing; the lossless recode opens no new bit-allocator/sensitivity input.

### DROP-IN HARDENING (carry forward)

Any future PR110++/recode candidate intended for submission MUST be dual-axis evaluated BEFORE the
submission packet is finalized; the lossless-recode class is axis-invariant in distortion (so its CUDA
score is fully predictable from any prior CUDA distortion measurement of the underlying candidate +
the new rate term), but the contest-final gate still requires both axis artifacts on the exact sha.

### Cost

1 Modal T4 CUDA dispatch (`fc-01KTRCQ6KYPYFNYB83GKYDYKXE`), completed in ~1.5 min on a warm T4 →
≈ **$0.03–0.05** (well under the <$1 budget). No fail-closed aborts.

---

## 7. Provenance

- Recoded archive sha256 `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e`, 177,169 B.
- CUDA eval: `experiments/results/cuda_pairing_recoded_r3_20260610/recoded_cuda_eval/` (contest_auth_eval.json,
  validation, provenance, report.txt, inflated_outputs_manifest.json). call_id `fc-01KTRCQ6KYPYFNYB83GKYDYKXE`.
- CPU eval (sibling, prior): `experiments/results/pr110_payload_entropy_recode_20260610/recoded_cpu_eval/`.
  call_id `fc-01KTRAYS68F3S0YWFT0CX35HDG`.
- Compliance: `experiments/results/cuda_pairing_recoded_r3_20260610/compliance_check_both_axis_contest_final.json`.
- Archive manifest: `experiments/results/pr110_payload_entropy_recode_20260610/archive_manifest.json`.
- Runtime-tree sha256 `12cd60fb407f117fbc14d55844f6af052f0af25fe104464836a11f5ac137e4cb`.
- R3-candidate CUDA reference (lossless verification baseline): sha `1ccae18d`, CUDA 0.22616377,
  d_seg 0.00066254, d_pose 0.00016857 (`experiments/results/cuda_axis_frontier_eval_20260610/candidate_cuda_eval/`).
