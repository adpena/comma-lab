# CUDA-axis paired frontier eval — the no-regret transfer probe (R3 candidate vs prior CUDA frontier)

**Subagent:** `cuda_axis_frontier_eval_20260610` · UTC 2026-06-10.
**Lanes:** `lane_pr110pp_r3_candidate_cuda_20260610` (candidate) +
`lane_pr106_format0d_cuda_frontier_control_20260610` (same-session control).
**Operator authorization:** MASTER_ROADMAP RANK 3, operator-approved with Modal spend.
**Mission:** score the current CPU-frontier archive (the R3 candidate, 0.19198275 [contest-CPU],
178,495 B) on the CONTEST-CUDA axis to (a) test whether the CPU-frontier transfers to CUDA and
beats the standing CUDA pointer (0.20533, a DIFFERENT archive, 186,876 B), and (b) satisfy the
required pre-submission gate (CLAUDE.md: BOTH axes on the same sha).

---

## 0. HEADLINE VERDICT — NO TRANSFER. Kill criterion HIT.

| eval | archive sha | bytes | final_score [contest-CUDA] (recomputed from components) | avg_posenet_dist | avg_segnet_dist | call_id |
|---|---|---:|---:|---:|---:|---|
| **candidate** (R3 CPU-frontier) | `1ccae18d…` | 178,495 | **0.22616377** | 1.6857e-04 | 6.6254e-04 | `fc-01KTQYSSV5PVR322RFBY771RK8` |
| **control** (prior CUDA frontier) | `9cb989ce…` | 186,876 | **0.20533003** | 3.188e-05 | 6.3042e-04 | `fc-01KTQYSVHNS2CWANF3CCZCANDG` |

- **ΔS (candidate − control) = +0.02083374** → the CPU-frontier archive scores **WORSE** on CUDA.
- **Kill criterion (candidate CUDA ≥ 0.20533) is HIT** (0.22616 ≥ 0.20533). The CPU-frontier does
  NOT transfer to the CUDA axis. No CUDA-frontier improvement; pointer unchanged.
- **Pre-registered prediction was TRANSFER** (CPU archive is 8 KB smaller + selector-corrected).
  **Prediction FALSIFIED** — the byte savings is real (rate −0.0056) but is overwhelmed by a pose
  penalty (+0.0232) that only appears on the CUDA axis.

Both evals: `passed=true`, `score_axis=contest_cuda`, validation `errors=None`. Both scores
**recomputed from components** (the rounding-trap discipline — the rounded `final_score` field said
0.23 / 0.21 respectively, which is 3 orders of magnitude too coarse to adjudicate a 0.02 gap; the
recomputed values match the formula to 8 decimals).

---

## 1. Apples-to-apples validation (the control IS the proof)

The same-session control re-eval of the prior CUDA frontier (sha `9cb989ce`) recomputed to
**0.20533002902019143** — **bit-exactly** the canonical CUDA pointer value
(`0.20533002902019143`, measured 2026-05-16 T4). Cross-session T4 reproducibility to the full
machine-precision of the stored pointer confirms:

1. The Modal T4 CUDA axis is rock-solid and deterministic (no intra-axis noise to gate against).
2. The +0.02083 candidate delta is a REAL, axis-valid difference, not measurement drift.
3. The eval substrate is 1:1 contest-compliant (NVIDIA T4, Linux x86_64, exact `upstream/evaluate.py
   --device cuda`, byte-identical inflate runtime per archive, 600 samples).

Axis tags are explicit per CLAUDE.md "Apples-to-apples evidence discipline": both rows are
`[contest-CUDA]` on T4; the candidate's CPU score (0.19198275 `[contest-CPU]`, Modal Linux x86_64,
call_id `fc-01KTQMWVQJJSP28GGBWZ91SYQC`) is the paired CPU half — the two axes are NOT inferred from
each other.

---

## 2. The mechanism — CPU→CUDA pose drift kills the selector lever

Component decomposition (`100·d_seg + √(10·d_pose) + 25·bytes/37,545,489`):

| term | candidate | control | Δ (cand − ctrl) |
|---|---:|---:|---:|
| `100·d_seg` | 0.066254 | 0.063042 | **+0.003212** |
| `√(10·d_pose)` | 0.041057 | 0.017855 | **+0.023202** ← the killer |
| `25·bytes/37.5M` (rate) | 0.118852 | 0.124433 | **−0.005581** (the 8 KB savings) |
| **total** | **0.226164** | **0.205330** | **+0.020834** |

The candidate's CUDA **pose distortion is ~5.3× worse** than the control (1.6857e-04 vs 3.188e-05).
Critically, the R3 verdict measured the candidate's **contest-CPU** pose at `2.942e-05` — essentially
identical to the control's CUDA pose. So the candidate's pose is fine on CPU but degrades ~5.7× on
CUDA. This is the **CPU→CUDA FP-drift class** (CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA":
"The CUDA−CPU gap is empirical and per-archive… pose component appears to be the dominant gap
source"). The R3 per-pair frame-0 selector (the 7-pair switch) was derived from the contest-CPU
host's PoseNet kernels; that per-mode pose ordering does NOT survive the CUDA kernels — the same
host-axis drift R3 documented at the macOS→Linux-CPU boundary, now manifest at the CPU→CUDA boundary.

The two archives are NOT the same substrate (the candidate is the PR110++/pr101-grammar selector
family; the control is the pr106 format0d latent-score-table family). So this is a cross-family axis
comparison, not a same-archive CPU/CUDA drift measurement — but the conclusion is the same: the
8 KB-smaller CPU-frontier archive is NOT a CUDA-frontier candidate.

---

## 3. Routing + wire-in (Catalog #125)

### Verdict

**DEFER the CPU-frontier → CUDA-frontier transfer hypothesis as FALSIFIED-at-this-archive** (not a
kill of any family; CLAUDE.md "Forbidden premature KILL"). The prior CUDA frontier (pr106 format0d,
0.20533) REMAINS the CUDA-axis champion. The R3 candidate is a contest-CPU-axis artifact only.

**Pre-submission gate status:** the R3 candidate now has BOTH axes on its exact sha
(`1ccae18d…`): CPU = 0.19198275, CUDA = 0.22616377. Per CLAUDE.md "Submission auth eval — BOTH CPU
AND CUDA", a submission packet must carry both. The candidate's CUDA score (0.22616) is **worse than
the contest's own CUDA reference cluster** and far above the CPU score it ranks on — it is a
CPU-leaderboard candidate whose CUDA axis is non-competitive. (The contest leaderboard ranks by CPU,
so the CPU score is the ranking-relevant number; the CUDA score is the secondary gate and a
mechanism diagnostic.)

### Wire-in (Catalog #125)

- **Hook #5 continual-learning:** two exact `[contest-CUDA]` rows ingested via the canonical
  `tools/ingest_exact_eval_to_candidate.py` path (candidate + control). Reseeds the V3 ΔS-judge with
  the FIRST CUDA-axis PR110++ transfer-probe rows.
- **Hook #6 probe-disambiguator:** RESOLVED the probe "does the 8 KB-smaller selector-corrected
  CPU-frontier transfer to CUDA?" → NO (pose drifts +0.023 on CUDA; net +0.021). The R3 per-pair
  selector lever is CPU-axis-only.
- **Hook #2 Pareto:** confirms the CUDA-axis binding constraint is the **pose distortion under CUDA
  kernels**, not bytes — the candidate paid back 0.0056 in rate but lost 0.023 in pose. CUDA-axis
  pose has more headroom than CPU-axis pose, but the CPU-tuned selector cannot capture it.
- **Hook #1 sensitivity / #3 bit-allocator / #4 autopilot-dispatch:** N/A — the result confirms a
  drift class and does not open a new bit-allocator/sensitivity input; it routes the CUDA frontier
  back to the pr106 family.

### Pointer

**NO pointer update.** The control reproduced the canonical CUDA pointer (0.20533002902019143)
bit-exactly; the candidate does NOT improve the CUDA frontier (0.22616 > 0.20533). The CPU frontier
pointer is unaffected (this work touched only the CUDA axis). `tools/refresh_canonical_frontier.py`
was NOT invoked to mutate the pointer because no axis improved.

### DROP-IN HARDENING RECOMMENDATION

Any future PR110++ per-mode / per-region / per-pair selector candidate that is tuned on the
contest-CPU host MUST be CUDA-paired BEFORE any submission or CUDA-frontier claim. The per-pair pose
ordering is CPU-kernel-specific and does not survive CUDA; the selector lever is contest-CPU-axis
only. A CUDA-frontier selector would have to derive the per-mode pose table on a CUDA host (the
analog of R3's "build the table on the contest host" fix, now at the CPU→CUDA boundary).

### Cost

2 Modal T4 CUDA dispatches (candidate + control), each completed in ~2 min on a warm T4 →
total ≈ **$0.05–0.10** (under the $5 STOP gate; well under the ~$1–2 budget envelope). Two earlier
fail-closed FATAL aborts (pairing-flag conflict + missing runtime-tree-hash + claim-conflict) cost
$0 — refused before provider upload/spend.
