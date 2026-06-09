# The Pact Evidence Constitution

UTC 2026-06-09 · claude · the human-readable canonical law for V3's proof lattice. The ENFORCEMENT
is in code (`tools/ingest_exact_eval_to_candidate.py` + `tac.optimization.harvest_evidence`); this
document is the contract that code implements. Every agent (Claude, Codex, future) obeys it.

## The three orthogonal questions every evidence row must answer
1. **`authority_tier`** — *where/how was it measured?*
2. **`metric_family`** — *what kind of number is it?*
3. **eligibility** — *what may this row legally change?*

A row missing any of the three is INADMISSIBLE — it may not rank, route, promote, or update anything.

## `authority_tier` (where/how)
| tier | meaning | promotion |
|---|---|---|
| `telemetry_proxy` | training loss / proxy during a run | never |
| `exact_cpu_advisory` | `evaluate.py` on local macOS CPU/MLX — a real number, NON-contest hardware | never |
| `contest_cpu` | `evaluate.py` on Linux x86_64 (the public-leaderboard axis) | only paired with cuda |
| `contest_cuda` | `evaluate.py` on NVIDIA T4/equivalent (the CUDA axis) | only paired with cpu |

## `metric_family` (what kind) — the anti-metric-laundering axis
| family | meaning | is it a score? |
|---|---|---|
| `telemetry_loss` | raw training loss | NO |
| `psnr_capacity` | RGB/Y reconstruction PSNR (carrier ability) | NO — capacity, not score |
| `scorer_proxy` | MLX/training scorer surrogate (e.g. boundary-hinge) | NO — surrogate, not official d_seg |
| `exact_pair_scorer` | the repo's own scorer over inflated frames, per-pair (diagnosis) | partial — advisory only |
| `exact_evaluate` | upstream `evaluate.py` on the full archive (d_seg/d_pose/bytes + report) | YES — the official score |

**Why both axes:** `authority_tier` alone is insufficient. A 21.74 dB `psnr_capacity` row measured by a
real local eval is `exact_cpu_advisory` by authority — but it is NOT a score. Without `metric_family`,
a capacity row could share schema shape with a score row and silently launder into a score claim.
`metric_family` makes the KIND explicit so that can never happen.

## Eligibility (what it may legally change) — the falling rule
- **`mechanism_update_eligible`** = TRUE for any real measurement
  (`metric_family ∈ {exact_evaluate, exact_pair_scorer, psnr_capacity, scorer_proxy}`).
  May direct *which experiment runs next*. NEVER promotes. (Arm A's 21.74 dB lives here.)
- **`score_roadmap_update_eligible`** = TRUE iff ALL of:
  `authority_tier ∈ {contest_cpu, contest_cuda}` **AND** `metric_family == exact_evaluate` **AND**
  d_seg + d_pose + archive_bytes + evaluate.py report all present. May change the *score roadmap /
  frontier*.
- **`promotion_update_eligible`** = TRUE iff **PAIRED** `contest_cpu` AND `contest_cuda` `exact_evaluate`
  on the **SAME `archive_sha256`**. A single-axis row can NEVER promote (the CUDA−CPU gap is empirical
  and per-archive; both axes required on identical bytes).

## The contest law this serves (the only objective)
`S(A) = 100·d_seg(A) + sqrt(10·d_pose(A)) + 25·|A|/37,545,489`, computed by `upstream/evaluate.py` on
the EXACT `archive.zip` bytes + the inflated SegNet/PoseNet frames. SegNet reads only the last frame of
each pair (argmax disagreement); PoseNet reads both frames via RGB→YUV6 (MSE on 6 pose dims); rate is
the compressed archive size. Nothing else is a score. The frontier literal lives ONLY in
`.omx/state/canonical_frontier_pointer.json` (never hardcoded).

## The non-negotiable rules (binding on every agent)
1. **Do not branch from `telemetry_proxy` or `psnr_capacity`.** They direct mechanism, never the score roadmap.
2. **Do not claim score from advisory.** `exact_cpu_advisory` is a real number but non-promotable.
3. **Every candidate from every lane becomes a typed `CandidateActionEvaluation`** with archive_sha256,
   d_seg, d_pose, bytes, score, ΔS-vs-base, authority_tier, metric_family, base_archive_sha256, stale flag.
   No lane is "important" without this row.
4. **ΔS is the only admission criterion:** admit an action σ iff exact `ΔS(base+σ) < 0`.
5. **Never auto-kill** (Forbidden premature KILL): a high score routes to INSPECT, never KILL.
6. **No upstream edits** for authority (pinned `evaluate.py` is the law; per-pair quantiles come from a
   SEPARATE repo-side `exact_pair_scorer` pass, never by patching upstream).

## One sentence
**A measurement may change exactly what its weakest of {authority_tier, metric_family, completeness}
permits — and the score roadmap moves only on a contest-axis `exact_evaluate` row, the frontier only on
the paired CPU+CUDA on identical bytes.** That is V3's proof system.
