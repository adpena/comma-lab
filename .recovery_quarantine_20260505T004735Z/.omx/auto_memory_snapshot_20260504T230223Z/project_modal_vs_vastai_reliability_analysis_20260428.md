# Modal vs Vast.ai reliability analysis

**2026-04-28 · platform-selection rubric**

## Today's NVDEC variability data

| Lanes attempted | NVDEC bad host rate |
|-----------------|---------------------|
| Cycle 1 (Ω-V2, EC, SAUG-V2) | 100% (3/3 hosts hit some launch failure) |
| Verified-survived in flight | ~20% (8 instances reached training) |
| Production destroys after auth-eval crashes | ≥1 today (Lane RM-d) |

Empirical 4090 reliability today: ~80% bad-host attrition before training
starts. Lane RM-d trained successfully but crashed at auth eval.

## Cost analysis for our long-training workload

Our typical training: ~12h on T4 (or ~2-3h on 4090).

| Platform | GPU | $/hr | Our 12h workload | Reliability | Effective $ |
|----------|-----|------|------------------|-------------|-------------|
| Vast.ai | RTX 4090 | $0.25 | 3h × $0.25 = $0.75 | ~30% (NVDEC + crashes) | $0.75 ÷ 0.30 = $2.50 |
| AWS spot | T4 g4dn | $0.22 | 12h × $0.22 = $2.64 | ~85% (spot reclaim) | $3.10 |
| Modal | T4 | $0.59 | 12h × $0.59 = $7.08 | ~99% | $7.15 |
| Lightning | T4 | $0.40 | 12h × $0.40 = $4.80 | ~95% | $5.05 |

Headline: Vast.ai 4090 is still cheapest IF the lane runs first try. With
today's 30% success rate the effective cost is competitive with AWS spot
and only ~3× cheaper than Modal — but Modal offers near-zero variance.

## Recommendation

* **Long training runs (>6h, anything we can't afford to lose):** Modal.
  Sub-Quantizr moonshots, Lane SZ Phase 2, Lane MAE-V, Lane W full self-
  compress training. The 2-3× cost premium buys us no NVDEC roulette + no
  Korea-region failures + no SSH rate-limiting + persistent volume for
  GT data.
* **Cheap moonshots (<2h, rapid iteration):** Vast.ai 4090. Lane sweeps,
  param exploration, FP4 sensitivity scans. The reliability penalty is
  acceptable when re-running costs <$1.
* **Auth eval only (15-30 min):** Modal first, Vast.ai as backup. Modal T4
  with the canonical `experiments/modal_auth_eval.py` is now the reliable
  fallback path when Vast.ai eats an instance mid-eval.

## Action items

1. Port Lane SZ Phase 2 to Modal as backup. SZ is the moonshot we cannot
   afford to lose (predicted band [0.30, 0.50]) and a 12h Vast.ai run with
   30% success = 70% chance of needing 2-3 reruns = $7-10 effective cost vs
   Modal's deterministic $7.
2. Use Modal for any lane that has >$3 of accumulated training spend before
   it produces an authoritative score. The premium is recoverable in one
   avoided re-run.
3. Keep Vast.ai for sweep orchestrators that produce 5-10 short outputs
   (the 30% attrition self-corrects via parallel launches).

## Cross-references

* `feedback_artifact_recovery_canonical_workflow_20260428` — the recovery
  workflow makes Vast.ai partial failures recoverable, narrowing the
  reliability gap.
* `feedback_vastai_nvdec_host_variation` — root cause of today's attrition.
* `feedback_vastai_launch_returns_success_before_lane_starts` — silent-non-
  start variant of the same reliability problem.
* `experiments/modal_auth_eval.py` — canonical Modal eval entry; updated
  2026-04-28 with archive-sha-keyed result sidecars.
* `tools/auth_eval_local.py` — local fallback when neither platform is
  available.

## Key insight

The right framing is NOT "Modal is more expensive, so use Vast.ai." It's
"factor in the re-run probability, and Modal becomes the cheaper option for
work we genuinely can't afford to lose." The Lane RM-d incident
($1.16 + 3.5h GPU + opportunity cost of stale baseline measurement) would
have been a $4-7 Modal eval that just worked.
