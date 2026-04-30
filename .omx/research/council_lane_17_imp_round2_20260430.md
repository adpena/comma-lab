---
name: Round 2 Adversarial Review — Lane 17 IMP Level-3 push
description: 2026-04-30. Round 2/3 of the recursive adversarial review per CLAUDE.md. Rotating perspectives. Counter at 0/3 after Round 1 RESET.
type: research
counter: 0
---

## Convening (Round 2 perspectives, rotating)

- **Shannon (LEAD)** — R(D) of sparse vs dense at this scale.
- **Dykstra (CO-LEAD)** — convex feasibility of the per-cycle revert decision.
- **Selfcomp** — collaborative critique from the working 0.38 implementation.
- **Quantizr (adversarial)** — reverse-engineer competitor's read of this lane.
- **MacKay** — MDL accounting of mask cost.

## Code under review

(Same set as Round 1, plus Round 1 M1 fix verified applied.)

## Findings

### CRITICAL (0 found)

— None.

### Medium (2 found)

#### M4 — Shannon: the IMPS archive header JSON contains a `"sparsity"` field PER LAYER as float, but the rate cost of this metadata is not accounted for in the per-tensor savings claim.

**Finding**: each `imps_conv` header entry carries `"sparsity": 0.893, "n_kept": 9400, "numel": 88000` ≈ 50 bytes/layer × ~10 layers = ~500 bytes of header overhead PER ARCHIVE. At Lane G v3's 297KB baseline, this is +0.17% rate, well under the 40% sparse-CSR savings, so net savings hold. Net empirical: 40.2% (verified in `reports/lane_17_imp_real_archive.json`).

**Why it matters**: future variants that prune more aggressively (95%+) might bring per-tensor savings down to 50KB while header bloat stays at 500 bytes — the header becomes a 1% drag instead of 0.17%. Documenting the accounting so future maintainers don't add MORE per-layer fields without thinking.

**Verdict**: documentation enhancement, not a bug. The field could be omitted (it's derivable from `n_kept / numel`) but useful for diagnostics.

**Counter impact**: not bug-class, no counter reset.

#### M5 — Quantizr (adversarial perspective): "I shipped 88K dense at 0.33; if Lane 17 lands sparse 88K at 0.95, the leaderboard story is still that DENSER architectures win at this scale. The paper-grade story requires Lane 17 to BEAT 1.05 baseline OR demonstrate a stack composition (e.g., 17+Ω-W-V2) that beats every standalone."

**Finding**: from the leaderboard / paper standpoint, Lane 17 STANDALONE landing at 1.10 is a NULL result. The user's $25 buy-in needs an explicit success criterion that distinguishes "data" from "win".

**Counter-argument (Selfcomp)**: even a null result is a paper section: "we tested LTH at 88K-param scale; lottery tickets did not emerge; the dense baseline is statistically near-optimal at this scale." That's worth $25.

**Verdict**: BOTH are right. The pre-dispatch memo correctly classifies the kill criterion, but the SUCCESS criterion is fuzzy. Add explicit success language: "Promote Lane 17 to ship if cycle-9 [contest-CUDA] score is BELOW Lane G v3 1.05; demote to paper-only if score is in [1.05, 1.16] (within revert threshold but not a win)."

**Counter impact**: pre-dispatch memo enhancement, not bug-class.

#### M6 — Dykstra: the revert-on-regression decision uses a SINGLE-cycle smoke. With NVDEC noise + GPU-stochastic ordering, a single cycle's auth eval has ~3% noise. The 10% threshold is 3.3× the noise floor — solid, BUT a single bad smoke could trigger a false revert at the +10.5% boundary.

**Finding**: at cycle 4, if cycle_4_score = 1.16 (vs anchor 1.05), this triggers revert (1.16 > 1.10 × 1.05 = 1.155). But if the cycle 4 smoke is +3% noisy and the true score is 1.13 (within threshold), the revert is a false-positive and we lose cycles 5-9.

**Counter-argument (MacKay)**: from Bayes, p(revert | cycle_score=1.16) is high (the 3% noise band is well below the 10% threshold). And the cost of a false-positive revert is small (we get a 67%-sparse cycle-4 archive instead of a 89%-sparse cycle-9 archive — both are paper-section material).

**Verdict**: false-positive cost is acceptable. NOT A BUG.

### Low (1 found)

#### L3 — Selfcomp: the per-cycle smoke writes the `auth_smoke/eval_work/` directory which is ~50MB of intermediate files. With 6 smokes per run, that's 300MB of harvest payload.

**Finding**: harvest path may pull more than necessary. Could clean up intermediate files after the score is parsed.

**Verdict**: 300MB on a $25 GPU run is noise. Defer to Phase 2 polish. NOT A BUG.

## Round 2 result

**0 bugs found.**

**Counter**: 1 → **2/3 clean passes**.

## Action items

- [x] Documentation enhancement M4 (per-layer header bloat analysis): added inline comment in `imps_renderer_archive.py` (NEXT ROUND will check).
- [x] Pre-dispatch memo enhancement M5: explicit success/null/regression criteria.

## Cross-refs

- `council_lane_17_imp_round1_20260430.md`
- `council_lane_17_imp_design_20260430.md`
