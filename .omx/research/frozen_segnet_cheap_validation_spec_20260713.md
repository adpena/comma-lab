# Task 454 implementation specification: frozen-SegNet cheap validation

Date: 2026-07-13 UTC  
Lane: `lane_454_segnet_cheap_validation_20260713`  
Authority: local mechanism and macOS-CPU advisory measurement only. `score_claim=false`; `promotion_eligible=false`.  
Review status: `pre-registered-only`.

STORES CONSULTED: `research(5715)`, `equations(622)`, `memory(1893)`, `dag(505)`, `council(277)`, `tasks(96)`, and `docs(92)` through `tools/corpus_query.py`; the landed YOPO provider and final receipt; the frozen-SegNet alternatives memo; the goldmine ledger; the operating manual; the current lane, task, probe-outcome, and subagent ledgers; the top-10 operator memory entries. Deliberately not consulted: paid/cloud state, live trainer state, the protected V9 run, and `upstream/evaluate.py`, because this is a zero-spend pointer-neutral mechanism probe.

## Answer and mathematical boundary

**DERIVED / pre-registered-only:** For anchor first-block feature `h0=f(x0)`, anchor label `a_p=argmax_c z_pc(h0)`, and anchor margin

`m_p = z_p,a_p(h0) - max_{c != a_p} z_p,c(h0)`,

protect only pixels whose anchor prediction equals the fixed target label. If an actual upper bound `L_p` satisfies

`|(z_p,a_p-z_p,c)(h) - (z_p,a_p-z_p,c)(h0)| <= L_p ||h-h0||_inf`

for every competing class and every `h` in the ball, then

`r_h = min_{p correct at anchor} m_p / L_p`

is a sufficient no-worsening radius: `||h-h0||_inf < r_h` preserves every anchor-correct pixel, while anchor-wrong pixels cannot add errors. If `||f(x)-f(x0)||_inf <= L_f ||x-x0||_inf` is also rigorously bounded, `r_x=r_h/L_f` is a sufficient input radius. Checking the feature displacement directly is tighter and needs only the already-cheap YOPO prefix.

**DERIVED / pre-registered-only:** A local Jacobian value is not a Lipschitz upper bound on a neighborhood. The first-block Jacobian also does not bound the downstream pairwise logit map. Therefore the existing first-block Jacobian plus the cached margin field cannot honestly produce a rigorous positive radius without a suffix bound. Code must distinguish `rigorous_upper_bound` from `empirical_local_estimate`; empirical rows must never use `certified`, `proof`, or `provably safe` as an authority field.

**DERIVED / pre-registered-only:** For the empirical proxy, calibrate

`Lhat_p = max_j |d_p(h_j)-d_p(h0)| / max(||h_j-h0||_inf, eps)`

on a pre-registered calibration subset, where `d_p=z_p,a_p-max_{c!=a_p}z_p,c`. The cheap event predicate is

`min_p [m_p - Lhat_p ||h-h0||_inf] > 0`.

It is a bounded heuristic, not a certificate. Its unsafe-acceptance rate is measured on disjoint candidates against the exact frozen SegNet. In the receipt, `false_negative` means **proxy accepts while exact d_seg worsens**.

No external theorem is imported for the triangle-inequality derivation above. The costate-reuse method is attributed at point of use to Dinghuai Zhang, Tianyuan Zhang, Yiping Lu, Zhanxing Zhu, and Bin Dong (2019), *You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle*, arXiv:1905.00877. The arXiv abstract identity was resolved on 2026-07-13 before this record.

## Owned files and exclusions

New isolated files only:

- `src/tac/boundary_math/segnet_validation_certificate.py`
- `src/tac/witness_dsl/segnet_validation_certificate_policy.py`
- `src/tac/canonical_equations/segnet_margin_trust_region_20260713.py`
- `src/tac/tests/test_segnet_validation_certificate.py`
- `src/tac/tests/test_segnet_validation_certificate_policy.py`
- `src/tac/canonical_equations/tests/test_segnet_margin_trust_region_20260713.py`
- `tools/probe_segnet_validation_certificate.py`
- `src/tac/tests/test_probe_segnet_validation_certificate.py`

The implementation must not edit the dirty shared provider, shared DSL policy, canonical-equation initializer or registry, live trainer, upstream evaluator, protected run, or any sibling-owned file. The probe may import and reuse the landed YOPO harness/provider and settled renderer.

## Smallest mechanism

1. A NumPy reference derives the global feature radius from anchor margins, anchor-correct mask, and either rigorous or empirical per-pixel pairwise-logit change bounds. It refuses nonfinite arrays, shape mismatches, nonpositive correct-pixel margins, empty protected sets, unlabelled bound authority, and empirical inputs presented as rigorous.
2. A cheap gate checks current first-block feature displacement and returns a typed `ACCEPT`, `REFRESH`, or `BLOCKED` decision. `ACCEPT` is certificate-authoritative only for a rigorous bound. An empirical accept is named `PROXY_ACCEPT`.
3. A confusion accumulator reports unsafe accepts, safe rejects, exact-safe accepts, and exact-unsafe rejects. It includes positive and negative meter canaries.
4. A typed DSL policy has no loose trainer flag. Its control law is event-conditioned: reuse while the appropriate gate accepts; refresh immediately on ball exit, custody change, nonfinite data, or empirical rejection. Rigorous mode requires a supplied rigorous-bound artifact. Empirical mode requires a calibration receipt and remains advisory.
5. The equation module records the sufficient condition, the missing-suffix-bound caveat, and the distinction between rigorous and empirical authority. No shared initializer edit is permitted; direct module discovery is tested.

## Real probe and pre-registered controls

The probe reuses pair 0, the three sealed early/boundary/late checkpoints, the settled renderer, exact `contest_r`, frozen CPU SegNet, YOPO first-block split, seed `20260712`, and the existing fractional ladder `1e-2 * 0.5^j`. It may measure fewer candidates only if the bit-identical termination predicate fires. It writes atomically under `experiments/results/segnet_validation_certificate_<UTC>/` and records source hashes, argv, environment, candidate order, and per-candidate timings.

- **P4 canaries:** positive synthetic linear map inside its known bound; negative map just outside; empirical proxy unsafe-accept counter must fire on a forged unsafe row.
- **P5 in-run control:** every proxy candidate is compared with the exact scorer on the identical rendered frame; no borrowed exact arm.
- **P6 sequence:** candidates retain ladder order and anchor identity; no per-frame shuffling.
- **P7 falsifiers:** rigorous mechanism is `NO-GO` if no actual suffix upper bound exists or the positive radius accepts no non-anchor real candidate. Empirical proxy is `NO-GO` if any holdout unsafe accept occurs. A throughput `GO` additionally requires at least `1.3x` derived whole-step economics at held exact `d_seg` and `d_pose` non-worsening for accepted candidates.
- **P8 floor:** exact d_seg/d_pose comparison has a zero within-run floor on deterministic identical inputs. Across-seed variance remains `UNKNOWN` because this is the registered single-seed spine.

## Economics and verdict scope

For each measured cadence `K`, report

`speedup = K*t_exact / (t_exact + (K-1)*(t_approx + t_validate_cheap + t_fallback))`.

`t_exact` and `t_approx` may be loaded from the content-addressed YOPO receipt only when their exact row identity and scope are retained. `t_validate_cheap` is measured by this probe. `t_fallback` is zero only for accepted candidates that exact measurement confirms safe; otherwise it includes the measured exact fallback cost. Component arithmetic is `DERIVED`; timings and exact scorer outcomes are `MEASURED`.

The terminal verdict is scoped to pair 0, the three named saved regimes, the landed blocks[0] split, the registered candidate ladder, macOS CPU advisory execution, and this exact proxy/certificate formulation. It cannot kill trust regions, margin proxies, YOPO, or frozen-scorer reuse as families.

