# Shannon-Floor Execution Readiness Plan

**Date:** 2026-04-30  
**Mandate:** drive from Lane G v3 `1.048866` toward the Shannon-floor band with maximum safe parallelism, no conservative under-dispatch, and no relaxation of contest rigor.  
**Primary references:** `grand_council_paradigm_shift_to_shannon_floor_20260430.md`, `contest_grade_all_lane_results_audit_20260430.md`, `all_scores_forensic_audit_20260430.md`, `recoverable_lanes_re_engineering_plans_20260430.md`.

---

## 1. Non-Negotiable Operating Standard

Aggressive execution is required. Evidence inflation is not allowed.

1. A lane can be designed, implemented, and dispatched from `[prediction]` evidence.
2. A lane can only promote/kill/anchor stack math from exact archive + CUDA + component recomputation evidence.
3. A final submission candidate must pass the Grade A++ gate in `contest_grade_all_lane_results_audit_20260430.md`: exact archive custody, clean manifest, payload closure, `inflate.sh -> upstream/evaluate.py`, 600 samples, 30-minute inflate budget, and contest-equivalent device evidence.
4. CPU/MPS/proxy/byte-only results are useful for routing, never for promotion.
5. Every high-impact lane gets recursive adversarial review: 3 consecutive clean passes before deployment or after any surprising result.

Current frontier:

| Anchor | Score | Grade | Note |
|---|---:|---|---|
| Lane G v3 PFP16 | `1.043987524793892` recomputed | Grade A++ contest-grade | Exact Lightning AI Tesla T4 eval, `gpu_t4_match=true`, SHA `0af839...ded7f`, 600 samples |
| Lane G v3 | `1.048866` recomputed | Grade A score-grade | Clean 3-file archive, CUDA RTX 4090, not A++ T4-matched |
| Lane G v3 + Ω-W-V2 | `1.070101` recomputed | Grade B diagnostic | Exact archive SHA `eba8e436...` not preserved locally; PoseNet regression lesson is credible |

---

## 2. Shortest Path Strategy

The grand-council stack is right: **β -> α -> γ**, with δ/ε/ζ only after measured components exist.

But wall-clock optimal execution is not serial. Run independent streams in parallel:

1. **β foundation:** sensitivity maps, Ω-W-V3, score-aware loss/byte allocation.
2. **α mask payload:** NeRV now; VQ-VAE/wavelet/STC-residual as parallel backup designs after NeRV's first CUDA truth point.
3. **Renderer compression:** IMP and Ω-W-V3 independently, then stack if each lands.
4. **Pose crumbs:** PFP16 immediately; Pint12-PCA next if PFP16 validates.
5. **Hidden-gem recovery:** re-engineer high-EV bugged lanes that test distinct hypotheses: Q-FAITHFUL, H-V3, SegMap clone, FL chunked, MAE-V.
6. **γ coordinator:** Joint-ADMM, Ballé/hyperprior, arithmetic, and bit optimizer only after at least one mask lane and one renderer lane produce exact archive evidence.

No new scorer-sensitive retraining lane may dispatch before
`.omx/state/lane12_nerv_l2_clearance.json` records
`cleared_for_retraining_unblock=true`, `lane12_l2=true`,
`geometry_gate_passed=true`, `grand_council_clean_passes>=3`, and evidence
paths. Build-only, harvest, and exact-eval-only lanes may continue; they do
not create new scorer-sensitive retraining noise.

---

## 3. Immediate Parallel Work Queue

### Wave 0 — Evidence Hygiene And Harvesting

| Work | Owner path | Done when | Priority |
|---|---|---|---|
| PFP16 A++ exact T4 bundle | `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json` | SHA `0af839...ed7f`, Tesla T4, `gpu_t4_match=true`, CUDA, 600 samples, recomputed `1.043987524793892` | P0 freeze/bundle |
| Recover/rebuild Ω-W-V2 exact archive | `experiments/results/lane_g_v3_omega_w_v2_stack_landed/` | archive matching SHA `eba8e436...1625` or rebuilt equivalent | P0 |
| Harvest active Lane 19/8/17/H-V3/SA/HM-S runs | live API plus lane-local artifacts | exact archives, logs, result JSONs copied locally | P0 |
| Retag all non-grade claims | `contest_grade_all_lane_results_audit_20260430.md` | no CPU/MPS/proxy score appears in authoritative table | P0 |

### Wave 1 — Floor-Moving Independent Dispatches

| Lane | Hypothesis | Current evidence | Next action | Promotion gate |
|---|---|---|---|---|
| β sensitivity-map | scorer derivatives identify safe bits | design + older sensitivity tooling | implement cross-validated per-channel score sensitivity | train/holdout sensitivity stability within 10%; used by Ω-W-V3 |
| Ω-W-V3 | recover Ω-W-V2 rate save without PoseNet pay | Ω-W-V2 diagnostic: `-0.034` rate, `+0.052` PoseNet pay | implement `src/tac/owv3_sensitivity_weighted.py` after β | exact archive score in `[1.025,1.045]` or better, PoseNet regression <= 20% |
| Lane 12 alpha redesign | replace 421KB masks with a scorer-preserving compact representation | current NeRV `jsonfix40` exact-CUDA retired for the measured implementation/config only at `26.03719330455429` from PoseNet collapse | diagnose geometry/temporal failure, redesign objective before rerun | exact archive eval beats PFP16 without PoseNet/SegNet collapse |
| Lane 19 logit margin | protect SegNet boundary pixels | CPU smoke, tests pass | CUDA A/B against Lane G v3 | score < 1.05, fragility histogram decays |
| Lane 17 IMP | sparse renderer lowers renderer bytes | CPU codec shows 40.2% byte saving at cycle 9 | continue active 10-cycle CUDA run | per-cycle auth eval, ship best cycle, no >10% regression |
| Lane PFP16 | zero-risk pose-byte reduction | A++ exact T4 score `1.043987524793892`; 7,439 bytes saved | freeze deploy baseline and bundle provenance | current baseline until beaten by exact archive evidence |
| Lane 8 multipass | compress-time score feedback improves archive | offline byte proxy only | CUDA exact-archive run | only promote if actual upstream score improves |

### Wave 2 — High-EV Recovery Bundle

Run these in parallel if GPU slots are available; they are different mechanisms, not redundant retraining:

| Lane | Why it belongs | Fix |
|---|---|---|
| Q-FAITHFUL | closest path to Quantizr-class architecture | fix dispatch args and FP4A/export gate ordering |
| H-V3 / V-family rebuild | half-frame trick is proven externally by Quantizr | fix channel path and train/inflate distribution |
| SegMap clones | Selfcomp-class path was OOM-bugged, not falsified | chunk/bf16 SegMapTrainer and verify post-harvest |
| FL chunked | RAFT pose path failed from OOM only | chunk RAFT inference on frames |
| MAE-V | crashed on missing `pydantic` | fix image dependency, dispatch |

These are allowed to run while Lane 12 is active because their failure modes and scientific hypotheses are distinct.

---

## 4. Stack Readiness Rules

No stack experiment gets launched merely because two predictions look additive.

| Stack | Launch condition | Stop condition |
|---|---|---|
| Redesigned alpha + PFP16 | redesigned alpha exact archive exists; PFP16 A++ archive validates | SegNet or PoseNet regression eats rate gain |
| NeRV + Ω-W-V3 | β sensitivity map passes CV; Ω-W-V3 standalone improves or flatlines PoseNet | PoseNet > 1.2x Lane G v3 or archive not smaller |
| IMP + Ω-W-V3 | IMP best-cycle archive exists; survivor-weight codec implemented | sparse mask overhead dominates or score regresses >10% |
| Lane 19 + NeRV | Lane 19 improves boundary metrics or score; NeRV underfit localized to boundaries | confident-wrong pixels increase |
| Full γ ADMM | at least one α and one β/renderer component are Grade A score-grade | KKT residual oscillation or hyperprior side-info > 50% of saved bytes |

The composition order remains:

```text
sensitivity analysis -> representation -> prediction/transform -> quantize/water-fill
-> hyperprior -> arithmetic -> deterministic pack -> contest_auth_eval
```

---

## 5. Mathematical Focus

The score derivative controls dispatch priority:

```text
score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * bytes / 37,545,489
```

1. Every 100KB saved is `0.06659` score.
2. The 421KB mask stream is the dominant rate lever; a 300KB mask reduction is about `0.1998` score before distortion.
3. PFP16's 7,439 bytes is only `0.00495`, but nearly free and should be banked.
4. Ω-W-V3 can only matter if it protects PoseNet; uniform Ω-W-V2 already proved rate wins are useless when PoseNet pays more.
5. ADMM/hyperprior/bit optimizers are second-order until α and β create a better base.

No "meat left on the bone" policy means we take the small deterministic wins while the large uncertain lanes run, not instead of them.

---

## 6. Review Loop

For every lane reaching local smoke or CUDA result:

1. **Shannon pass:** recompute score, byte slopes, and R(D) claim.
2. **Yousfi/Fridrich pass:** identify scorer blind-spot exploitation and boundary/sensitivity risks.
3. **Contrarian pass:** check artifact custody, no-op/encode-discard, CPU/MPS/proxy leakage, and stack conflict.
4. **Hotz/Carmack pass:** identify the fastest hardening patch or kill.
5. **Dykstra/Boyd pass:** check composition, KKT/waterline, ADMM convergence, and nonconvex failure.

Three clean passes are required to promote. Any issue resets the counter and creates a concrete patch or rerun.

---

## 7. Current Decision

Execute aggressively:

1. Freeze PFP16 A++ as the deploy baseline and complete the provenance/paper bundle.
2. Keep Lane 19, Lane 8, Lane 17, H-V3, SA, and HM-S harvest paths active, accepting only lane-local exact artifacts.
3. Implement β sensitivity-map artifacts and redesign Ω-W-V3 after the suspicious Modal size-regression smoke; this is foundational for the next frontier jump but not yet score evidence.
4. Queue the Wave 2 recovery bundle as GPU capacity permits; do not let Lane 12 block unrelated hidden-gem recovery.
5. Do not call any result floor-moving until it survives exact archive scoring and adversarial review.

This is the highest-throughput path: parallelize independent hypotheses, stack only measured components, and convert every prediction into archive-backed evidence as fast as possible.

---

## 8. Implementation Update — 2026-04-30

Status after the first implementation pivot:

1. **β / Ω-W-V3 scaffold landed.** `src/tac/sensitivity_map.py` defines the per-Conv2d-channel artifact contract, CUDA-authoritative device gate, save/load path, and CV-distance helper. `src/tac/owv3_sensitivity_weighted.py` now implements the mixed-channel archive: high-sensitivity output channels are stored FP16, lower-sensitivity channels are packed through OWV2.
2. **OWV3 decode is contest-path reachable.** `OWV3` is in the canonical magic registry and `submissions/robust_current/inflate_renderer.py` dispatches `magic == b"OWV3"` without scorer imports. Tests cover magic uniqueness, mixed-channel round trip, protected-channel FP16 fidelity, callable forward pass, and inflate-loader dispatch.
3. **Lane 12 `.nrv` packaging is partially unblocked.** `contest_auth_eval.py` now accepts `.nrv` archive members and the inflate mask resolver can find `masks.nrv` when callers still pass the legacy `masks.mkv` default. This removes a validation/resolution blocker, but does not promote Lane 12.
4. **Lane 12 current implementation is retired.** The `jsonfix40` NeRV archive has exact CUDA negative evidence. Future alpha work must start from a redesigned geometry/temporal/PoseNet-preserving objective and then repeat exact archive SHA custody plus `contest_auth_eval.py -> inflate.sh -> upstream/evaluate.py`.
5. **γ remains a coordinator, not a near-term full learned codec import.** Council research points to hyperprior-lite/static-ANS/range coding and actual-byte MDL gates for existing qint/mask/pose streams. Full Ballé/DCVC-style stacks are too dependency-heavy unless a component stream first creates measured heteroscedastic payload value.

Evidence label: OWV3 and `.nrv` resolver changes are implementation readiness only. They are not Grade A score-grade evidence and must not be used to rank, promote, or kill lanes until exact archive CUDA eval exists.

## 9. Reconciliation Update — 2026-04-30T16:45Z

The controlling baseline is now PFP16 A++ from Lightning AI Tesla T4:
`experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json`,
score `1.043987524793892`, archive SHA
`0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
archive bytes `686635`, `n_samples=600`, `gpu_t4_match=true`.

Lane 12 NeRV `jsonfix40` is no longer pending. Exact CUDA evidence at
`experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json`
retires that measured implementation/config with recomputed score
`26.03719330455429` and PoseNet `49.77849960`. This does not kill all
alpha/mask compression; it requires a redesigned, scorer-preserving alpha
objective before more NeRV/INR spend.

OWV3/Fisher Modal smoke produced real build artifacts, but no exact eval and a
rate-regressing archive: `912971` bytes, `+218897` vs Lane G v3, SHA
`710cba0c7c490b13db8b0aee897dd0f33cb8b66a6ed229466bf0d1aea392f5a3`.
This is suspicious negative smoke only. It should trigger encoder overhead and
configuration review, not a broad beta-method kill.
