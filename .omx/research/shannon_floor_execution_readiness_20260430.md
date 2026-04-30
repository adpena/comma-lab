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

The previous "do not spawn new retraining lanes until Lane 12 L2" should be interpreted narrowly: do not launch duplicate same-hypothesis retraining noise. It does not block independent β work, pose-byte work, data-side lanes, or re-engineering lanes with different mechanisms.

---

## 3. Immediate Parallel Work Queue

### Wave 0 — Evidence Hygiene And Harvesting

| Work | Owner path | Done when | Priority |
|---|---|---|---|
| PFP16 exact CUDA eval | `experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip` | `contest_auth_eval.json` with SHA `0af839...ed7f`, CUDA, 600 samples | P0 |
| Recover/rebuild Ω-W-V2 exact archive | `experiments/results/lane_g_v3_omega_w_v2_stack_landed/` | archive matching SHA `eba8e436...1625` or rebuilt equivalent | P0 |
| Harvest active Lane 12/19/8/17 runs | `.omx/state/active_dispatches.md` | exact archives, logs, result JSONs copied locally | P0 |
| Retag all non-grade claims | `contest_grade_all_lane_results_audit_20260430.md` | no CPU/MPS/proxy score appears in authoritative table | P0 |

### Wave 1 — Floor-Moving Independent Dispatches

| Lane | Hypothesis | Current evidence | Next action | Promotion gate |
|---|---|---|---|---|
| β sensitivity-map | scorer derivatives identify safe bits | design + older sensitivity tooling | implement cross-validated per-channel score sensitivity | train/holdout sensitivity stability within 10%; used by Ω-W-V3 |
| Ω-W-V3 | recover Ω-W-V2 rate save without PoseNet pay | Ω-W-V2 diagnostic: `-0.034` rate, `+0.052` PoseNet pay | implement `src/tac/owv3_sensitivity_weighted.py` after β | exact archive score in `[1.025,1.045]` or better, PoseNet regression <= 20% |
| Lane 12 NeRV | replace 421KB masks with <80KB payload | 94.4% byte saving, 2.003% argmax disagreement, CPU partial | finish CUDA run and exact archive eval | payload <= 100KB and SegNet <= Lane G v3 + 25% |
| Lane 19 logit margin | protect SegNet boundary pixels | CPU smoke, tests pass | CUDA A/B against Lane G v3 | score < 1.05, fragility histogram decays |
| Lane 17 IMP | sparse renderer lowers renderer bytes | CPU codec shows 40.2% byte saving at cycle 9 | continue active 10-cycle CUDA run | per-cycle auth eval, ship best cycle, no >10% regression |
| Lane PFP16 | zero-risk pose-byte reduction | 7,439 archive bytes saved; predicted `1.045047` | exact CUDA eval | no PoseNet regression; score improves by rate arithmetic |
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
| NeRV + PFP16 | NeRV exact archive exists; PFP16 exact archive validates | SegNet or PoseNet regression eats rate gain |
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

1. Keep Lane 12, Lane 19, Lane 8, and Lane 17 active or harvest them immediately.
2. Start/finish PFP16 exact CUDA eval.
3. Implement β sensitivity-map artifacts and Ω-W-V3 design next; this is foundational for the next frontier jump.
4. Queue the Wave 2 recovery bundle as GPU capacity permits; do not let Lane 12 block unrelated hidden-gem recovery.
5. Do not call any result floor-moving until it survives exact archive scoring and adversarial review.

This is the highest-throughput path: parallelize independent hypotheses, stack only measured components, and convert every prediction into archive-backed evidence as fast as possible.

---

## 8. Implementation Update — 2026-04-30

Status after the first implementation pivot:

1. **β / Ω-W-V3 scaffold landed.** `src/tac/sensitivity_map.py` defines the per-Conv2d-channel artifact contract, CUDA-authoritative device gate, save/load path, and CV-distance helper. `src/tac/owv3_sensitivity_weighted.py` now implements the mixed-channel archive: high-sensitivity output channels are stored FP16, lower-sensitivity channels are packed through OWV2.
2. **OWV3 decode is contest-path reachable.** `OWV3` is in the canonical magic registry and `submissions/robust_current/inflate_renderer.py` dispatches `magic == b"OWV3"` without scorer imports. Tests cover magic uniqueness, mixed-channel round trip, protected-channel FP16 fidelity, callable forward pass, and inflate-loader dispatch.
3. **Lane 12 `.nrv` packaging is partially unblocked.** `contest_auth_eval.py` now accepts `.nrv` archive members and the inflate mask resolver can find `masks.nrv` when callers still pass the legacy `masks.mkv` default. This removes a validation/resolution blocker, but does not promote Lane 12.
4. **Lane 12 still needs full proof.** Full CUDA NeRV training, clean payload closure for `tac.nerv_mask_codec` in the contest inflate environment, exact archive SHA custody, and `contest_auth_eval.py -> inflate.sh -> upstream/evaluate.py` score remain required before any promotion.
5. **γ remains a coordinator, not a near-term full learned codec import.** Council research points to hyperprior-lite/static-ANS/range coding and actual-byte MDL gates for existing qint/mask/pose streams. Full Ballé/DCVC-style stacks are too dependency-heavy unless a component stream first creates measured heteroscedastic payload value.

Evidence label: OWV3 and `.nrv` resolver changes are implementation readiness only. They are not Grade A score-grade evidence and must not be used to rank, promote, or kill lanes until exact archive CUDA eval exists.
