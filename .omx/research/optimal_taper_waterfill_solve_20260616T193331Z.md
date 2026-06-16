# Optimal byte-neutral d_seg-aware TAPER — WATERFILL SOLVE on the converged n600 basin

**`[contest-CPU advisory] NON-PROMOTABLE`.** No score claims. The map's Δd_seg are sensitivity
MEASUREMENTS (exact SegNet under per-tensor weight noise), not frontier claims; the solved taper is
a PROPOSAL the from-scratch A/B validates. HEAD `d3006c98d`. CPU-only, $0, no GPU touched.

## What this upgrades
The hand-tuned `dseg_aware_taper` heuristic (`[16,16,17,19,19,14,10]`, "pull early down, push
mid-late up") is replaced by a genuine **waterfill solve** driven by the MEASURED per-stage /
per-tensor d_seg-sensitivity map on the CONVERGED 600-pair small basis
(`torch_vehicle_full_mps_basin_bc20_n600/best`, base_ch=20, latent_dim=28, 83,356 params,
89,136-byte archive, d_seg=0.00260 / d_pose=0.00034 advisory). Per the rate-headroom reframe
(sister memo `small-basis-rate-headroom-is-the-sub015-asset...`), the small basis's rate+pose floor
is ~0.114 (< 0.15); sub-0.15 needs only `100·d_seg < 0.036 → d_seg < 0.00039`. The taper
REALLOCATES the fixed param budget across the 7 decoder stages at CONSTANT bytes (conv params are
`in·out·k²`, resolution-independent), so it lowers d_seg WITHOUT spending the rate headroom. This
solve replaces the heuristic guess with the measured marginal.

## 1. The measured sensitivity map (REAL exact_eval, NO-FAKE)
Probe: `experiments/probe_dseg_sensitivity_map_basin_n600.py` (clean variant of the gate-2 probe,
`_BASE` → the n600 basin; 20%-RMS Gaussian noise per weight tensor, fixed seed 1234, Δd_seg via
`RealScorerContext.exact_eval` on the contest SegNet). Eval over the first 96 latents (the
sensitivity SHAPE — which stage carries d_seg — is what the waterfill consumes; still the authority
exact_eval, just on a 96-pair subset for speed). Baseline d_seg=0.002548 (eval over 96 pairs;
consistent with the basin's 600-pair 0.00260). Output: `reports/dseg_sensitivity_map_n600.json`.

**Band shares (Φ3 test):**
| band | param-share | sensitivity-share | density-ratio | verdict |
|---|---|---|---|---|
| LOW (6×8–24×32) | 67.9% | 44.7% | **0.66×** | over-provisioned |
| MID (48×64–96×128) | 21.1% | 24.1% | 1.14× | ~balanced |
| HIGH (192×256–384×512) | 11.0% | 31.2% | **2.84×** | UNDER-provisioned |

**VERDICT: Φ3 CONFIRMED** — the HIGH-res band carries 31% of d_seg sensitivity on 11% of params →
the taper should widen it.

**Per-stage density (Δ-share / param-share — the waterfill input):**
`st0=0.46  st1=0.90  st2=0.80  st3=0.88  st4=1.36  st5=1.90  st6=3.88`

**Per-TENSOR density (the faithful objective input — a channel `c_j` appears in TWO tensors, so the
solve credits per-tensor, not per-stage):**
| tensor | stage | params | density |
|---|---|---|---|
| stem.weight | 0 | 26,880 | 0.46 (lowest, huge) |
| blocks.0.weight | 1 | 14,400 | 0.90 |
| blocks.1.weight | 2 | 14,400 | 0.80 |
| blocks.2.weight | 3 | 10,800 | 0.78 |
| blocks.3.weight | 4 | 5,940 | 1.18 |
| blocks.4.weight | 5 | 3,960 | 1.52 |
| blocks.5.weight | 6 | 3,600 | 2.09 |
| skips.2.weight | 3 | 300 | **4.48** |
| skips.3.weight | 4 | 165 | **7.78** |
| skips.4.weight | 5 | 110 | **15.56** |
| refine.0.weight | 6 | 450 | 2.25 |
| refine.1.weight | 6 | 450 | 1.54 |
| rgb_0.weight | 6 | 270 | 0.00 |
| rgb_1.weight | 6 | 270 | **38.19** |

Top sensitivity-DENSITY tensors: **rgb_1 (38.2), skips.4 (15.6), skips.3 (7.8), skips.2 (4.5)** —
all TINY (110–300 params) but extreme density; the 1×1 skips scale with BOTH adjacent widths, so
the way to feed them more capacity is to widen the channels they connect (c3,c4,c5). The stem
(26,880 params, density 0.46) is the over-provisioned sink to drain.

## 2. The waterfill solve (math + solved channels)
Tool: `experiments/solve_taper_waterfill.py`. The OBJECTIVE is the byte-neutral waterfill
`J(c) = Σ_tensor density_tensor · params_tensor(c)`: density (Δd_seg-per-param measured by perturbing
that tensor) × the tensor's actual param mass = predicted d_seg-reduction credit. Maximizing J at
fixed total params IS the waterfill — reallocate params from low-density tensors (stem) to
high-density tensors (the mid-late skips/blocks). The per-tensor mass formulas sum EXACTLY to
`decoder_param_count` (self-consistency verified).

**The solve = greedy STRICT-climb marginal equalization with a TRUST REGION:** repeatedly pick the
single byte-neutral +1/−1 channel pair that most increases J, accept only on a strict J increase
(finite bounded lattice ⇒ converges, no cycles), subject to: (a) total params within ±3% of vendored
(byte-neutral); (b) each stage within ±40% of its vendored width (TRUST REGION — the density is a
FIRST-ORDER marginal measured AT the vendored point; extrapolating to a 1-channel stage is outside
the measured regime and architecturally invalid, so the trust region keeps the solve where the
measured density is a faithful gradient); (c) final stage capped at the vendored final (10) — raising
it inflates `refine` QUADRATICALLY (a rate-blowup). It is a first-order/marginal solve: the
per-tensor density is the measured first-order sensitivity; the **downstream from-scratch A/B is the
authority** that validates the realized Δd_seg (the solve PROPOSES; the exact_eval DISPOSES).

**SOLVED taper (default final_cap=10):** `[22, 16, 15, 14, 15, 14, 10]` — 82,899 params, **−0.55%**
vs vendored 83,356 (byte-neutral; slightly UNDER, strictly better for the rate term). Converged in
10 moves (NOT max_iters). Output: `reports/taper_waterfill_solve_n600.json`.

| stage | vendored | heuristic | SOLVED | density |
|---|---|---|---|---|
| 0 | 20 | 16 | 22 | 0.46 |
| 1 | 20 | 16 | 16 | 0.90 |
| 2 | 20 | 17 | 15 | 0.80 |
| 3 | 15 | 19 | 14 | 0.88 |
| 4 | 11 | 19 | **15** | 1.36 |
| 5 | 10 | 14 | **14** | 1.90 |
| 6 | 10 | 10 | 10 | 3.88 |

### Heuristic-vs-solved diff (they DIFFER)
- **AGREE on the structural direction:** both widen the high-density mid-late stages (4: 11→15/19,
  5: 10→14/14) and both hold the final at 10 (refine-quadratic cap).
- **DIFFER on the early stages:** the heuristic pulls ALL of stages 0–3 DOWN uniformly
  (`16,16,17,19`), shoveling that budget into stages 3–4 (`19,19`). The SOLVE instead (i) widens
  stage4/5 (the measured high-density band) more modestly, and (ii) leaves stage0 at 22 — a
  COUPLING artifact: bumping c0 cheaply grows blocks.0 (density 0.90) as well as the stem
  (0.46), so the per-tensor objective credits the blocks.0 growth. This is a real but weak
  first-order effect and is exactly the kind of coupling distortion the trust region + downstream
  A/B exist to bound.
- **The heuristic over-rotated stage3 UP (15→19) on a stage whose MEASURED density is only 0.88**
  (below average) — the guess put capacity where the converged basin says d_seg is NOT especially
  sensitive. The solve corrects this (stage3 → 14, near vendored).
- **Net:** the solve is a smaller, measured first-order step from vendored; the heuristic was a
  larger hand-shaped step partly mis-aimed (stage3). The A/B arbitrates which realizes lower d_seg.

### The final-stage tension (honest caveat)
The single highest-density tensor is **rgb_1 (38.2)** at the final stage, but the final stage is
CAPPED at 10 because raising the final channel `cf` inflates `refine` (`cf·(cf//2)·9·2`) quadratically
— a rate-blowup that would forfeit the byte-neutrality the whole strategy depends on. A `--final-cap`
sweep confirms the trade: cap 11 → `[22,15,15,14,15,14,11]` (−1.68%), cap 12 → `[21,15,15,14,15,14,12]`
(−2.96%, band edge), cap 14 → solver refills the budget by inflating stage0 to 25 (incoherent,
−2.5%→+2.5%). The default cap=10 solve keeps the byte-neutral guarantee clean; a cap=11/12 variant is
a defensible secondary arm if the A/B shows the rgb_1 head is the binding d_seg lever. **rgb_1 being
the top tensor is itself a strong signal** — the last RGB→pixel conv (rgb_1 = frame_1, the d_seg
frame) is where SegNet argmax is decided; a from-scratch run may also benefit from a **wider final
OR a deeper rgb head** (orthogonal to the taper; out of scope here).

## 3. Parity verification (the solved taper is REAL + byte-neutral + parity-clean)
`ConfigurableTaperHNeRVDecoder(latent_dim=28, base_channels=20, channels=[22,16,15,14,15,14,10])`:
- instantiates; `nn` param count 82,899 == `decoder_param_count` (exact).
- forward shape `(2,2,3,384,512)`, all finite.
- codec round-trip: `build_archive`→`parse_archive`→strict `load_state_dict`→forward
  max|Δ|=0.24 (within int8 quant tolerance < 2.0). Archive 83,892 bytes (< basin's 89,136 — fewer
  params = fewer bytes, strictly better for rate).
- `partition_params_for_muon` covers ALL params (14 muon + 20 adamw groups).
The schedule-agnostic vendored codec round-trips the solved taper unchanged → the rate term is REAL.

## 4. Ready from-scratch A/B (the solve PROPOSES; exact_eval DISPOSES)
Re-tapering changes channel SHAPES → the basin CANNOT warm-start (the basin decoder has vendored
shapes) → the A/B is **FROM-SCRATCH**. Two arms, same seed / budget / data / device; the verdict is
cross-arm best-d_seg at matched bytes.

### Primary A/B (available today — `launch_taper_ab.py`, plain CE, cleanest ALLOCATION isolation)
`launch_taper_ab.py` trains both arms from-0 with the IDENTICAL plain vendored CE curriculum, so the
A/B isolates ALLOCATION from total capacity (the cleanest architecture A/B; same-seed-same-distribution
init — a bit-identical init is impossible across architectures). Run ONE arm per invocation, DETACHED
(SIGURG kills tool-bg bash at ~3 min). Recommended budget **800 epochs @ 96 pairs** (the launcher
default; rides the power-law far enough to separate the arms; ~16h/arm on the MPS gradient path; a
fast research read — gate any exact claim on a 600-pair from-0 re-run):

```bash
# arm A — baseline (vendored taper). MAY reuse the existing from0_ab_v2_n96/control run via
# --baseline-dir instead of re-training (d_seg 0.00359 at 800ep@96 CE).
nohup bash -c '.venv/bin/python experiments/launch_taper_ab.py \
    --arm baseline --out-dir experiments/results/taper_wf_ab_20260616/baseline \
    --train-device mps --split-by-head --total-epoch-budget 800 --n-pairs 96 \
    --pose-grad-every-k 4 --pose-grad-resume-threshold 0.001 --seed 0 \
    --targets-cache experiments/results/capstone_gt_targets_cache --go' \
    </dev/null >experiments/results/taper_wf_ab_20260616/baseline.outer.log 2>&1 & disown

# arm B — SOLVED waterfill taper (the custom arm).
nohup bash -c '.venv/bin/python experiments/launch_taper_ab.py \
    --arm custom --taper-channels "22,16,15,14,15,14,10" \
    --out-dir experiments/results/taper_wf_ab_20260616/solved \
    --train-device mps --split-by-head --total-epoch-budget 800 --n-pairs 96 \
    --pose-grad-every-k 4 --pose-grad-resume-threshold 0.001 --seed 0 \
    --targets-cache experiments/results/capstone_gt_targets_cache --go' \
    </dev/null >experiments/results/taper_wf_ab_20260616/solved.outer.log 2>&1 & disown
```
Verdict: arm-B best d_seg < arm-A best d_seg at matched (−0.55%) bytes → the measured waterfill
realloc wins → carry the solved taper into the long Track-A train (+ oomph + FiLM). All in-loop
numbers `[contest-CPU advisory]` until a byte-closed archive runs `upstream/evaluate.py`.

### oomph + FiLM-v2 from-scratch variant — a NAMED TOOLING GAP (do NOT invent flags)
The task's "BOTH oomph + FiLM-v2, from-scratch" arm is NOT runnable with the current launchers:
`launch_taper_ab.py` is plain-CE + `pose_film_enabled=False`; `launch_oomph_finetune_disambiguator.py`
carries the oomph (`_OOMPH`: soft_cosine T0.3→0.05 + τ=0.5 + renorm + seg_weight 1.5×, though the
optimal-config memo DROPS the 1.5× crank for sw=1.0) + `--pose-film-v2`, but it REQUIRES
`--warm-start-dir` (a converged checkpoint of the SAME shape) — incompatible with a re-tapered
(shape-changed) decoder. The clean fix is a small launcher extension: add `--taper-channels` to a
FROM-0 oomph+FiLM driver path (compose the oomph curriculum overlay + `pose_film_enabled=True`/v2 +
`ConfigurableTaperHNeRVDecoder(channels=...)` in one `TorchVehicleConfig`). That is a ~30–60 LOC add,
not a flag invention; it should land BEFORE the oomph+FiLM taper A/B. Until then, the plain-CE A/B
above is the cleanest available isolation of the ALLOCATION lever (the oomph/FiLM levers are
orthogonal and tested separately on the vendored taper).

## Wire-in (6-hook, per Catalog #125)
1. sensitivity-map: ACTIVE — `reports/dseg_sensitivity_map_n600.json` is the per-tensor d_seg
   sensitivity on the converged basin (the canonical taper input).
2. Pareto: ACTIVE — byte-neutral realloc is a constant-rate move on the d_seg axis (preserves the
   0.114 floor).
3. bit-allocator: ACTIVE (PRIMARY) — the waterfill IS a capacity (param-byte) allocator across stages.
4. cathedral autopilot: N/A — advisory $0 design tool, not archive-deployable until the A/B picks a taper.
5. continual-learning: N/A — advisory non-promotable (no exact-eval row; the A/B produces the anchor).
6. probe-disambiguator: ACTIVE — `solve_taper_waterfill.py` + the from-scratch A/B IS the
   heuristic-vs-solved arbitration.

## Artifacts
- `experiments/probe_dseg_sensitivity_map_basin_n600.py` (measured map probe)
- `experiments/solve_taper_waterfill.py` (the waterfill solver)
- `reports/dseg_sensitivity_map_n600.json` (the REAL measured map)
- `reports/taper_waterfill_solve_n600.json` (the solved taper + diagnostics)
