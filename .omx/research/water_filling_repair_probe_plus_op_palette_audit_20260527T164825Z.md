# Water-Filling REPAIR Probe + 5D-Canvas Op-Palette Audit — 2026-05-27T16:48:25Z

**Evidence grade:** `[macOS-MLX research-signal]` — NON-PROMOTABLE per Catalog #192/#127/#323.
**Score claim:** `false`. **Promotion eligible:** `false`. **Ready for exact-eval dispatch:** `false`.
**Subagent:** `water_filling_repair_probe_20260527`. **Lane:** `lane_water_filling_repair_probe_plus_op_palette_audit_20260527`.

This memo answers the three operator-routed deliverables: (1) OP-COVERAGE, (2) PALETTE+GAPS,
(3) THE KEY — water-filling REPAIR run with numbers. All $0; no paid GPU. The water-fill
allocator + 12 canvas operators + master-gradient sensitivity map were READ, not reimplemented.

---

## TL;DR (the water-filling-repair verdict)

**INDETERMINATE — `INDETERMINATE_UPPER_BOUND_PROXY_NOT_REALIZABLE`.**

The per-byte scorer-sensitivity ranking is **NON-DEGENERATE** (Gini 0.660, top-1 byte = 0.47%
of total L1 mass, 113,578 bytes carry 90% of mass, 178,457/178,517 nonzero) — **the rank-1
tautology is NOT present**, so the ranking is a real spread, not a single concentration.

BUT the marginal-distortion-per-repair-byte computed from the master-gradient is an
**UPPER-BOUND PROXY**, not a realizable score delta. The naive greedy water-fill estimate
EXCEEDS the *entire distortion budget* present at the operating point (0.2532) at every budget
level — even the smallest budget of 64 bytes claims to recover 0.2599 distortion, which is
physically impossible. This proves the FD-magnitude marginal measures
`|∂score / ∂(byte_VALUE)|` (sensitivity to perturbing an *existing* byte), **NOT**
`|∂score / ∂(ADDING a repair byte)|` (what REPAIR actually does).

**The value-sensitivity → repair-recoverability gap is UNMEASURED.** Declaring a FIRE candidate
from this proxy alone would repeat the unbounded-upper-bound error class — the sister of the
rank-1 problem-spec tautology that commit `21c76632a` extinguished. Per CLAUDE.md
"Apples-to-apples evidence discipline" + "Forbidden premature KILL without research exhaustion",
the disciplined verdict is **DEFER-pending-realizability-micro-probe**, NOT a phantom FIRE and
NOT a null.

This is fully consistent with the canonical paradox-Half-2 verdict
(`paradox_half2_5d_canvas_repoint_empirical_verdict_20260527T142332Z.md`), which independently
established that the `−0.0154 replace_many` magnitude is "**the SUM of per-pair finite-difference
gradient MAGNITUDES (a LEVERAGE indicator), NOT a realizable contest-CPU score delta**." The
water-fill probe reaches the same wall from the repair-budget direction.

---

## DELIVERABLE 1 — OP-COVERAGE: which of the 12 operators were EMPIRICALLY exercised

**Canonical source of the empirical record:** the rank-fix synergy re-run `21c76632a` +
its verdict memo `paradox_half2_5d_canvas_repoint_empirical_verdict_20260527T142332Z.md`.

**Correction to the prompt's "5":** the canonical verdict memo records **6/12 productive
operators**, not 5. The prompt's recollection of "5" undercounts by one (it omits `full_drop`,
which composes via the `RAW_RESIDUAL` receiver-runtime path). The empirically-exercised set is:

| # | operator | photoshop analog | EMPIRICALLY exercised in `21c76632a` re-run? | heuristic-prior leverage (NOT score) |
|---|---|---|---|---|
| 1 | `replace_many` | content-aware fill (multi-region) | **YES** (rank 1) | −0.0154 |
| 2 | `temporal_coherence` | clone-stamp across time / onion-skin | **YES** (rank 2) | −0.0108 |
| 3 | `reorder_pair` | layer-reorder / frame shuffle | **YES** (rank 3) | −0.0090 |
| 4 | `merge_pair` | merge-down / flatten two layers | **YES** | (composed) |
| 5 | `motion_conditional` | motion-aware healing (FOE-prior) | **YES** | (composed) |
| 6 | `full_drop` | delete layer / erase | **YES** (via RAW_RESIDUAL receiver) | (composed) |
| 7 | `replace_one` | spot-heal (single region) | **NO — scaffolded only** | — |
| 8 | `drop_frame` | delete keyframe | **NO — scaffolded only** | — |
| 9 | `synthesize_frame` | generative fill / interpolate-frame | **NO — scaffolded only** | — |
| 10 | `repair` | healing brush (drop + inject signal) | **NO — scaffolded only** | — |
| 11 | `masked` | layer-mask / per-region select | **NO — scaffolded only** | — |
| 12 | `feathered` | feathered-mask / soft-edge select | **NO — scaffolded only** | — |

**6 EXERCISED:** `replace_many`, `temporal_coherence`, `reorder_pair`, `merge_pair`,
`motion_conditional`, `full_drop`.
**6 UNTESTED (scaffold-only):** `replace_one`, `drop_frame`, `synthesize_frame`, `repair`,
`masked`, `feathered`.

**Note on `repair` specifically:** the operator this water-fill probe targets
(`CanonicalOperation.REPAIR` = "drop + add per-pair/per-frame repair signal", Atick-Redlich
cooperative-receiver) is in the UNTESTED-scaffold-only set. The canvas has the `generate_repair_starts`
scaffold (`pair_frame_scorer_geometry_lattice_5d_canvas.py:1361`, refuses `RAW_RESIDUAL`,
defaults `SMOOTHED_RESIDUAL`) but it has NEVER been run against the master-gradient to measure a
realized distortion-reduction-per-repair-byte. That gap is precisely what makes the
water-fill-repair verdict INDETERMINATE rather than NULL or FIRE.

---

## DELIVERABLE 2 — PALETTE + GAPS

### Each of the 12 canvas operators → photoshop analog

(see table above). The canvas's design intent maps cleanly onto a photo-editing palette:
- **Removal class** (`full_drop`, `drop_frame`) = delete/erase.
- **Replacement class** (`replace_one`, `replace_many`, `merge_pair`) = spot-heal / content-aware
  fill / merge-down.
- **Injection class** (`repair`, `synthesize_frame`) = healing-brush / generative-fill.
- **Selection-mask class** (`masked`, `feathered`) = layer-mask / feathered-mask.
- **Temporal class** (`reorder_pair`, `temporal_coherence`, `motion_conditional`) = frame-shuffle /
  onion-skin / motion-aware healing.

### Photoshop-like tools NOT yet built, ranked by predicted contest-EV

These rankings are **heuristic-prior** (NOT score claims). EV is judged by (a) whether the tool
INJECTS signal where the scorer is sensitive (the only direction that can beat the 25/N rate
cost at this frontier — per the paradox-Half-2 finding that rate-attack single-op is
Pareto-optimal and DISTORTION-removal does not reopen), and (b) numpy-portable inflate feasibility.

| rank | photoshop tool | proposed canvas op | mechanism | predicted contest-EV | rationale |
|---|---|---|---|---|---|
| 1 | **dodge/burn = per-region gain** | `per_region_gain` | scale frame-region luma/chroma toward scorer-optimal exposure | **LOW-but-nonzero** | seg/pose are exposure-robust (FastViT/EfficientNet normalize); marginal at best. Beats smudge because per-region gain is the cheapest injection (a few scale bytes per region). |
| 2 | **selective-sharpen** | `selective_sharpen` | high-pass boost on high-pose-sensitivity regions (FOE / road-edge) | **LOW** | pose is edge/motion-driven; sharpening road-edge regions COULD recover pose distortion, but the FD-sensitivity → recoverability gap (this probe's INDETERMINATE finding) applies identically. |
| 3 | **frequency-separation** | `freq_separation` | split low-freq (chroma-LUT-codeable) + high-freq (residual) bands | **SPECULATIVE** | aligns with Daubechies wavelet partition prior + the `feathered` op; could let the codec spend bytes only on high-freq detail the scorer cares about. Highest *upside* but unmeasured + most LOC. |
| 4 | **content-aware-scale** | `content_aware_scale` | seam-carve-style resize preserving high-sensitivity regions | **NULL-likely** | contest output is fixed 1164×874×1200×3; no resize freedom. Out of scope at the contest output contract. |
| 5 | **smudge** | `smudge` | local averaging / blur toward neighbor pixels | **NULL** | smudge REMOVES detail = increases distortion; only useful for rate-saving, which `full_drop`/`masked` already cover better. Dominated. |

**Gap verdict:** the highest-EV missing tool (`per_region_gain`) is still gated by the SAME
realizability gap as `repair` — none of these INJECTION tools can be ranked above DEFER until a
byte-mutation realizability micro-probe measures realized-distortion-reduction-per-injected-byte
on the actual archive. Building more scaffolded injection operators without closing the
realizability gap would be scaffold-accumulation (the Catalog #298 anti-pattern), not progress.

---

## DELIVERABLE 3 — THE KEY: water-filling REPAIR run (with numbers)

**Tool:** `tools/probe_water_filling_repair.py` (NEW; owned by this lane).
**Inputs (READ-ONLY):** master-gradient `(178517, 600, 3)` =
`master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy`; canonical frontier pointer.
**Allocator:** ranks per-byte by aggregate seg+pose sensitivity (L1 over 600 pairs; rate axis
excluded because rate is the COST the repair must beat, not a distortion it can reduce), then
greedy water-fill of the top-budget bytes (the canonical
`tac.optimization.bit_allocator_end_to_end.allocate_per_pair_bits` cascade ranks the same way).

### Degeneracy assertion (the rank-1 tautology guard) — PASSED

| metric | value | interpretation |
|---|---|---|
| Gini coefficient | **0.660** | real spread (0=uniform, 1=single mass) |
| top-1 byte mass fraction | **0.47%** | NOT a point mass |
| bytes to reach 90% of mass | **113,578** | sensitivity is broadly distributed |
| nonzero bytes | **178,457 / 178,517** | 28 decoder mantissa-byte spans dominate; 99.97% nonzero |
| degenerate? | **FALSE** | rank-1 tautology is structurally absent |

**The ranking is genuinely non-degenerate.** This is the load-bearing assertion the operator
asked for: unlike the rank-1 problem-spec tautology (commit `21c76632a`), the per-byte
sensitivity here has real Lorenz-curve spread. Water-fill is NOT trivially "optimal" by
construction.

### Budget sweep (rate cost per byte = 25/N = 6.659e-7; operating-point distortion budget = 0.2532)

| budget (bytes) | naive Σ-marginal | exceeds 0.2532 budget? | realizable (capped) | realizable marginal/byte | rate/byte | naive verdict |
|---:|---:|:--:|---:|---:|---:|:--:|
| 64 | 0.2599 | **YES** | 0.2532 | 3.96e-3 | 6.66e-7 | beats (proxy) |
| 256 | 0.5365 | **YES** | 0.2532 | 9.89e-4 | 6.66e-7 | beats (proxy) |
| 1024 | 0.9449 | **YES** | 0.2532 | 2.47e-4 | 6.66e-7 | beats (proxy) |
| 4096 | 1.5295 | **YES** | 0.2532 | 6.18e-5 | 6.66e-7 | beats (proxy) |
| 16384 | 2.0577 | **YES** | 0.2532 | 1.55e-5 | 6.66e-7 | beats (proxy) |

### The verdict math (apples-to-apples)

The naive water-fill says the marginal beats the rate cost by ~6098× (top-byte marginal
4.06e-3 vs 6.66e-7). **This is the trap.** Three observations make it INDETERMINATE, not FIRE:

1. **Physical impossibility:** the naive sum-of-marginals (0.26 at b=64, 2.06 at b=16384)
   exceeds the entire distortion budget (0.2532) present at the operating point. You cannot
   recover more distortion than exists. An estimate 8× larger than the total distortion budget
   is not a realizable score delta — it is an unbounded upper bound.
2. **Wrong derivative:** the master-gradient FD measures `∂score/∂(byte_value)` — how the score
   responds to PERTURBING an existing decoder byte. REPAIR's quantity is
   `∂score/∂(adding a repair byte)` — how the score responds to ADDING new bytes that the
   receiver runtime must then consume. These are different operators; the former is a sensitivity
   map, the latter requires a measured receiver response.
3. **Rate axis is zero:** `rate_axis_signal_in_gradient = 0.0` — the master-gradient has NO rate
   dimension. The 25/N cost is fully external; the gradient cannot tell us the rate↔distortion
   tradeoff a repair byte actually realizes.

**Does marginal distortion-reduction per repair-byte beat the 25/N rate cost?**
**The honest answer: UNKNOWN at the proxy level, and the proxy is provably an over-estimate.**
The realizable answer requires measuring the actual receiver-consumed distortion reduction.

---

## Operator-gated next step (the disciplined path forward)

**$0 byte-mutation realizability micro-probe (Catalog #139 / #220 / #272 lineage):**
mutate the top-K (e.g. K=64) highest-sensitivity decoder bytes toward a repair-signal target on
the ACTUAL FEC6-frontier `archive.zip`, re-run the parity-validated MLX scorer oracle, and
measure the REALIZED `d_seg`/`d_pose` reduction per repair byte. Then re-compare to 25/N.
This closes the value-sensitivity → repair-recoverability gap that this probe identifies. It is
the same discipline that the rank-fix re-run applied to synergy: replace the proxy with a
measured quantity before claiming a frontier move.

**If that micro-probe shows realized reduction > 25/N per byte:** name the FIRE candidate as a
single-frame/single-pair REPAIR on the top-sensitivity region, with a numpy-portable inflate
path = `SMOOTHED_RESIDUAL` receiver runtime (the canvas `repair` op already refuses
`RAW_RESIDUAL`; `SMOOTHED_RESIDUAL` is a numpy-portable smooth-residual add, ≤200 LOC + ≤2 deps
per HNeRV parity L4). Then paired contest-CUDA/CPU exact-eval per Catalog #246 before any score
claim.

**If that micro-probe shows realized reduction < 25/N per byte:** REPAIR joins `full_drop` in
the falsified-at-this-operating-point set, and the paradox-Half-2 verdict (single-op rate-attack
Pareto-optimal; class-shift / PACT-NeRV is the only sub-frontier lever) stands unamended.

Per the paradox-Half-2 verdict + the standing PACT-NeRV class-shift directive, the highest-EV
direction remains **original-substrate class-shift**, NOT another distortion-injection operator.

---

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map contribution** — ACTIVE: the probe consumes the per-pair master-gradient as
   the canonical sensitivity map; the degeneracy assertion + per-byte L1 ranking are a reusable
   sensitivity-spread diagnostic.
2. **Pareto constraint** — ACTIVE (advisory): the rate↔distortion tradeoff (25/N vs realizable
   marginal) is the Pareto facet; the probe records it cannot be resolved from the value-FD proxy
   alone.
3. **Bit-allocator hook** — ACTIVE: the probe ranks bytes exactly as
   `allocate_per_pair_bits` does (seg+pose L1 over pairs); the realizability cap is a new
   guard that allocator consumers should respect.
4. **Cathedral autopilot dispatch hook** — ACTIVE (DEFER signal): the verdict gates the autopilot
   ranker — do NOT fire a multi-byte REPAIR budget from the proxy; the micro-probe is the gate.
5. **Continual-learning posterior update** — ACTIVE: the Catalog #313 probe-outcome DEFER row is
   appended; its reactivation criterion is the realizability micro-probe.
6. **Probe-disambiguator** — ACTIVE: this probe IS the disambiguator between
   "proxy-marginal-beats-rate (phantom FIRE)" and "realized-marginal-beats-rate (real FIRE)";
   it resolves to INDETERMINATE-pending-micro-probe.

## Discipline anchors

Catalog #229 (premise verification — read canvas + allocator + producer memo before claiming) +
#192/#127/#323 (canonical Provenance; macOS-MLX non-promotable) + #139/#220/#272 (byte-mutation
realizability lineage for the next step) + #296 (no predicted band without realizability check —
this probe IS that check, and it refuses) + #298 (no scaffold-accumulation — declined to build
5 new injection operators) + #307 (any future micro-probe falsification is IMPLEMENTATION-level,
not paradigm-level) + #313 (probe-outcomes row LANDED) + #341 (canonical routing markers; the
probe output carries non-promotable markers) + #348 (this gate has no new catalog #; not
applicable) + #314/#340 (sister-checkpoint guard honored).
