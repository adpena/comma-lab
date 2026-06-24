# Concentrated-saliency (high-res-weighted) taper screen — d_seg lever NO-GO

**Date:** 2026-06-23/24 · **Subagent:** concentrated-taper-screen-20260623
**Authority:** `[contest-CPU advisory]` / `[macOS-CPU advisory]` — **NON-PROMOTABLE**, pointer-only.
The exact d_seg/d_pose/rate numbers below are from the in-vehicle CPU authority eval
(`--device cpu`, byte-closed) — NOT MPS (MPS = training gradient only). A score is authoritative
only after `upstream/evaluate.py` on the byte-closed archive; these advisory numbers rank arms, they
do not move the frontier pointer (UNMOVED 0.19110).

---

## The question (decisive, cheap)

Does a HIGH-RES-WEIGHTED (concentrated-saliency) per-block channel taper, **at the same byte budget**
as the generic bc20 decreasing taper, **lower d_seg** by reallocating representational bandwidth to the
high-resolution boundary stages where the d_seg argmax-flips live? This is the LAST remaining RGB-rung
d_seg lever before conceding the capacity cliff.

**Hypothesis (E-axis / bandwidth):** d_seg is a high-frequency codimension-1 boundary at high resolution;
the generic decreasing taper `[20,20,20,15,11,10,10]` allocates the FEWEST channels at the high-res
stages (channels[5..6]=10,10 at 192×256 / 384×512). Shifting capacity there (CONC-B
`[12,12,14,16,18,20,20]`, CONC-C `[10,10,12,16,18,22,22]`) should lower d_seg at ~constant params/bytes.

## Param-matching (byte budget held)

`ConfigurableTaperHNeRVDecoder(latent_dim=28)`, conv params are resolution-independent so all tapers are
~byte-matched:

| arm | taper (ch[0]=6×8 stem … ch[6]=384×512) | params |
|---|---|---|
| GENERIC (control) | `[20,20,20,15,11,10,10]` (vendored default) | 83,356 |
| CONC-B | `[12,12,14,16,18,20,20]` | 79,952 |
| CONC-C | `[10,10,12,16,18,22,22]` | 77,847 |
| DISK-conc (existing n600 run) | `[16,16,17,19,19,14,10]` | 83,422 |

## Method

`experiments/launch_split_by_head_basin.py --no-split-by-head --train-device mps --device cpu
--base-channels 20 --latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 --muon-lr-floor-fix
--eval-every 50 [--taper-channels …]`. Every flag identical across arms — **ONLY `--taper-channels`
differs** → a clean isolate. n=100 is the memorization-regime proxy (no n=96 cache exists; n=100 is the
smallest faithful cache that reaches a basin; targets are byte-identical to a fresh compute). Arms launched
as truly-detached durable daemons (`tools/spawn_durable_daemon.py`, setsid, group-kill on stop — no
orphans; the design survived 6+ SIGURG-144 monitor-shell teardowns during the run with zero impact on the
training daemons). Authority d_seg via the in-vehicle CPU byte-close eval at each ge%50==0.

NOTE on budget mechanics (measured, important): `--total-epoch-budget` proportionally scales ALL 8 PR95
stages; budget=80 (≈78 ep) produces near-random d_seg≈0.50 (never leaves the early stages). The d_seg
basin lives **entirely in stage-0 CE** (the converged n600 disk run stayed in stage 0 through ge=2398).
budget=3000 gives stage-0 ≈ 304 CE epochs — the right vehicle. The arms were run through the stage-0
descent (ge=150→300); 3-way MPS contention held s/ep ≈ 4-5.5 (sync evals block ~2 min each).

---

## RESULT 1 — Controlled n=100 screen (taper-ONLY isolate) — the decisive table

| ge  | GENERIC d_seg | CONC-B d_seg | d_seg Δ (CONC vs GEN) | GEN bytes | CONC-B bytes | byte Δ |
|----:|:---:|:---:|:---:|:---:|:---:|:---:|
| 150 | 0.00918 | 0.01083 | **+18.0%** | 80580 | 78086 | −3.1% |
| 200 | 0.00675 | 0.00804 | **+19.1%** | 79798 | 77101 | −3.4% |
| 250 | 0.00548 | 0.00648 | **+18.2%** | 79534 | 76456 | −3.9% |
| 300 | 0.00476 | 0.00561 | **+17.9%** | 79293 | 76053 | −4.1% |

CONC-C (3-arm phase, stopped early to free MPS) tracked between GEN and CONC-B; at ge=100 it was
0.04634 vs GEN 0.05287 — the early ge=100 ordering (CONC-B 0.0264 vs GEN 0.0529, an apparent −50%)
was **transient noise** that fully reversed by ge=150 and stayed reversed.

**Verdict (controlled isolate): the concentrated/high-res-weighted taper does NOT lower d_seg — it
RAISES it by a stable ~18% at matched budget.** Its only benefit is a small ~3-4% byte reduction
(rate co-benefit). The bandwidth hypothesis is **REFUTED** in the controlled isolate: more capacity at
the high-res stages did not buy lower d_seg; if anything the generic taper (more capacity at the
mid-res 20×24→48×64 stages) converges d_seg faster.

## RESULT 2 — Disk n=600 fully-converged anchors (existence-proof cross-check; CONFOUNDED)

Pre-existing fully-trained runs on disk (decoder taper recovered authoritatively from block-weight
shapes), S recomputed via `tac.contest_score.compute_contest_score`:

| config | taper | hinge | best d_seg | bytes | S |
|---|---|:---:|---|---|---|
| GENERIC, no hinge (ge2325) | generic | no | 0.00256 | 89073 | 0.3705 |
| GENERIC, +hinge (ge2950) | generic | yes | 0.00224 | 89237 | 0.3288 |
| CONC `[16,17,19,19,14,10]`, +hinge (ge11000) | concentrated | yes | 0.00206 | 82607 | 0.3087 |
| CONC, +hinge +muonjump (ge24725) | concentrated | yes | 0.00208 | 79592 | 0.3067 |

**Decomposition (n600 converged):** margin-hinge alone = −12.5% d_seg (0.00256→0.00224); taper alone
(hinge held) = −8.0% d_seg (0.00224→0.00206) + −7.4% bytes. So the disk "concentrated wins by −0.064 S"
is driven by **margin-hinge + much longer training**, NOT the taper. The taper's converged d_seg edge
(−8%) is the OPPOSITE SIGN of the controlled-isolate result (+18%) — and the converged comparison is
confounded (the concentrated run trained 4-10× longer: ge 11000-24725 vs 2950).

## Existence-proof / 5-lens joint review

- **Existence proof:** the controlled isolate (the only clean A/B) is the trustworthy signal and it is a
  clean NO. The disk "win" is NOT an existence proof for the taper because the variable was not isolated
  (margin-hinge + train-length co-varied). Cross-check fails → reject the bandwidth claim.
- **Geometry/physics lens:** the bandwidth argument is intuitive but the measured gradient disagrees —
  d_seg descent is governed by the whole-decoder optimization, and starving the mid-res stages
  (CONC-B's ch[0..2]=12,12,14 vs generic 20,20,20) slows d_seg more than the high-res boost helps. The
  high-res flip-boundary is real, but added high-res CHANNELS are not the binding constraint on it at
  this capacity — consistent with the standing finding that d_seg at ~80KB is **capacity-bound /
  power-law-slow**, not bandwidth-allocation-bound.
- **NO-FAKE lens:** the ge=100 −50% headline would have been a fake win; held to 4 matched checkpoints it
  reversed. The disk −8% would have been a fake taper attribution; decomposed against the
  generic+hinge anchor it shrinks and the controlled isolate flips its sign.
- **Scorer/score lens:** even the best concentrated disk d_seg=0.00206 → seg_term=0.206, which is
  **224× above** the seg_term=0.092 that the 9.2e-4 sub-0.19 d_seg line requires. No taper reshuffle at
  ~80KB approaches the line. d_seg term ALONE (0.206) exceeds the whole 0.191 frontier.
- **Full-space lens:** the only taper benefit (−3-4% bytes) is a rate-axis crumb; the frontier is
  rate-dominated only at the borrowed-codec operating point, and our basin is d_seg-dominated, so a
  3-4% byte cut on an 80KB archive (rate_term ~0.053→~0.051) is noise against the 0.206 seg_term.

## S-projection / L13-witness deployment

Best concentrated disk arm (d_seg=0.00206, d_pose=0.000227, 82607 B) → **S=0.3087** as a standalone
witness; the param-matched controlled CONC-B would land WORSE on d_seg (+18%) for the same bytes. No
configuration in this family is within ~2× of the 0.191 frontier, let alone sub-0.19/sub-0.15.

---

## GO / NO-GO

**NO-GO** for a full multi-day concentrated-saliency n=600 run. The controlled, matched-budget,
taper-only isolate shows the high-res-weighted taper does **not** lower d_seg (it raises it ~18%); the
apparent disk win is a margin-hinge + train-length confound, not the taper. The single recommended next
step is below; the taper itself is at best a ~3-4% byte (rate) micro-lever to fold into an existing run
for free, **never** a d_seg lever and never worth a dedicated burn.

**Capacity-cliff honesty:** this closes the last RGB-rung d_seg lever screened cheaply. d_seg at the
~80KB bc20 budget is capacity-bound (power-law slow), not taper-allocatable. Sub-0.19 on the d_seg axis
needs MORE capacity (bc36-class ≈ 230KB = the borrowed-ceiling rate cost) OR a different (concentrated-
SALIENCY, not concentrated-CHANNEL) representation that puts BYTES where the SegNet boundary flips live —
i.e. a structured/sparse boundary code, not a uniform channel reshuffle. The "concentrated saliency"
intuition is correct; "concentrated channels" is the wrong actuator for it.

## Reactivation criteria

Re-open the concentrated-taper-as-d_seg-lever ONLY if a clean, matched-budget, matched-flags
(margin-hinge held identical) n=600 controlled A/B shows concentrated d_seg < generic d_seg at the basin
(ge≥1000) — the disk confound must be removed before any converged claim. Until then the +18% controlled
result stands. Separately, the orthogonal "concentrated SALIENCY" path (spend bytes, not channels, at the
SegNet flip-boundary via a sparse boundary sidecar) is a distinct, un-screened hypothesis worth its own
$0 probe.

## Artifacts (durable)

- `experiments/results/taper_screen_GENERIC_n100_b3000/` (control, generic taper)
- `experiments/results/taper_screen_CONCB_n100_b3000/` (concentrated `[12,12,14,16,18,20,20]`)
- `experiments/results/taper_screen_CONCC_n100_b3000/` (concentrated `[10,10,12,16,18,22,22]`, partial)
- Disk anchors: `torch_vehicle_full_mps_basin_bc20_n600/`,
  `yousfi_r2_marginhinge_fullmps_20260620/`, `yousfi_r3_taper_marginhinge_e5_20260620/`,
  `yousfi_r3_MUONJUMP_stage8_lr1e3_20260623T180100Z/` (tapers recovered from decoder block shapes).

## 6-hook wire-in

#1 sensitivity-map: ACTIVE (the per-stage taper IS a channel-sensitivity allocation; result = high-res
channels are NOT the binding d_seg constraint at ~80KB). #2 Pareto: ACTIVE (taper trades ~3-4% rate for
~+18% d_seg = dominated on the d_seg axis). #3 bit-allocator: N/A (channel taper, not symbol bits).
#4 cathedral autopilot: N/A (advisory non-promotable). #5 continual-learning: this memo + checkpoint.
#6 probe-disambiguator: ACTIVE (the controlled isolate IS the disambiguator that resolved the
disk-confound vs taper-effect ambiguity).
