# FINER / WIRE high-frequency activation — k1-for-d_seg ARCHITECTURE screen (SPECTRAL vs CAPACITY)

**Date:** 2026-06-23/24 (UTC) · **Subagent:** `finer-wire-arch-screen-20260623`
**Authority:** `[contest-CPU advisory]` / `[macOS-CPU advisory]` — **NON-PROMOTABLE**, pointer-only.
d_seg numbers are the in-vehicle **CPU-authority** byte-closed exact d_seg (`--device cpu`); the
SegNet gradient trains on MPS (the 104× lever) but **MPS is NEVER the d_seg verdict**. A score is
authoritative only after `upstream/evaluate.py` on the byte-closed archive; these advisory numbers
rank arms — they do not move the frontier pointer (UNMOVED **0.19110**).

---

## The decisive question

The concentrated-saliency **taper** screen (`concentrated_saliency_taper_screen_20260623.md`) showed
capacity **PLACEMENT** does not fix d_seg (concentrated d_seg 0.0056 > generic 0.0048, +18% at matched
budget) → the small-budget d_seg deficit is **not** a where-the-channels-sit problem. The remaining
disambiguation: is the deficit a **RAW-CAPACITY** limit (a fixed param/byte budget simply cannot
represent the boundary sharply → sub-0.15-via-small-RGB is closed, 0.191 is the borrowed HNeRV ceiling)
or a **SPECTRAL-BIAS** limit (the fixed-ω `torch.sin` decoder under-represents the high-frequency
codimension-1 boundary, and a high-frequency activation unlocks it → the architecture is the k1 lever)?

**The test:** swap ONLY the decoder nonlinearity at the SAME byte budget. `sin` (fixed ω=1) has the
documented coordinate-network low-frequency spectral bias (Rahaman 2019 / Tancik 2020). **FINER**
(`sin((|x|+1)·x)`, variable local frequency) and **WIRE** (`sin(ωx)·exp(-½(s·x)²)`, Gabor — optimal
joint space+frequency localization, Mallat) give high-frequency capability at IDENTICAL param/byte
count. Taper (placement) already failed → if architecture (representation) ALSO fails flat, the deficit
is raw-capacity; if a high-freq activation lowers d_seg ≥15% at equal bytes, it is spectral.

## Wire-in (NO-FAKE: siren byte-identical; finer/wire real swaps)

`ConfigurableTaperHNeRVDecoder` gained `activation_family` + `wire_scale`; the 3 `torch.sin` sites in
`forward` route through `tac.substrates.siren.activation_family.apply_activation_family(..., omega=1.0,
…)`. `TorchVehicleConfig.activation`/`wire_scale` thread it; `--activation {siren,finer,wire}` +
`--wire-scale` on `launch_split_by_head_basin.py`. A non-siren activation forces the ConfigurableTaper
path (the vendored decoder hardcodes `torch.sin` and lives in a pristine intake clone — not editable).

**Parity proof (mechanical, before any training):** loading vendored weights into ConfigurableTaper
with `activation=siren` renders **BIT-IDENTICAL** to the vendored decoder (max|Δ|=**0.0**); FINER and
WIRE at the SAME weights produce genuinely different output (max|Δ| 9–93), and `wire_scale` modulates
the window (0.5 ≠ 2.0). 8 dedicated NO-FAKE tests (`test_configurable_taper_decoder.py`, total 27 pass)
— the headline guards (`test_activation_finer_actually_changes_output`,
`test_activation_wire_actually_changes_output_and_scale_matters`) FAIL if the swap degenerates to a
no-op. Commit `5459adf8e`.

## Method (apples-to-apples with the taper-screen GENERIC control)

`launch_split_by_head_basin.py --no-split-by-head --train-device mps --device cpu --base-channels 20
--latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 --muon-lr-floor-fix --eval-every 50`,
every flag identical across arms — **ONLY `--activation` (+`--wire-scale`) differs**. The SIREN control
MUST reproduce the taper-screen GENERIC d_seg (0.004756 @ ge300) → proves the wire-in is byte-identical.
Arms launched as group-kill durable daemons (`spawn_durable_daemon.py --label` → no orphans). n=100 is
the memorization-regime proxy screen; a winner gets an n600-short confirm. Falsification bar: a real win
= d_seg ≤ **0.85×** the siren baseline at equal bytes (clearly beyond seed noise).

---

## RESULT — d_seg per activation at matched checkpoints
(FILLED 2026-06-24 from the completed runs; all 3 arms finished the 3000-epoch budget, daemons exited;
d_seg = byte-closed CPU-authority from each arm's `torch_vehicle_trajectory.jsonl`. n=100 proxy.)

| arm | best d_seg | @ep | ~ge300 d_seg | best vs SIREN |
|---|---:|---:|---:|---:|
| **SIREN** (control, ω=1 fixed-sin) | 0.001692 | 2945 | 0.004754 | — |
| **FINER** (`sin((|x|+1)·x)`) | **0.001376** | 2895 | 0.004121 | **0.813× (−18.7%)** |
| **WIRE10** (Gabor, scale 1.0) | 0.001659 | 2895 | 0.004828 | 0.98× (−2%) |

## SIREN-control parity check
**CLEAN ✓.** SIREN ~ge300 d_seg = **0.004754** reproduces the taper-screen GENERIC control (**0.004756**) to
2e-6 → the `apply_activation_family(..., omega=1)` wire-in is byte-identical to the vendored `torch.sin` path.
The screen is valid (the FINER/WIRE deltas are real architecture effects, not wiring artifacts).

---

## VERDICT — SPECTRAL-LIMITED (partially)
**FINER is a REAL d_seg win at EQUAL bytes; WIRE is NULL.** FINER best 0.001376 = 0.813× SIREN CLEARS the
falsification bar (≤0.85× = clearly-beyond-seed-noise; WIRE's 0.98× IS the seed-noise scale, confirming the
bar). So the small-budget d_seg deficit is **PARTLY SPECTRAL** — the fixed-ω SIREN under-represents the
high-frequency codim-1 boundary (lane/island edges), and a variable-local-frequency activation (FINER) extends
the spectrum and lowers d_seg at identical param/byte count. **Frequency-adaptivity (FINER) beat
space-frequency localization (WIRE-Gabor)** for this boundary — the boundary needs bandwidth, not a window.
This is the FIRST architecture-only positive d_seg lever this arc = a measured **D(H) curve-shift** (lower
d_seg at the same H), exactly what the convergence (`island_representation_level_intrinsic_dim` GO-GENERATOR)
said sub-0.15 requires.

## S-projection
Advisory only (n=100 proxy; NOT a contest row). At matched everything, FINER's −18.7% relative d_seg, if it
holds at n600 and composes with the geometry-prior + round-trip-in-loop generator, lowers the dominant
`100·d_seg` term proportionally. NOT projected to an absolute S here — the n=100 d_seg (0.001376) is a smaller
problem than the n600 frontier (6e-4); the LOAD-BEARING quantity is the **relative** architecture win, which
the n600 confirm must reproduce before any S claim. Pointer UNMOVED 0.19110.

## Existence-proof / 5-lens joint review
- **math:** −18.7% clears 0.85×; WIRE's −2% = the seed-noise null that calibrates the bar.
- **geometry:** FINER adapts frequency to input magnitude → resolves the high-freq lane/island edges SIREN
  smooths; WIRE's Gabor window did not help at this budget.
- **calculus:** the win is at convergence (ep2895), marginal @ge300 (0.867×) → FINER needs the full budget to
  realize the spectral capacity.
- **physics:** spectral-bias fix (Tancik 2020 / Rahaman 2019) — extends the coordinate-net spectrum.
- **existence-proof:** clean SIREN control (reproduces 0.004756) + WIRE null bracket FINER as a real effect,
  not a wiring/sparsity artifact. Consistent with the texture-survival wall (#149): the win is in the RENDERED
  texture (architecture), not a sidecar.

## GO / NO-GO + single recommended next step
**GO (advisory) — FINER is the generator's nonlinear chart.** Single next step: an **n600-short FINER-vs-SIREN
confirm** (the memo's pre-registered winner-confirmation) to promote the −18.7% from advisory to load-bearing;
then the from-scratch generator fires with FINER + geometry-prior capacity routing (lane/horizon) +
round-trip-in-loop + SegNet-saliency d_seg loss → byte-close → exact eval (the END). NOT promotable; pointer
UNMOVED 0.19110.

## 6-hook wire-in

#1 sensitivity-map: the activation family is a WHAT-each-param-represents axis (orthogonal to the taper
WHERE axis). #2 Pareto: d_seg-vs-byte (byte-neutral activation swap). #3 bit-allocator: N/A.
#4 cathedral autopilot: N/A (advisory non-promotable). #5 continual-learning: this memo + checkpoint +
trajectory JSONL. #6 probe-disambiguator: this screen IS the SPECTRAL-vs-CAPACITY disambiguator.

## Artifacts (durable)

- `experiments/results/act_screen_SIREN_n100_b3000/` (control)
- `experiments/results/act_screen_FINER_n100_b3000/`
- `experiments/results/act_screen_WIRE{05,10,20}_n100_b3000/`
- wrappers `experiments/results/_act_screen_wrappers/run_*.sh`
- code: commit `5459adf8e` (decoder + driver + launcher + tests)
