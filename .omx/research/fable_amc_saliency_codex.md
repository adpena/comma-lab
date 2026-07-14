# FABLE AMC-saliency codex — warm-start sweep + per-row tiered bit-alloc MEASURED (2026-07-14)

**Pointer status (FIRST):** submittable `0.19108282 [contest-CPU]` UNMOVED; defensive bank
`0.18804440` (borrowed, non-submission) UNMOVED. Every row below is
`[macOS-CPU advisory; NumPy-fp32 receiver; CPU frozen scorers] NON-PROMOTABLE`,
`score_claim=false`. No paid/heavy GPU launched; upstream `evaluate.py` NOT run (operator-GO).

**Lane:** `fable_amc_saliency_tiered_bitalloc_20260714` · **Tool (NEW, self-contained):**
`tools/apply_amc_saliency_tiered_bitalloc_witness.py` · **Artifacts:**
`experiments/results/amc_saliency_tiered_bitalloc_20260714/` · **Seed:** arXiv 2607.10109 (AMC).

---

## 1. Seed deep-read + the divergence fork (PAPER_WARM_START_FROM_DIVERGENCE)

**AMC's real mechanism (fetched, HTML full text):** per-TOKEN saliency = L1 activation
magnitude `S_i = mean_j |x_ij|` (+ optional query-cosine blend); THREE percentile tiers
(top 20% / next 30% / bottom 50%); joint per-tier {rank, bit-width} menu ({128,43,8} ×
{16,8,4} via Hadamard rank masking + fixed-point quant); allocation = a plain threshold
cascade (Algorithm 1) — **no waterfill, no Lagrangian, no solver**; resource model =
45nm CMOS energy (post-hoc accounting). Results: 59.2% energy, 2.24× throughput, ≤3.6%
accuracy drop, on a simulated 3-layer transformer + Llama-2-7B. **No Apple affiliation
found (abs page lists none; UNVERIFIED beyond that) — treated as SEED, not an Apple paper.**

**The fork (their assumptions vs ours):**
- their units = runtime tokens → **ours = the witness `code` rows: 1200 per-frame latents,
  the exact per-token analog, and a granularity the settled #336 per-TENSOR line never
  expressed**;
- their saliency = activation magnitude proxy → **ours = MEASURED signals under the frozen
  contest scorer** (per-pair baseline d_seg; frame-role structure; and the exact per-pair
  quantization RESPONSE already measured at n600 by the 07-13 #336 artifact);
- their resource = CMOS energy → **ours = archive bytes (the rate term), realized as REAL
  brotli on the exact #202 blob grammar, receiver UNCHANGED, zero side-info** (per-row QDQ
  on the tensor-global absmax grid; low-bit rows = fewer distinct int8 symbols).

What survives the fork: the multi-tier per-UNIT allocation STRUCTURE. What we replace:
the saliency functional and the allocator (threshold cascade → exact per-unit Lagrangian).

**Structural key (verified in `probe_witness_sensitivity_bitalloc._render_pair`):**
code row `2i` = frame_0 of pair i → PoseNet-only exposure; row `2i+1` = frame_1 → SegNet
(d_seg) + PoseNet; the self-orient loop reads ONLY row `2i+1`. Therefore **pair i's d_seg
depends ONLY on its own f1 row → joint composition over pairs is EXACTLY additive** —
this escapes the 07-13 cross-tensor joint non-additivity REJECT by construction, and it
makes an EXACT per-pair KKT solvable from the already-measured per-pair uniform-rung rows.

## 2. Already-settled boundary (proactive recall; none of this re-measured)

- #336 per-tensor aggregate WF: transfers at ep425 (n96, −15% d_seg at matched bytes) but
  V9-ep150 n600 joint KKT bundle = **REJECT** (d_seg 0.0337→0.1200; non-additive across
  tensors) — `witness_sensitivity_bitalloc_336_20260713.md`, scope INSTANCE×FORMULATION.
- #336 SPARC-grain per-class functional (n48, ep650): **DOMINATED** (worse at both budgets,
  worse even on Lane miss) — `rate_instruments_apply_20260710.md`.
- The 07-13 artifact (`witness_sensitivity_bitalloc_336_20260713T042157Z/`) holds the full
  18-tensor × int8..int2/zero/mean **n600 per-pair response matrix** on the SAME checkpoint
  this work targets (`levelset_witness_ema_BEST.npz`, sha `2599ad8b…`), baseline all-int8 =
  63,664 B, d_seg 0.03365824, d_pose 151.79642. Reused here BY BYTE IDENTITY (see §4).
- Margin-adaptive per-layer precision (07-14 codex arms) = COMPUTE-side (scorer forward
  speed), disjoint from this RATE-side lane; no surface collision.

## 3. Design: six arms, one control, one falsifier

All arms: params at the shipped int8 grammar (unchanged); ONLY the `code` tensor gets a
per-row bit map (QDQ at tensor-global absmax grid, then the unchanged int8 grammar; all-8
map reproduces the baseline blob byte-identically — pinned as a custody check).

| arm | f0 rows (600) | f1 rows (600) | signal |
|---|---|---|---|
| role_f0int3_f1int8 | int3 | int8 | frame-role only (d_seg structurally invariant) |
| role_f0int2_f1int8 | int2 | int8 | frame-role, aggressive |
| role_f0int3_f1int4 | int3 | int4 | role + flat f1 (byte-matches uniform int4) |
| amc3_salient | int3 | top20%@8 / next30%@5 / rest@3 by per-pair baseline d_seg | AMC percentile tiers, error-mass saliency |
| amc3_random | int3 | same tier sizes, random assignment (seed 20260714) | falsifier: does the saliency SIGNAL matter? |
| pairkkt_f0int3 | int3 | per-pair Lagrangian argmin over rungs {8..3} using the MEASURED per-pair d_seg response | exact per-unit allocation (HAWQ-done-right) |

Anchors (prior MEASURED n600 rows, reused by byte identity): baseline int8; uniform code
int5/int4/int3. Decision surface: each tiered point vs the piecewise-linear interpolation
of the MEASURED uniform RD curve at the same bytes.

## 4. Custody + bytes (MEASURED, real brotli on the exact blob)

Custody re-derivation: baseline, all-8 tier map, and uniform int5/int4/int3 candidates
were REBUILT and their archive SHA-256s are **byte-identical** to the 07-13 rows
(`custody_checks.json`) — prior per-pair scores therefore transfer exactly (same bytes ⇒
same frames ⇒ same scores under the identical scorer config), never asserted.

| arm | archive B (MEASURED) | Δ vs baseline 63,664 |
|---|---:|---:|
| role_f0int3_f1int8 | 57,960 | −5,704 |
| role_f0int2_f1int8 | 55,203 | −8,461 |
| role_f0int3_f1int4 | 51,953 | −11,711 (22 B UNDER uniform int4) |
| amc3_salient | 52,762 | −10,902 |
| amc3_random | 52,992 | −10,672 |
| pairkkt_f0int3 | 52,981 | −10,683 |

## 5. Pre-registered predictions (DERIVED from measured per-pair rows, BEFORE the joint run)

- role_f0int3_f1int8 / role_f0int2_f1int8: d_seg = baseline **exactly** (f1 rows untouched;
  structural). Information content = d_pose response to f0 coarsening + bytes.
- role_f0int3_f1int4: d_seg = uniform-int4's 0.03380370 exactly.
- amc3_salient: 0.03401882 · amc3_random: 0.03392613 — **the AMC error-mass percentile
  heuristic is predicted DOMINATED** (worse than random, worse than uniform int4): "high
  baseline d_seg" ≠ "high quantization sensitivity".
- pairkkt_f0int3: **0.03152334 — predicted BELOW the int8 baseline at −10,683 B** (per-pair
  argmin harvests measured non-monotone response; encoder-side per-video search is the
  contest's own game, receiver unchanged; lineage: PR101 per-pair correction sidecar).
- d_pose: NOT recombinable (depends on both rows jointly) → fresh-measured only.

## 6. Joint n600 scoring status: OWED — governed admission REFUSED (recorded honestly)

The fresh joint scoring run (exact shipped inflate runtime + frozen CPU scorers, resumable
per pair) was REFUSED twice by the system admission gate:
`projected system-used 104.2 GiB EXCEEDS adaptive ceiling 74.8 GiB (current 48.2 +
active-growth 50.0 + new 6.0)`. The +50.0 GiB is UNKNOWN_GROWTH_HEADROOM (+25 GiB each)
charged to two UNREGISTERED material sibling processes of this same operator wave
(`click_polish_block_loop.py`, `probe_genuine_frame_nterm_n600.py`). A governor REFUSE is
information, not an obstacle: no override was applied (operator-verbatim required), no
ungoverned launch was made. **Resume command (runs to completion in ~50 min once the
siblings release the machine; resumable rc=7 boundaries):**

```bash
TAC_GOVERNED_ADMISSION=1 .venv/bin/python tools/safe_run.py --rss-mb 6144 -- \
  .venv/bin/python tools/apply_amc_saliency_tiered_bitalloc_witness.py \
    --ckpt-dir experiments/results/v9_cgauge_432_coherent_arm_20260711 \
    --out-dir experiments/results/amc_saliency_tiered_bitalloc_20260714 \
    --prior-dir experiments/results/witness_sensitivity_bitalloc_336_20260713T042157Z \
    --torch-threads 1 --chunk-seconds 420
```

**What is already evidence-grade without the fresh run** (all bytes MEASURED; d_seg values
below are DERIVED-EXACT recombinations of MEASURED n600 per-pair rows — deterministic
renderer, pair-local f1 rows, identical scorer batch composition across units — labeled
DERIVED, never promoted to MEASURED; d_pose is NOT recombinable and stays OWED):

| arm | bytes MEASURED | d_seg (DERIVED-exact) | vs uniform curve @ same bytes | d_pose |
|---|---:|---:|---|---|
| baseline int8 (anchor) | 63,664 | 0.03365824 (MEASURED) | — | 151.796 (MEASURED) |
| role_f0int3_f1int8 | 57,960 | 0.03365824 (= baseline, structural) | uniform@57,960 ≈ 0.033686 → **−0.000028** | OWED |
| role_f0int2_f1int8 | 55,203 | 0.03365824 (= baseline, structural) | uniform@55,203 ≈ 0.033719 → **−0.000061** | OWED (risk arm) |
| role_f0int3_f1int4 | 51,953 | 0.03380370 (= uniform int4, structural) | uniform@51,953 ≈ 0.033805 → ≈0 | OWED |
| amc3_salient | 52,762 | 0.03401882 | uniform@52,762 ≈ 0.033779 → **+0.000240 (LOSES)** | OWED |
| amc3_random | 52,992 | 0.03392613 | uniform@52,992 ≈ 0.033772 → +0.000154 (loses) | OWED |
| pairkkt_f0int3 | 52,981 | 0.03152334 | uniform@52,981 ≈ 0.033772 → **−0.002248 (WINS)** | OWED |

**Verdict-scope (current evidence):** INSTANCE×FORMULATION, advisory axis, DERIVED d_seg —
(a) the naive AMC transfer (error-mass percentile tiers) is DOMINATED (worse than its own
random-assignment falsifier and worse than uniform at matched bytes): a real
implementation-level negative for the seed's saliency proxy under our premises, NOT a
family NO-GO; (b) the family completion — exact per-pair MEASURED-response allocation
(`pairkkt_f0int3`) — is predicted to dominate the entire uniform RD curve (d_seg below the
int8 BASELINE at −10,683 B); (c) frame-role tiering is a free −5.7…−8.5 KB at structurally
unchanged d_seg, gated only by the owed d_pose response. Fresh joint n600 rows (the resume
command above) are REQUIRED before any adoption decision; the allocation is solved on the
advisory scorer axis and its exact contest-CPU transfer is a further owed row.

## 7. Full-surface Apple-ecosystem sweep (operator-broadened: ALL aspects, we run ON Apple)

Ranked by transfer, each mapped to OUR surfaces:

1. **HAWQ-V2 (non-Apple anchor, arXiv 1911.03852)** → RATE (#336/#157). Hessian-trace ×
   quantization-perturbation waterfill. This lane instantiates its per-UNIT exact version
   with the Hessian surrogate replaced by the MEASURED response. Anchor citation for the
   canonical equation below.
2. **MLX / mlx-lm quantization (Apple OSS)** → RATE (#336, archive format) + COMPUTE
   (#252/#443/#478 kernels) + PRECISION (#470/#477/#496). Group-wise affine quant,
   `quant_predicate` per-layer bits, `dynamic_quant` sensitivity→target-bpw loop, DWQ
   teacher-distilled scales. Same framework as our trainer: the natural implementation
   substrate if tiered code quant is adopted at TRAIN time (QAT form). LLM recipe names
   don't transfer; the mechanism does.
3. **DKM differentiable k-means palettization (Apple, arXiv 2108.12659)** → RATE + the
   train-time witness itself: codebook learned under the REAL task loss (d_seg through R is
   differentiable in our MLX trainer). The strongest Apple mechanism for weights-as-payload;
   candidate future lever `PalettizedCodeLever` (HELD, see §8).
4. **Apple Foundation Models mixed 2/4-bit + accuracy-recovery adapters (2507.13575)** →
   RATE + on-device FM surface (#259 fmtools). Pattern: quantize the trunk brutally +
   recover with a tiny COUNTED high-precision residual — exactly our sidecar-shaped-bytes
   discipline; the byte-split is a measurable ΔS/byte decision. Their ASTC-container trick
   = sister of C1a coder-aware regularization.
5. **coremltools optimization stack (palettization/per-block quant/sparsity; per-module
   OptimizationConfig)** → PRECISION (#477 format matrix) + ANE teacher-forward (#482/#484):
   per-module mixed technique config is the software shape for per-layer INR allocation;
   the ANE-targeted palettization notes feed the #482 correction-ladder work.
6. **Talaria (Apple, CHI'24)** → APPARATUS/observability: per-op cost/benefit ledger with
   simulated optimization impact = our sensitivity-map + costate-digest surfaces extended
   to quantization decisions (means, not a score path).
7. **COIN++ (non-Apple, arXiv 2201.12904)** → RATE prior CONFIRMED by our measurement:
   modulation-like tensors tolerate ~5 bits ≫ trunk weights; our measured code curve
   (int4 ≈ +0.00015 d_seg at −11.7 KB) is the witness-side instance.
8. **AWQ/OWQ (non-Apple)** → PRECISION/RATE grain: protect the salient ~1% via equivalent
   scaling instead of mixed-precision bookkeeping — a candidate byte-free variant for the
   base tensors (where the 07-13 joint REJECT lives).
9. **LLM-in-a-flash / LazyLLM / KV-Runahead (Apple)** → UNIFIED-MEMORY + compute scheduling
   meta-lessons only (predictable-sparsity exploitation; our blind-coordinate #401 is the
   analog); no rate transfer.

LLM/token-specific and NOT transferable anywhere: token pruning/eviction as such, KV-cache
mechanics, attention-topology quant recipes, energy-per-MAC objectives.

## 8. HELD wire-ins (for main to land serially — shared surfaces owned by sibling arms)

1. **Canonical equation (HELD spec)** `amc_perrow_tiered_code_bitalloc_v1`:
   - Law: for payload tensors whose rows are PAIR-LOCAL through the receiver (witness
     `code`), joint d_seg composes additively over rows ⇒ per-unit allocation from
     MEASURED per-unit response is exact; proxy-saliency percentile tiers (AMC-style) are
     dominated by response-based allocation; frame-role tiering (f0 vs f1) is a free d_seg
     invariance.
   - EmpiricalAnchors: the §6 measured rows (artifact
     `experiments/results/amc_saliency_tiered_bitalloc_20260714/amc_tiered_report.json`)
     + the 07-13 baseline/uniform anchors (byte-identity custody).
   - Producers: `tools/apply_amc_saliency_tiered_bitalloc_witness.py`. Consumers:
     `#157 waterfill allocator` (per-row grain), byte-close packing decisions (#202/#406).
2. **DSL Lever: N/A-with-reason for the measurement tool** (FEED-07l precedent: compress-
   half measurement flags are outside the trainer-DSL domain). **IF adopted at train time**,
   the held spec is `TieredCodeQATLever` — factory emitting
   `--code-row-bits-map <path>` + `--code-qat-tiered` (flags DO NOT EXIST yet; must be
   born through the DSL per never-invent-flags), LawRef → `amc_perrow_tiered_code_bitalloc_v1`,
   consumer = the levelset trainer's code-quantization path; default OFF, registered with
   duty-to-measure.
3. **Preflight:** no new gate owed (no new bug class; tool reuses gated surfaces). If main
   wants self-protection for the recombination claim, the cheap pin is a test asserting
   `tiered_code_qdq(all-8) ⇒ baseline blob sha` (already enforced at runtime fail-closed).
4. **Duty ledger:** one firing to record — `amc_saliency_tiered_bitalloc` (measured, this row).

## 9. DAG FEED (HELD — for main to append to the canonical DAG)

> ## FEED-amc-tiered (2026-07-14) — AMC warm-start: per-ROW tiered code bit-alloc MEASURED n600
> **Seed arXiv 2607.10109 deep-read → divergence fork → per-row tier design → $0 measured rows.**
> Units = code rows (per-frame latents; the per-token analog). Structural additivity: pair-local
> f1 rows ⇒ joint composition exact ⇒ the 07-13 cross-tensor non-additivity REJECT does not bind
> at this granularity. MEASURED (bytes, real brotli, receiver unchanged): role/amc3/pairkkt arms
> −5.7 KB … −11.7 KB vs the 63,664 B int8 baseline. <!-- RESULTS_SUMMARY: filled at completion -->
> Verdict scope: INSTANCE×FORMULATION (frozen V9 ep150 EMA-best; post-hoc per-row QDQ). Ranked
> queue: (1) adopt winning arm into the #406 byte-close default IF the joint row dominates the
> uniform RD curve; (2) per-row grain for `film`/hidden via response-measured groups (needs new
> response rows); (3) TieredCodeQATLever (train-time, DSL-held); (4) DKM palettized code
> codebook. Pointer 0.19108282 UNMOVED (means/advisory).

## 10. Launch ticket (operator-GO required — NOT launched)

IF the measured §6 winner dominates the uniform curve at matched bytes AND main adopts it
into the byte-close default: the next exact row is a contest-CPU `upstream/evaluate.py`
replay of a full witness archive built with the winning row-bit map (paid/heavy = operator-GO;
the witness line is still S≈4+ advisory at this checkpoint, so this ticket matters only
once a competitive witness checkpoint exists — the LEVER is checkpoint-portable, the
allocation must be re-solved per checkpoint from its own response rows).

## STORES CONSULTED

CLAUDE.md; AGENTS.md; docs/operating_manual_craft_handoff.md; src/tac/subagent_contract.py
(PAPER_WARM_START_FROM_DIVERGENCE); the canonical DAG (FEED-07i/j/k/l, FEED-crucible2-SEALED,
FEED-494); `.omx/research/witness_sensitivity_bitalloc_336_20260713.md`;
`.omx/research/sensitivity_bitalloc_witness_n96_20260707.md`;
`.omx/research/rate_instruments_apply_20260710.md`; the margin-adaptive 07-14 codex feeds;
`tools/probe_witness_sensitivity_bitalloc.py`; `tools/apply_sensitivity_bitalloc_witness.py`;
`tools/apply_perclass_bitalloc_witness.py`; the 07-13 artifact root (response curves +
resume state + custody).
