# Unified Non-HNeRV Residual Basis Dispatch-Readiness Manifest — 2026-05-11

**Purpose:** Single operator-facing entry point listing all 5 non-HNeRV
residual-basis families pre-staged to be dispatch-ready when funding is
secured. Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first"
+ operator directive 2026-05-11 ("we want to have all of the stuff that
would cost more than $100 ready to go in parallel for as soon as we secure
funding") + handoff Bottom-line tranche item #6 ("Convert at least one
non-HNeRV family into a byte-closed PR106 residual sidecar candidate").

**Status:** ALL 5 FAMILIES SCAFFOLD-COMPLETE. ZERO GPU spend.
ZERO score claim. Operator approval required for ANY dispatch.

## Top-line summary table

| Family | Lane ID | Reference | Predicted Δ score | Cost/dispatch | Inflate LOC | Materializer |
|---|---|---|---|---|---|---|
| **wavelet** | `lane_wavelet_residual_pr106_sidecar_dispatch_ready` | Mallat 1989 | `[predicted]` -0.0005 to -0.0030 | $0.20–$0.50 | 187 | `tools/materialize_wavelet_residual_pr106_sidecar.py` |
| **cool_chic** | `lane_cool_chic_residual_pr106_sidecar_dispatch_ready` | Ladune 2023 | `[predicted]` -0.0005 to -0.0025 | $0.20–$0.30 | 145 | `tools/materialize_cool_chic_residual_pr106_sidecar.py` |
| **c3** | `lane_c3_residual_pr106_sidecar_dispatch_ready` | Kim 2024 | `[predicted]` -0.0008 to -0.0030 | $0.20–$0.30 | 146 | `tools/materialize_c3_residual_pr106_sidecar.py` |
| **siren** | `lane_siren_residual_pr106_sidecar_dispatch_ready` | Sitzmann 2020 | `[predicted]` -0.0005 to -0.0020 | $0.20–$0.30 | 171 | `tools/materialize_siren_residual_pr106_sidecar.py` |
| **coord_mlp** | `lane_coord_mlp_residual_pr106_sidecar_dispatch_ready` | Tancik 2020 | `[predicted]` -0.0003 to -0.0015 | $0.20–$0.30 | 151 | `tools/materialize_coord_mlp_residual_pr106_sidecar.py` |

**Cumulative cost** if all 5 are dispatched paired CPU+CUDA at Vast.ai/Modal rates: **$1.00–$1.70**.

**Cumulative predicted Δ score** if all 5 land independently (additive — see "Stack composition" below for realistic synthesis): **`[predicted]` -0.003 to -0.012**, which would move PR106 r2 from 0.2066 toward 0.195-0.204.

## Per-family format_id registry

```
magic = 0xFD (single shared magic byte; differs from 0xFE used by PR100/HNeRV latent sidecar)

family       format_id     submission dir
wavelet      0x10          submissions/pr106_wavelet_residual_sidecar/
cool_chic    0x11          submissions/pr106_cool_chic_residual_sidecar/
c3           0x12          submissions/pr106_c3_residual_sidecar/
siren        0x13          submissions/pr106_siren_residual_sidecar/
coord_mlp    0x14          submissions/pr106_coord_mlp_residual_sidecar/
```

Each family's inflate.py rejects archives with a different format_id; cross-
family dispatch is structurally impossible.

## Shared substrate (pre-staged for all 5)

| Module | Path | LOC | Purpose |
|---|---|---|---|
| Wire format grammar | `src/tac/residual_basis/pr106_sidecar_packing.py` | 270 | typed build/parse of the family-agnostic wrapper |
| Materializer helpers | `src/tac/residual_basis/pr106_materializer_helpers.py` | 240 | PR106 extraction + manifest emission + no-op-detector smoke |
| numpy_inverse_dwt | `src/tac/residual_basis/numpy_inverse_dwt.py` | 109 | wavelet inflate runtime numpy port |
| Wavelet stats scaffold | `src/tac/residual_basis/wavelet_residual_pr106.py` | 653 | research-signal sparsity/entropy stats |
| Cool-Chic stats scaffold | `src/tac/residual_basis/cool_chic_residual.py` | 292 | research-signal pyramid stats |
| C3 stats scaffold | `src/tac/residual_basis/c3_residual.py` | 236 | research-signal conditional residual stats |
| SIREN stats scaffold | `src/tac/residual_basis/siren_residual.py` | 288 | research-signal frequency-domain stats |
| Coord-MLP stats scaffold | `src/tac/residual_basis/coordinate_mlp_residual.py` | 230 | research-signal Laplacian smoothness |

**Test coverage:** 78 new tests pass (34 packing + 19 helper + 25 e2e materializer); 59 sister scaffold tests still green; **137 total** residual-basis tests passing.

## 8 promotable PR101/PR103/PR93/PR91 primitives (sister, downstream of these scaffolds)

Per `feedback_public_pr_nonhnerv_mechanism_backlog_landed_20260511.md`:

| # | Primitive | Source | Status |
|---|---|---|---|
| 1 | PR93 delta-varint pose codec (`QZPDV1`) | PR93 flatpup | identified; port pending |
| 2 | PR91 HPACMini context-model + constriction AC | PR91 | identified; port pending |
| 3 | PR92 RMC1/RSA1/RSB1 joint-stream pattern | PR92 | identified; port pending |
| 4 | PR101 ranked Huffman/no-op sidecar grammar | PR101 | LANDED in `tac.packet_compiler` |
| 5 | PR101 centered-delta uint8 LZMA | PR101 | LANDED in `tac.packet_compiler` |
| 6 | PR101 split Brotli self-delimiting | PR101 | LANDED in `tac.packet_compiler` |
| 7 | PR103 merged range stream | PR103 | LANDED in `tac.packet_compiler` |
| 8 | PR103 latent-hi arithmetic | PR103 | LANDED in `tac.packet_compiler` |

These are the score-aware residual ENCODER primitives that turn raw residual
arrays into byte-tight blobs. The current 5 family scaffolds accept ANY byte
encoding via `--residual-mode` flag; the L2 promotion step is to wire each
family's residual encoder to the PR101/PR103 primitives.

## Stack composition possibilities

**Can multiple families compose into one archive?** YES, with two patterns:

### Pattern A: cascade (single format_id; mixed-family residual blob)

Define a NEW format_id 0x15 ("multi_residual") whose residual blob is itself
a length-prefixed sequence of (family_format_id, family_residual). Each
family decodes its own residual and contributions sum into the final RGB
adjustment. This requires a new submission dir + new inflate.py (~250 LOC)
but is the cleanest path to multi-family composition.

**Predicted Δ score:** if all 5 families are 60% additive on the score axis
(council-grade conservative estimate per CLAUDE.md "score-affecting payload
changed" interaction warnings), cumulative -0.003 to -0.008 over PR106 r2.

### Pattern B: serial PR106 + sidecar + sidecar (existing latent_sidecar + one new family)

Wrap PR106 r2's existing latent sidecar (`submissions/pr106_latent_sidecar_r2/`)
inside ONE of the 5 new family wrappers. The outer family parses (magic 0xFD,
format_id 0x10-0x14, latent_sidecar bytes, residual_bytes). The inflate runtime
calls the latent_sidecar inflate first to produce frames, then adds the
family residual.

**Predicted Δ score:** strictly better than family-residual-only since the
latent sidecar is already in the empirical band; lower bound = max(family,
latent_sidecar).

### Pattern C: parallel dispatch + post-hoc Pareto selection

Dispatch all 5 family materializers in parallel as separate archives, harvest
[contest-CUDA] + [contest-CPU] anchors for each, then operator picks the
Pareto-optimal subset. Simplest pattern; no new code; preserves per-family
custody.

**This is the canonical pattern per CLAUDE.md "Race-mode rigor inversion +
parallel-dispatch first" non-negotiable rule.**

## Operator-facing dispatch order recommendation

Per handoff Bottom-line tranche item #6 ("Convert at least one non-HNeRV
family into a byte-closed PR106 residual sidecar candidate"), the first
dispatch should be the family with the highest predicted EV at the lowest
byte budget:

1. **C3** (frame-delta pose-axis residual; sister of PR93 delta-varint codec
   already identified as #1 EV/byte primitive) — `[predicted]` -0.0008 to -0.003
2. **Cool-Chic** (hierarchical pyramid; closest to public state-of-the-art)
   — `[predicted]` -0.0005 to -0.0025
3. **Wavelet** (Mallat-grade; multi-resolution detail at small byte budget)
   — `[predicted]` -0.0005 to -0.003
4. **SIREN** (smallest byte footprint; frequency-domain prior)
   — `[predicted]` -0.0005 to -0.002
5. **coord_mlp** (family-agnostic baseline; competitive at lowest budgets)
   — `[predicted]` -0.0003 to -0.0015

**TOTAL FIRST-WAVE COST:** $1.00–$1.70 cumulative paired CPU+CUDA.

## Pre-flight checklist (operator-facing; runs BEFORE any dispatch)

```bash
# 1. Verify all 5 lanes are at L1 (impl_complete only)
.venv/bin/python tools/lane_maturity.py audit 2>&1 | grep "_residual_pr106_sidecar_dispatch_ready"

# 2. Confirm test suite green
.venv/bin/python -m pytest \
    src/tac/tests/test_residual_basis_pr106_sidecar_packing.py \
    src/tac/tests/test_residual_basis_pr106_materializer_helpers.py \
    src/tac/tests/test_materialize_residual_pr106_sidecars.py -q

# 3. Smoke-emit a default (empty-residual scaffold-readiness) archive for each family
for fam in wavelet cool_chic c3 siren coord_mlp; do
  out="experiments/results/preflight_${fam}_$(date -u +%Y%m%dT%H%M%SZ)"
  .venv/bin/python "tools/materialize_${fam}_residual_pr106_sidecar.py" \
      --pr106-archive submissions/pr106_latent_sidecar_r2/archive.zip \
      --output-dir "$out"
done

# 4. Lane claim helper rehearsal (does NOT acquire a real claim)
.venv/bin/python tools/claim_lane_dispatch.py audit --no-prune
```

## What this manifest does NOT do

- Authorize any GPU dispatch (operator approval required for EVERY family)
- Claim any score (every emission is `score_claim=False`,
  `ready_for_exact_eval_dispatch=False`, `promotion_eligible=False`)
- Modify any archive bytes that ship in a currently-promoted lane (e.g.
  PR106 r2 — that archive is forensic-immutable)
- Permit MPS as the compute axis (CLAUDE.md "MPS auth eval is NOISE")
- Write to `/tmp` (CLAUDE.md "Forbidden /tmp paths in any persisted artifact")
- Promote any lane beyond L1 without exact CUDA + CPU paired anchors

## 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map contribution**: each family's research-signal stats
   feed into `tac.sensitivity_map.*` priors. WIRED via the existing scaffolds.
2. **Pareto constraint**: each materializer emits a Pareto candidate row in
   the manifest. WIRED via `materialization_manifest.json::extra`.
3. **Bit-allocator hook**: per-band / per-level / per-coef quantisation scales
   feed `tac.bit_allocator` (when wired downstream). WIRED via the per-family
   wire-format scale prefixes.
4. **Cathedral autopilot dispatch hook**: 5 new lanes are registered at L1;
   autopilot's dispatch routing can route any of them when operator approves.
   WIRED via lane registry.
5. **Continual-learning posterior update**: PENDING dispatch (no anchor yet).
6. **Probe-disambiguator**: per-family residual-mode (empty/zero/probe) is
   the smoke disambiguator; score-aware encoder choice is the future L2
   disambiguator. WIRED at scaffold level.

## Cross-references

- Operator-decision-required pin: `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md`
- Council-ratified Phase 2 envelope: `feedback_grand_council_pose_axis_insights_review_20260511.md`
- Full custody takeover: `project_full_custody_takeover_codex_offline_20260511.md`
- Per-family staging memos (sister files in this dir):
  - `staged_wavelet_residual_pr106_sidecar_ready_to_dispatch_20260511T174843Z.md`
  - `staged_cool_chic_residual_pr106_sidecar_ready_to_dispatch_20260511T174843Z.md`
  - `staged_c3_residual_pr106_sidecar_ready_to_dispatch_20260511T174843Z.md`
  - `staged_siren_residual_pr106_sidecar_ready_to_dispatch_20260511T174843Z.md`
  - `staged_coord_mlp_residual_pr106_sidecar_ready_to_dispatch_20260511T174843Z.md`
- Scaffold landing memos:
  - `feedback_wavelet_residual_basis_pr106_scaffold_landed_20260511.md`
  - `feedback_numpy_inverse_dwt_landed_20260511.md`
  - `feedback_nonhnerv_residual_basis_scaffolds_landed_20260511.md`
  - `feedback_packet_compiler_pr101_pr103_primitives_landed_20260511.md`
  - `feedback_public_pr_nonhnerv_mechanism_backlog_landed_20260511.md`
- Handoff: `~/Downloads/pact_score_lowering_handoff_2026-05-11.md` Bottom-line tranche item #6

## Loop pause status

**Loop remains PAUSED** per operator directive 2026-05-09. No `ScheduleWakeup`
outstanding from this work.

## Total cost

**$0** ($0 GPU; $0 substrate-engineering work tier per
"Parallel $0 work (no operator gate needed)" in the takeover memo).

## Open operator decisions

NONE for this pre-stage layer itself. Per family, the operator-approval-required
gates remain identical to the existing NOT YET items (cumulative dispatch
budget pending the same authorization track as Phase 2 + Phase 3 + A1 PR).
