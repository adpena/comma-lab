# Substrate implementation tradition taxonomy — 2026-05-12

**Status**: CANON-1.A operator-approved Option C — preserve both traditions
with explicit reactivation criteria.
**Source**: `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md`
CANON-1.A.
**Wave**: WAVE-A-2 (combined hygiene wave 2026-05-12).
**Operator directive**: "proceed with all, ground in literature first if
possible, $20 or less".

## Two substrate implementation traditions co-exist

The pact codebase hosts TWO valid substrate implementation traditions. They
are NOT competing canonicals; both ship into the canonical substrate
inventory (`canonical_substrate_inventory()`) so the autopilot, Pareto
solver, sensitivity-map, and bit-allocator see every substrate. The
taxonomy is binding per CLAUDE.md "Multiple contenders → multiple paths"
non-negotiable + "KILL is LAST RESORT".

### TRADITION 1 — `src/tac/substrates/<name>/` subpackages

**Lifecycle phase**: L0 SKETCH, `research_only=true` per Catalog #124
declaration at design time.

**Discipline**:

- Catalog #124 STRICT — declare 8 archive-grammar fields at design time:
  `archive_grammar`, `parser_section_manifest`, `inflate_runtime_loc_budget`,
  `runtime_dep_closure`, `export_format`, `score_aware_loss`,
  `bolt_on_loc_budget`, `no_op_detector_planned`.
- 13 HNeRV parity discipline lessons (CLAUDE.md "HNeRV / leaderboard-
  implementation parity discipline").
- 3-clean-pass grand-council adversarial review before any first-anchor
  dispatch.
- Monolithic single-file `0.bin` archive grammar with fixed offsets.
- `inflate.py` <= 100 LOC (default budget; explicit waiver for <= 200 LOC).
- Score-aware training loop (gradient-through-SegNet/PoseNet on the
  contest video).

**Structure** (per-subpackage):

```
src/tac/substrates/<name>/
    __init__.py
    architecture.py     # score-aware substrate model class
    archive.py          # monolithic 0.bin builder/parser
    inflate.py          # <= 100 LOC inflate consumer
    score_aware_loss.py # the Lagrangian
    tests/
        test_<name>_roundtrip.py  # Catalog #91 encoder/decoder roundtrip
```

**Today's 15 substrates** (per `SUBSTRATE_SCAFFOLDS` mapping):

| Subpackage | Inventory id | Literature anchor |
|---|---|---|
| sane_hnerv | sane_hnerv | Chen et al. NeurIPS 2023 |
| balle_renderer | balle_renderer | Ballé et al. ICLR 2018 |
| hybrid_renderer_residual | hybrid_renderer_residual | HNeRV + residual |
| self_compress_nn | self_compress_nn | He et al. 2024 + Selfcomp |
| pr101_lc_v2_clone | pr101_lc_v2_clone | PR101 forensic |
| cool_chic | cool_chic_full_renderer | Ladune et al. ICCV 2023 |
| wavelet | wavelet_full_renderer | Mallat PAMI 1989 |
| grayscale_lut | grayscale_lut | Selfcomp PR #56 |
| vq_vae | vq_vae_substrate | van den Oord et al. NeurIPS 2017 |
| siren | siren_substrate | Sitzmann et al. NeurIPS 2020 |
| block_nerv | block_nerv_substrate | HNeRV per-pair block |
| tc_nerv | tc_nerv_substrate | HNeRV temporal-consistency |
| ff_nerv | ff_nerv_substrate | HNeRV DCT-frequency |
| ds_nerv | ds_nerv_substrate | HNeRV depth-separable |
| hi_nerv | hi_nerv_substrate | HNeRV hierarchical |

### TRADITION 2 — `src/tac/<name>_as_renderer.py` single-file substrates

**Lifecycle phase**: production-mature; many have hit `[contest-CUDA]`
evaluations. Pre-dates the Fields-medal substrate-scaffold subpackage
discipline (2026-05-12).

**Discipline**:

- In-line `MaskRenderer` / `Renderer` class co-located with training
  script + archive builder.
- No separate `inflate.py` <= 100 LOC budget; the inflate logic is often
  inside a shared runtime helper or the trainer ships a custom inflate.
- No monolithic `0.bin` grammar contract; archive layout varies per
  substrate.
- Production-tested: many have empirical anchors recorded in
  `tac.continual_learning` posterior + Lane registry contest_cuda gate.

**Today's substrates** (under `src/tac/<name>_as_renderer.py` or
`src/tac/<name>_renderer.py`):

| File | Inventory id | Tradition lineage |
|---|---|---|
| blocknerv_as_renderer.py | blocknerv | KK NeRV-family expansion 2026-05-11 |
| ffnerv_as_renderer.py | ffnerv | KK NeRV-family expansion 2026-05-11 |
| dsnerv_as_renderer.py | dsnerv | KK NeRV-family expansion 2026-05-11 |
| hinerv_as_renderer.py | hinerv | KK NeRV-family expansion 2026-05-11 |
| tcnerv_as_renderer.py | tcnerv | KK NeRV-family expansion 2026-05-11 |
| mnerv_as_renderer.py | mnerv | HH NeRV-family completion 2026-05-11 |
| vqvae_as_full_renderer.py | vqvae_as_full_renderer | HH NeRV-family completion |
| cnerv_as_renderer.py | cnerv | All-NeRV-family expansion 2026-05-11 |
| e_nerv_as_renderer.py | e_nerv | All-NeRV-family expansion 2026-05-11 |
| ego_nerv_as_renderer.py | ego_nerv | All-NeRV-family expansion 2026-05-11 |
| nervdc_as_renderer.py | nervdc | All-NeRV-family expansion 2026-05-11 |
| lane_12_v2_nerv_as_renderer.py | lane_12_v2_nerv | Phase B-Option-C reference |
| quantizr_faithful_renderer.py | quantizr_faithful | Lane Q-FAITHFUL PR55 reverse |
| mlx_renderer.py | mlx_mask_renderer | MLX Apple Silicon dev/research |
| dp_sims_renderer.py | dp_sims_renderer | DP-SIMS semantic synthesis CVPR 2024 |
| contrib/diffusion_renderer.py | diffusion_renderer | Diffusion-based renderer |

## Why both traditions co-exist (rationale)

The Fields-medal council formalized Catalog #124 + the 13 HNeRV parity
discipline lessons on 2026-05-12 after the leaderboard-implementation-
parity audit found that PR #100 (hnerv_lc_v2) won by binding architecture
+ score-aware training + archive grammar + inflate runtime + export
contract SIMULTANEOUSLY in one ~600-LOC artifact. The substrate-scaffold
subpackage discipline (TRADITION 1) is designed to make every NEW
representation/codec lane bind those five surfaces from byte zero.

The older `<name>_as_renderer.py` files (TRADITION 2) pre-date the
discipline. They are production-mature, but they do NOT enforce the
8-archive-grammar-fields-at-design-time contract. Council members who
reviewed CANON-1.A enumerated four options:

- **Option A** — converge: rewrite TRADITION 2 into TRADITION 1
  subpackages. *Cost: high; risk: behaviour drift in production code.*
- **Option B** — retire TRADITION 1: older single-file is canonical;
  delete substrate scaffolds. *Cost: lose the new discipline; risk:
  silently re-shipping non-Catalog-#124-compliant lanes.*
- **Option C** — explicit taxonomy: preserve both with reactivation
  criteria. *Cost: documentation; risk: future subagent confusion if
  taxonomy is not visible.* **CHOSEN.**
- **Option D** — both at once: build TRADITION 1 + retire TRADITION 2 in
  the same wave. *Cost: highest; risk: convergence storms.*

Council positions per CANON-1.A:
- Shannon + Hotz: Option C — information theoretically optimal; no
  information lost on retirement.
- Selfcomp + Quantizr: Option B (older = canonical, retire scaffolds).
- Operator: Option C — per CLAUDE.md "KILL is LAST RESORT" + "Multiple
  contenders → multiple paths" non-negotiable.

## Reactivation criteria

A TRADITION 1 SKETCH SUBSUMES its sibling TRADITION 2 implementation when:

1. The TRADITION 1 subpackage produces a verified `[contest-CUDA]` archive
   with score `<= 0.21` (sub-Quantizr-0.33-band → medal-band).
2. The auth eval custody record (per `tac.continual_learning`) includes
   `archive_sha256` + `runtime_manifest_sha256` + `inflate_smoke_log` so
   the bytes are byte-closed.
3. Grand council 3-clean-pass adversarial review confirms the TRADITION 1
   subpackage is byte-deterministic, has Catalog #124 8 fields declared,
   and the archive grammar matches the trained-weights export contract.
4. The TRADITION 2 sibling `<name>_as_renderer.py` is archived to
   `.omx/research/historical_substrates/<name>_<YYYYMMDD>/` with
   provenance (commit SHA + last contest-CUDA score + lane registry
   pointer).

Until ALL four conditions hold, the TRADITION 2 sibling remains in
production and the TRADITION 1 scaffold remains L0 SKETCH `research_only=true`.

## Sister-duplicates (residual_basis vs substrate)

CANON-1.C documents three substrate names that appear in BOTH the
residual_basis sidecar tradition (`src/tac/residual_basis/<name>_residual.py`)
AND the substrate-scaffold tradition (`src/tac/substrates/<name>/`):

- **siren** → `residual_basis/siren_residual.py` (sidecar) +
  `substrates/siren/` (full substrate with zero latents).
- **cool_chic** → `residual_basis/cool_chic_residual.py` (sidecar) +
  `substrates/cool_chic/` (full per-frame AR-latent renderer).
- **wavelet** → `residual_basis/wavelet_residual.py` (sidecar) +
  `substrates/wavelet/` (full Mallat scattering renderer).

These are NOT duplicates — they serve different purposes:

- **Residual basis** = SIDECAR on top of an existing host substrate's RGB
  output (composes via `STACKABLE_SERIAL` rule in the composition matrix).
- **Substrate** = REPLACEMENT for the renderer slot entirely (composes via
  `REPLACEMENT` rule with any other RENDERER_REPLACEMENT).

The composition matrix encodes the orthogonality via `SubstrateClass`
(`RESIDUAL` vs `RENDERER_REPLACEMENT`) so the autopilot ranker sees them
as distinct slots and never tries to compose `cool_chic_residual` with
`cool_chic_full_renderer` (rule 12: self-with-self → REDUNDANT alpha=0).

## Migration to a single tradition (long-term)

Per CLAUDE.md "Unified Lagrangian action principle" non-negotiable + GR-
style variational action `S_total(theta, archive_bytes, hardware)`, the
long-term migration target is ONE substrate tradition. The path is:

1. As each TRADITION 1 SKETCH hits its reactivation criteria, archive
   the TRADITION 2 sibling and mark the TRADITION 1 row as the canonical.
2. Eventually, all TRADITION 2 single-file substrates are archived; only
   TRADITION 1 subpackages remain in the live tree.
3. The 8-archive-grammar-fields-at-design-time discipline becomes
   universal; Catalog #124 strict-flip lands at 0 live violations.

Until that day, BOTH traditions are live; the canonical inventory reflects
both; the composition matrix sees both; the autopilot ranks both.

## Wire-in declaration (per Catalog #125)

This memo IS a research-only documentation landing. The 6-hook
wire-in declaration:

| Hook | Status | Rationale |
|---|---|---|
| 1. Sensitivity-map | N/A — META documentation memo, no per-tensor saliency contribution |
| 2. Pareto constraint | EXERCISED — taxonomy formalizes substrate feasibility partitioning the inventory rows the Pareto solver consumes |
| 3. Bit-allocator | N/A — taxonomy does not introduce per-tensor bit allocation changes |
| 4. Cathedral autopilot dispatch hook | EXERCISED — inventory wire-in (Part 2 of WAVE-A-2) feeds autopilot composition ranking |
| 5. Continual-learning posterior update | N/A — taxonomy declaration, no new empirical anchor |
| 6. Probe-disambiguator | EXERCISED — Option C ("ship both, math arbitrates via empirical anchors") IS the probe-disambiguator pattern applied at substrate-tradition scope |

## Cross-references

- `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md`
  (CANON-1.A source)
- `feedback_grand_council_fields_medal_substrate_design_20260512.md`
  (Fields-medal council design memo)
- `feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512.md`
  (TRADITION 1 inventory wire-in)
- `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`
  (the leaderboard-parity audit that motivated Catalog #124)
- `src/tac/substrates/__init__.py` (TRADITION 1 docstring; this taxonomy
  is referenced inline)
- `src/tac/optimization/substrate_composition_matrix.py`
  `canonical_substrate_inventory()` (unified inventory of both traditions)

## Forbidden patterns honored

- ZERO `/tmp` paths in this memo
- ZERO score claims (every score is `[predicted]` or cites an empirical
  evidence-tagged anchor in another memo)
- ZERO MPS-derived strategic decisions
- NO design decision unilaterally adopted (Option C is per CANON-1.A
  Shannon + Hotz consensus + operator decision)
- NO KILL verdicts (both traditions preserved with explicit reactivation
  criteria)
