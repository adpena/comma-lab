# Codex findings — realization G2c cell-interior fills

Date: 2026-07-21
Task: #578, round 3
Lane: `lane_realization_g2c_interior_578_20260721`
Authority: `[macOS-CPU advisory]`; no score claim; pointer `0.1910828242 [contest-CPU]` unmoved.

## Verdict

`MEASURED_INTERIOR_FORMULATIONS_NOT_ADMISSIBLE`.

The exact factor-2 lattice is not the blocker: R1–R4 each transported and
double-decoded exactly on 600/600 pairs.  The best zero-byte formulation was
R2 max-margin, but it produced 0/600 whole-description semantic-exact pairs,
114/3,188 surviving declared writes, and 0/600 pose-tube pairs.  This is a
negative for the measured fixed-magnitude, context-free max-margin, frozen-bank
Hopfield-prox, and single-pixel exception formulations on this seed—not for
spatially contextual, textured, learned, or explicit frame0-carrier families.

## Measured D rows

All entries below are native CPU-Torch hard-oracle measurements through exact
uint8 factor-2 realization.  `writes` means surviving seed-declared semantic
writes; semantic-exact is whole realized argmax equality to the seed description.

| Rung | n16 semantic / writes / pose / bytes | n64 semantic / writes / pose / bytes | n600 semantic / writes / pose / bytes |
|---|---|---|---|
| R1 fixed magnitude | 0/16 · 0/97 · 0/16 · 0 | 0/64 · 0/337 · 0/64 · 0 | 0/600 · 1/3,188 · 0/600 · 0 |
| R2 max margin | 0/16 · 5/97 · 0/16 · 0 | 0/64 · 13/337 · 0/64 · 0 | 0/600 · 114/3,188 · 0/600 · 0 |
| R3 Hopfield prox | 0/16 · 1/97 · 0/16 · 0 | 0/64 · 1/337 · 0/64 · 0 | 0/600 · 18/3,188 · 0/600 · 0 |
| R4 R2-dying exceptions | 0/16 · 2/97 · 0/16 · 492 | 0/64 · 11/337 · 0/64 · 1,748 | 0/600 · 42/3,188 · 0/600 · 16,568 |

R2 n600 class decomposition: class 0 survives 113/1,379; class 1 survives
0/1,807; class 2 survives 1/1; class 4 survives 0/1.  Strata: boundary codim-1
113/3,183, critical event 0/3, movable track 1/2.  Margin buckets are decisive:
all positive-margin writes survive (86/86 in `(0,1]`, 28/28 in `(1,4]`) and
all nonpositive-margin writes die (0/3,074).  The class-1 constant-tile best
margin is `-5.11375105381012`, so the assigned context-free cell formulation
does not contain a Lane interior.

R4 encoded exactly the 3,074 R2 dying write ordinals plus RGB triplets:
15,370 record bytes plus 1,198 nonempty-pair header bytes.  It reduced survival
from 114 to 42 writes and worsened d_seg; source RGB at an isolated scorer pixel
does not recreate the missing spatial SegNet context.

## D3 — winning hard oracle

R2's seed description fidelity was `d_seg=0.3434977213541667`; realized R2 was
`d_seg=0.5106008572048611` against the frozen target and
`d_seg=0.6560023159450954` against the description.  Mean pose MSE against the
frozen target was `158.98986618601776`; outside-tube debt was
`153.7215541738262`; pose tubes held on 0/600 pairs.  The seed has no intra-pair
frame0 appearance carrier, so both synthesized frames use the same decoded
cell field.  This is an instance limitation, not a PoseNet-family verdict.

## D4 — canonical admission

`predict_project_realization_admissibility_v1` remains fail-closed.  Its n600
R2 anchor passes n600, factor-2 exactness, double-decode identity, zero bytes,
and receiver-derived RGB; it fails `semantic_cells_to_rgb_exact` and
`pose_within_declared_tube`.  The existing charged source-RGB anchor is retained
as a separate conjunctive failure.

## Premise corrections and reformulation queue

- #583 supplies an exact bounded Diophantine resize-preimage predicate; it does
  not supply a spatial SegNet argmax-cell polytope or its Chebyshev center.
- The #581 probe's registered rank-4 valid-cell prototypes and logits-to-uint8
  adapter remain absent.  R3 here is a deterministic frozen constant-tile bank,
  not authority for the optimal spatial Hopfield formulation.
- The next honest optimal forms are a spatial contextual cell optimizer, a
  texture/procedural decoder, an explicit frame0 pose carrier, and the registered
  Fisher/secant/QP plus curvelet/shearlet EV-ranked exception path.

## Custody, resumability, and stores consulted

Compact receipt:
`.omx/research/realization_g2c_interior_receipt_20260721.json`, SHA-256
`3e5307c471d9e7feaa28a608496b972431f73f6b060a94b979ca32c86d3bfe64`.
Full SSD receipt SHA-256:
`cef451a404c77fe08ab2d041b0f59972ff73c0a84ba3ff20281b8651835323b8`.
Four immutable stage trees contain 600 pair rows each; their ordered content
hashes are in the compact receipt.  No RGB, camera, logit, or exception payload
bulk was persisted.

Command:

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/measure_realization_g2_lattice.py \
  --interior-rungs --stop-after-prefix 600 \
  --output-root /Volumes/VertigoDataTier/pact/evidence/realization_g2c_20260721/r2_residual_v2 \
  --threads 8 --chunk-size 8
```

STORES CONSULTED: `CLAUDE.md`, `AGENTS.md`, `reports/latest.md`,
`.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, latest
sister memos, `realization_is_quantization_gated_minimal_writes_die_at_uint8_20260720.md`,
`palette_artifact_probe_verdict_20260710.json`, #580 receiver/projector source,
#583 equation/tests, the seed and preserved predecessor SSD stages, both live
inboxes, and the final G2c SSD receipt/stage trees.

MAIN landing review is required.  In particular, MAIN must verify the
formulation scopes, the explicit frame0 limitation, receipt/source hashes, and
the second empirical equation anchor before merge/registry population.
