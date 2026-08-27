The Stage-0 deadlock is cured, but the exact selected object hit a charter-defined storage stop before Stage 2.

- Cure commit: `72eb83538b`
- Selected-object gate commit: `971c6d52cc`
- Memo/ledger commit: `d37ee4b05a`
- Durable handoff: [ddm_bs4x_stage0_cure_and_stage_fire_20260826.md](/Users/adpena/Projects/pact/.omx/research/ddm_bs4x_stage0_cure_and_stage_fire_20260826.md)

Both post-cure Stage-0 runs returned `READY_FOR_STAGE_1`, with 151/151 identity controls passing, consistent DX2 pins, a free scorer slot, and zero scorer forwards. The original checkpoint remained byte-identical.

The selected codes have minimum endpoint margin 200, forcing 177 retained candidate evaluations per pair rather than Stage 0’s universal minimum of 67:

- Mandatory payload: `51,859,719,936 B`
- Reserve: `8,589,934,592 B`
- Fire floor: `60,449,654,528 B`
- Observed free: `48,102,899,712 B`
- Shortfall: `12,346,754,816 B`

The typed refusal is retained at:

`/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/checkpoints/stage_15_selected_storage_preflight.json`

SHA-256: `6a53dda5d2bd8cb40c2bc1c85309955577962f24f816644310656731f5765749`

Stages 1–4 did not fire. Stage 5 remains gated. No `d_seg`, `d_pose`, final bytes, recovery factor, or `S` was measured, so BS4X neither confirms nor revises BO2’s instance-scoped 209.07× refusal. The own-vehicle frontier remains GB1 at `S=0.14811799921260607`, 180,215 B, `[contest-CUDA T4 n600]`.

Verification passed: focused tests `3 passed`, Ruff, compilation, serializer post-content checks, ledger validation, custody hashes, unchanged `upstream/`, and clean worktree. Prior memory informed denominator and scope discipline only; current facts were verified from live artifacts.

## NEXT_IF_RESUMED

- **BLOCKED-WITH-A-FIRE-ORDER** — owner: `MAIN storage/scorer router`; consumer store: `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/`; fire trigger: at least `60,449,654,528 B` free under certify-or-block custody, scorer slot free, and every pin revalidated; action: begin the immutable order at Stage 1 and retain additive Stage 1–4 checkpoints.
- **QUEUED-BEHIND-THE-EXACT-SOLVE** — owner: `MAIN scorer-lane successor`; consumer store: `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/checkpoints/stage_50_learned_implicit_screen.json`; fire trigger: retained Stage-4 same-instrument measurements and real-byte arithmetic leave the learned screen live; action: run only the labeled deterministic holdout screen.

## LIVE-HYPOTHESES

- Fresh exact-object QS5 compensation may reduce stale-carrier pose debt because the known defect is cross-object compensation transfer; RJ2’s n1 45.073% recovery supports possible improvement, but not a majority or 10× claim.
- The resolved object is unlikely to beat GB1 if its last-frame Seg damage resembles BO2, since frame-0 compensation cannot repair frame-1 Seg damage.
- Lossless candidate-payload storage might reduce physical capacity, but it requires an amended charter proving exact recovery, per-candidate hashes, and resumability.

## DEAD-ENDS

- Do not use the old `28,220,450,048 B` trigger; the selected object falsifies it.
- Do not launch Stage 2 at current AP capacity or redirect, split, discard, recompute, delete, or move retained custody.
- Do not substitute stale compensation, CP135, linear/autograd proxies, prefixes, or Stage 5 for the exact fire order.
- Do not claim scientific measurements or a formulation/family refusal: the scorer denominator is `0/32`.

