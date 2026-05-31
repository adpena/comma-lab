# Codex Findings: PR98 / Z7 / PACT-NeRV Runtime Contract Review

UTC: 2026-05-31T00:51:52Z

## Scope

Adversarial review of partner-landed score-lowering surfaces around PR98 L28
channel balance, Z7-Mamba-2 MLX-local smoke/training, and PACT-NeRV repair
archive-family coverage. No score authority is claimed from this review.

## Findings And Actions

1. PR98 L28 decode-side channel balance was score-directed but duplicated the
   canonical offset loop in substrate inflates. I routed NSCS06 and PR101 clone
   inflates through the canonical helper surface and added torch-native helper
   coverage so future runtimes do not reimplement the same three-offset loop.

2. Z7-Mamba-2 MLX now has an explicit backend-lineage blocker. The trainable
   MLX module exists, but the recurrence is still `reference_s6_mlx`, not the
   canonical `tac.substrates._shared.mamba2_ssd` path. Smoke manifests and
   archive metadata now carry `canonical_ssd_mlx_backend_wired=false` with
   blocker `canonical_ssd_mlx_backend_not_wired`.

3. PACT-NeRV archive-family routing now separates score-affecting adapter
   availability from runtime portability. A PACT-NeRV selector packet can have
   a score-affecting adapter implemented while still carrying
   `runtime_portability_blockers=["pact_nerv_inflate_torch_dependency"]` and
   `numpy_portable_inflate=false`.

## Authority

All touched surfaces remain fail-closed:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The next score-lowering action is not another prose audit; it is to route Z7
through the canonical SSD MLX helper or keep demoting any Z7 canonical-SSD claim
until that delegation exists.
