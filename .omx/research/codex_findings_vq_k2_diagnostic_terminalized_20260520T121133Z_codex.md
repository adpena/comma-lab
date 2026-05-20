# Codex Findings - VQ K=2 Diagnostic Terminalized

**UTC:** 2026-05-20T12:11:33Z
**Lane:** `lane_e7_vq_k_sweep_plus_e8_sgld_convergence_prep_20260518`
**Call ID:** `fc-01KS21XSVGM2KJ5ET0ET3YCCFN`
**Dispatch label:** `substrate_vq_vae_k_sweep_modal_a10g_diagnostic_dispatch_20260520T064144Z_k_2_codex_a10g_diagnostic_20260520`
**Score claim:** false
**Promotion eligible:** false
**Rank/kill eligible:** false

## Verdict

`COMPLETED_DIAGNOSTIC_NO_SCORE_AUTHORITY`.

The corrected VQ K=2 A10G diagnostic completed and was terminalized in
`.omx/state/active_lane_dispatch_claims.md`. It is a useful implementation and
provenance diagnostic, but it is not a contest score, not a CUDA score, and not
promotion evidence.

## Empirical facts

| Field | Value |
|---|---|
| Harvest dir | `experiments/results/vq_vae_k_sweep_harvest_20260520` |
| Remote rc | `0` |
| Elapsed seconds | `3427.187586582` |
| Codebook size | `2` |
| Alpha rate | `1.0` |
| Best epoch | `19` |
| Best validation Lagrangian | `85.24295806884766` |
| Archive SHA-256 | `fea2cd8af897fcc22525b86a4a6bc9745b47a385cc83c392e01e56fdb93dda76` |
| Archive bytes | `617830` |
| Payload SHA-256 | `07054553b55ac202892e72dfd2c8cbf83f5eeabc2a38905741d917946efee9b8` |
| Payload bytes | `7386569` |
| Auth eval device | `cpu` |
| Score axis | `diagnostic_cpu` |
| Evidence grade | `B` |
| Canonical diagnostic score | `78.07586900258559` |
| Seg contribution | `50.48244` |
| Pose contribution | `27.182041365578122` |
| Rate contribution | `0.4113876370074711` |

Empirical recomputation:

```bash
shasum -a 256 \
  experiments/results/vq_vae_k_sweep_harvest_20260520/lane_substrate_vq_vae_k_sweep_results__output__archive.zip \
  experiments/results/vq_vae_k_sweep_harvest_20260520/lane_substrate_vq_vae_k_sweep_results__output__0.bin
wc -c \
  experiments/results/vq_vae_k_sweep_harvest_20260520/lane_substrate_vq_vae_k_sweep_results__output__archive.zip \
  experiments/results/vq_vae_k_sweep_harvest_20260520/lane_substrate_vq_vae_k_sweep_results__output__0.bin
```

Observed:

```text
fea2cd8af897fcc22525b86a4a6bc9745b47a385cc83c392e01e56fdb93dda76  archive.zip
07054553b55ac202892e72dfd2c8cbf83f5eeabc2a38905741d917946efee9b8  0.bin
617830 archive.zip
7386569 0.bin
```

## Terminal claim

Terminal row appended with:

```text
status=completed_modal_training_recovered_diagnostic_cpu_no_score_claim
agent=codex:vq_k2_terminalize_20260520
```

This closes the newer active claim for
`substrate_vq_vae_k_sweep_modal_a10g_diagnostic_dispatch_20260520T064144Z_k_2_codex_a10g_diagnostic_20260520`.

## Recipe-surface resolution

The stale tracked recipe path
`.omx/operator_authorize_recipes/substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml`
was restored as a fail-closed compatibility pointer, not as a dispatchable
recipe. It now declares:

- `dispatch_enabled: false`
- `research_only: true`
- `platform: none`
- blockers pointing at the corrected A10G diagnostic recipe

The live dispatch surface remains:

```text
.omx/operator_authorize_recipes/substrate_vq_vae_k_sweep_modal_a10g_diagnostic_dispatch.yaml
```

## Interpretation

The K=2 diagnostic confirms the corrected dispatch path now threads
`codebook_size=2` and `alpha_rate=1.0`, binds archive identity to
`archive.zip`, and keeps CPU diagnostic auth-eval out of contest-CUDA claim
fields.

It does not support further paid K-sweep fan-out by itself. The score is far
outside the frontier, the archive is large, and the result is diagnostic-only.
Future VQ work should be reactivated only through a K-dependent archive grammar
or a different component-moving formulation, not by repeating this fixed-int16
diagnostic at nearby K values.
