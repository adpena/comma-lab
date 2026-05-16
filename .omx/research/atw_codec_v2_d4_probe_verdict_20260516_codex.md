# ATW Codec V2 D4 Probe Verdict

- observed_at_utc: `2026-05-16T22:47:41+00:00`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- axis_label: `[diagnostic-CPU; H(latent|scorer_class) probe]`
- dispatch_attempted: `false`

## Verdict

- substrate_id: `atw_codec_v2`
- verdict: `INDEPENDENT`
- I(latent; scorer_class): `0.006385502752` bits/symbol
- H(latent): `7.039023260840` bits/symbol
- H(latent | scorer_class): `7.032637758088` bits/symbol
- Wyner-Ziv gain ceiling fraction: `0.000907157501`
- unique class signatures: `2`

## ATW V2 Phase 2 Consequence

- phase2_status: `defer_measured_a1_latent_class_conditioning_surface`
- recommended_variant: `none`
- next_action: `do_not_dispatch_atw_v2_phase2_from_this_signal`

The measured mutual information is below the canonical independence
tolerance, so ATW v2's class-conditional WZ surface should not receive
Phase 2 lift authority from this class signal. This is a deferral of
the measured A1-latent/class-conditioning configuration, not a kill of
the broader cooperative-receiver paradigm.

## WZ Residual-Surface Proxy

- global-centroid residual entropy: `7.032294474992` bits/symbol
- class-centroid residual entropy: `7.030370042433` bits/symbol
- residual entropy delta: `0.001924432559` bits/symbol
- residual gain fraction: `0.000273656424`

This proxy subtracts per-class, per-latent-dimension centroids before
re-estimating byte entropy. It is diagnostic only; the canonical WZ
decision remains the H(latent|class) verdict above.

## Provenance

- command: `.venv/bin/python tools/run_atw_v2_d4_probe_from_a1.py`
- research_json: `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.json`
- output_dir: `experiments/results/atw_codec_v2_d4_probe_20260516T224741Z`
- A1 archive: `submissions/a1/archive.zip`
- A1 archive sha256: `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- A1 inner member sha256: `8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243`
- class artifact: `experiments/results/tishby_ib_pure_d4_probe_20260516T212557Z/per_pair_segnet_class.json`
- class artifact sha256: `8a552691af26848c00e253086c507f836f9b8cd8d1d75f6b363835655be3ff61`
- latent stream sha256: `a374e01a1d4cd0639d97cd02c70846906e38e10a8bad99215e9b3e8793f30959`
- tiled class stream sha256: `9cedb90528394a7342f1b150f7653e4adf75ae219c0a159c198911a3e00fa23f`
- global residual stream sha256: `c1386149e265efdbe788bf789d2e90caae1397f4e0673db4550a8e22cc658ec6`
- class residual stream sha256: `7e975d26c7f3f578711357b5358aa8b5309485500a9dbd2baa67b82c2fb2104f`

## Reactivation Criteria

1. Replace the saturated per-pair SegNet composite class with a richer
   side-information signal, such as per-region class histograms, logits,
   pose bins, or hard-pair/object-state features.
2. Rerun the same probe on trained ATW v2 residuals rather than A1
   HNeRV latents if a non-promotional timing smoke produces them.
3. Require paired CPU/CUDA exact-eval custody before any score, rank, or
   promotion claim.
