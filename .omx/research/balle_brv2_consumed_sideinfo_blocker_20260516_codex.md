# Ballé BRV2 Consumed-Sideinfo Blocker

Date: 2026-05-16
Lane: `lane_substrate_balle_renderer_20260512`
Scope: Ballé/CompressAI consumed-sideinfo BRV2 audit/prototype
Evidence grade: local code audit + unit guard, no score claim

## Finding

`BRV1` is not a valid non-smoke Ballé monolithic archive contract for score-bearing runs. It stores `LATENTS_BLOB` as raw int16 render latents and stores `SCALES_BLOB` as raw int16 hyper-latents. The runtime checks that `SCALES_BLOB` matches `hyper_analysis(latents)`, but valid side-info does not drive entropy decode or pixel reconstruction. This is the raw-int16/render-silent side-info issue.

## Fail-closed closure

The Ballé runtime now refuses non-smoke `BRV1` packets before rendering. `BRV1` remains usable only for explicit research packets or small smoke fixtures. A smoke fixture must declare `smoke=true`, carry `sideinfo_consumption_contract="brv1_smoke_closure_check_only"`, and stay at 16 pairs or fewer.

Guard:

```text
src/tac/substrates/balle_renderer/inflate.py::_require_non_smoke_sideinfo_decode_contract
```

Blocker string:

```text
BRV1 raw-int16 side-info is render-silent: LATENTS_BLOB carries the rendered latent values directly, while SCALES_BLOB is closure-checked only. Non-smoke Ballé monolithic archives require a BRV2 consumed-sideinfo decode contract before inflate.
```

Regression tests:

```text
src/tac/substrates/balle_renderer/tests/test_balle_renderer_roundtrip.py::test_inflate_refuses_non_smoke_brv1_render_silent_sideinfo
src/tac/substrates/balle_renderer/tests/test_balle_renderer_roundtrip.py::test_inflate_refuses_large_smoke_tagged_brv1
src/tac/substrates/balle_renderer/tests/test_balle_renderer_roundtrip.py::test_inflate_accepts_closed_scales_stream
src/tac/substrates/balle_renderer/tests/test_balle_renderer_roundtrip.py::test_inflate_rejects_mutated_scales_stream
```

## BRV2 requirement

The next non-smoke monolithic Ballé archive must not be `BRV1`. `BRV2` must make hyper-latents operational by consuming them in the decode path that reconstructs main latents before rendering. Acceptable BRV2 evidence:

- archive grammar declares `BRV2` magic/schema and no raw-int16 direct-render main-latent authority;
- inflate reconstructs main latents through a sideinfo-conditioned decode path before `model.latents.copy_`;
- mutating hyper-latents changes reconstructed latents or output pixels, not merely pass/fail status;
- a focused test proves hyper-latent byte mutation changes decode/render output;
- exact eval remains blocked until paired CPU/CUDA custody lands.

Until those conditions exist, Ballé non-smoke dispatch should stop at this blocker.
