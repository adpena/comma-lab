# VQ-VAE indices_blob procedural residual variant design

timestamp_utc: 2026-05-21T05:09:32Z
agent: codex
lane_id: lane_codex_vq_vae_indices_blob_procedural_variant_20260521
horizon-class: parser_pursuit
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
paid_dispatch_attempted: false
research_only: true
canonical_equations_referenced:
  - procedural_predictor_plus_residual_correction_savings_v1

## Summary verdict

Verdict: L0_SCAFFOLD_COMPLETE_NO_DISPATCH

The VQ-VAE `indices_blob` surface is **not** a direct equation #26 replacement
surface. It stores score-affecting decoder addresses, so pure removal is
refused and direct replacement is not promoted. The landed scaffold implements
a separate `VQPI` procedural-index residual envelope:

```text
indices_blob := b"VQPI" + envelope_header + seed_bytes + brotli(residual_uint16)
indices      := (procedural_predictor(seed) + residual) mod codebook_size
```

This routes byte accounting through
`procedural_predictor_plus_residual_correction_savings_v1`, not
`procedural_codebook_from_seed_compression_savings_v1`.

## Canonical-vs-unique decision per layer

| layer | decision | rationale |
|---|---|---|
| archive grammar | FORK | Raw VQV1 indices require exact int16 length; procedural residual indices require a typed `VQPI` sentinel envelope. |
| predictor | CANONICAL | Uses `tac.procedural_codebook_generator.derive_codebook_from_seed`. |
| byte accounting | CANONICAL | Uses `procedural_predictor_plus_residual_correction_savings_v1` because residual bytes are charged. |
| parser | EXTEND | `parse_archive()` now accepts raw int16 indices or `VQPI` procedural indices. |
| trainer dispatch | DEFER | No paid-dispatch recipe is activated; scaffold is local proof only. |

## Paradigm classification

| paradigm | verdict | reason |
|---|---|---|
| REPLACEMENT-UPSTREAM | REFUSED_FOR_INDICES_BLOB | `indices_blob` is score-affecting and not a learned codebook/LUT replacement surface. |
| RESIDUAL-CORRECTION-DOWNSTREAM | SELECTED | Seed predicts indices; residual stream restores exact indices. |
| REMOVAL | REFUSED | Parser-safe is not score-opaque; removal would alter decode. |

## 9-dimension success checklist evidence

| dimension | status | evidence |
|---|---|---|
| exact reconstruction | PASS | `test_encode_decode_procedural_indices_roundtrip_exact` |
| parser integration | PASS | `test_compose_with_procedural_indices_parses_and_preserves_indices` |
| honest byte math | PASS | `test_compose_with_procedural_indices_records_honest_byte_delta` |
| seed consumption | PASS | `test_mutating_seed_inside_envelope_changes_decoded_indices` |
| malformed input refusal | PASS | sentinel and residual-length tests |
| base VQV1 compatibility | PASS | raw archive still parses |
| equation routing | PASS | residual equation test suite: 82 passed |
| dispatch safety | PASS | no recipe activation; local predeploy unchanged |
| runtime closure | PASS | trainer runtime vendor now includes `indices_procedural_variant.py` and procedural generator package |

## Cargo-cult audit per assumption

| assumption | verdict | correction |
|---|---|---|
| "Parser-safe means removable" | CARGO-CULTED | Parser-safe `indices_blob` is score-affecting; removal refused. |
| "Equation #26 applies to any seed-derived bytes" | CARGO-CULTED | Residual bytes are charged; use the residual equation. |
| "Residual scaffold should promise savings" | CARGO-CULTED | Byte accounting can report `RATE_REGRESSION`; no score claim. |
| "Procedural archive validity follows from composition" | CARGO-CULTED | The previous codebook scaffold proved this false; parser/inflate consumption is now tested. |

## Observability surface

New code:

- `src/tac/substrates/vq_vae/indices_procedural_variant.py`
- `src/tac/substrates/vq_vae/archive.py` parser extension for `VQPI`
- `src/tac/substrates/vq_vae/tests/test_indices_procedural_variant.py`
- `experiments/train_substrate_vq_vae.py` runtime vendoring for the new parser dependency
- `experiments/train_substrate_vq_vae.py` default-off flags:
  `--enable-procedural-indices-residual`, `--procedural-indices-seed-bytes`,
  and `--procedural-indices-generator-kind`
- `scripts/remote_lane_substrate_vq_vae.sh` env ladder for the same flags,
  disabled unless `VQ_VAE_ENABLE_PROCEDURAL_INDICES_RESIDUAL=1`

The public API emits:

- `ProceduralIndicesComposition`
- `derive_procedural_indices_predictor`
- `encode_procedural_indices_blob`
- `decode_procedural_indices_blob`
- `analyze_procedural_indices_blob`
- `compose_with_procedural_indices`

All emitted analysis includes `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Predicted ΔS band

No frontier score band is claimed.

The byte-accounting equation is:

```text
delta_s_rate_only = 25 * (seed_bytes + residual_stream_bytes + overhead_bytes - original_indices_bytes) / 37_545_489
```

Local synthetic smoke:

| field | value |
|---|---:|
| original_archive_bytes | 2740 |
| new_archive_bytes | 2761 |
| original_indices_bytes | 72 |
| replacement_total_bytes | 93 |
| delta_bytes_replacement_minus_original | +21 |
| verdict | RATE_REGRESSION |

This is expected for a tiny synthetic random-ish index grid and is useful
because it proves the scaffold reports regressions honestly.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/substrates/vq_vae/tests/test_indices_procedural_variant.py src/tac/substrates/vq_vae/tests/test_procedural_variant.py src/tac/substrates/vq_vae/tests/test_vq_vae_roundtrip.py
```

Result: `49 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider src/tac/tests/test_parser_safe_methodology_extension_smoke.py src/tac/tests/test_procedural_replacement_surfaces.py src/tac/canonical_equations/tests/test_procedural_codebook_savings_domain_refinement.py src/tac/canonical_equations/tests/test_procedural_predictor_residual_savings.py src/tac/tests/test_check_359_residual_hybrid_misapplication.py
```

Result: `82 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/local_pre_deploy_check.py --strict --trainer experiments/train_substrate_vq_vae.py --recipe substrate_vq_vae_k_sweep_modal_a10g_diagnostic_dispatch
```

Result: `ALL 9 CHECKS PASSED. Safe to dispatch.`

## 6-hook wire-in declaration

- Hook #1 sensitivity-map: deferred until real trained VQ-VAE archive indices exist.
- Hook #2 Pareto constraint: active through residual byte accounting.
- Hook #3 bit-allocator: active as an indices-section byte budget candidate.
- Hook #4 cathedral autopilot dispatch: research-only; no provider dispatch is enabled.
- Hook #5 continual-learning posterior: deferred until a trained archive anchor exists.
- Hook #6 probe-disambiguator: active; the scaffold distinguishes removal, direct replacement, and residual correction.

## Next action

Run this scaffold against a real VQ-VAE diagnostic archive if/when one is
harvested. Promote only if real indices have compressible residual structure;
otherwise record `RATE_REGRESSION` and keep the lane as a negative-control
guardrail against equation #26 overreach.
