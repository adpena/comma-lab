# Procedural Predictor Plus Residual Equation Landed

**Author**: codex  
**UTC**: 2026-05-21T01:05:24Z  
**Equation**: `procedural_predictor_plus_residual_correction_savings_v1`

## Why

Catalog #359 correctly prevents canonical equation #26
(`procedural_codebook_from_seed_compression_savings_v1`) from being applied to
residual-hybrid magic-codec contexts. The replacement equation says:

```text
Delta S = -25 * (N_codebook - K_seed) / 37,545,489
```

That is valid for direct codebook replacement. It is not valid for
predictor-plus-residual stacks, where the archive still pays for the residual
stream.

## New equation

The new sister equation uses the correct byte accounting:

```text
Delta S_rate = 25 * (K_predictor + R_residual + H_envelope - N_original) / 37,545,489
```

It is explicitly `rate_axis_only_no_scorer_distortion_claim`; every emitted
prediction carries:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `rank_or_kill_eligible=false`
- `promotable=false`

## Anchors

Two existing magic-codec residual smokes become zero-residual anchors under the
correct formula:

| Anchor | Original bytes | Predictor bytes | Residual/envelope bytes | Delta bytes | Delta S |
|---|---:|---:|---:|---:|---:|
| pair #1 DWT detail dense-stream residual | `131779` | `96` | `186958` | `+55275` | `+0.036805353633828024` |
| pair #2 FEC6 null-byte SRL1 residual | `16292` | `32` | `97441` | `+81181` | `+0.05405509567341099` |

Interpretation: both residual-hybrid attempts were rate regressions because
residual bytes dominated. This does not kill the magic-codec residual paradigm;
it gives future residual-hybrid smokes the correct accounting primitive.

## Files

- `src/tac/canonical_equations/procedural_predictor_residual_savings.py`
- `src/tac/canonical_equations/tests/test_procedural_predictor_residual_savings.py`
- `.omx/state/canonical_equations_registry.jsonl`

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/canonical_equations/tests/test_procedural_predictor_residual_savings.py src/tac/canonical_equations/tests/test_procedural_codebook_savings_domain_refinement.py src/tac/tests/test_check_359_residual_hybrid_misapplication.py
```

Result:

- `57 passed`
- registry append succeeded with `2` empirical anchors

