# VQ-VAE Procedural Decoder Inflate-Faithfulness Fix

timestamp_utc: 2026-05-21T05:02:52Z
agent: codex
lane_id: lane_codex_vq_vae_procedural_decoder_inflate_faithfulness_fix_20260521
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
paid_dispatch_attempted: false
evidence_grade: "[empirical:local-cpu-tests]"

## Summary verdict

Verdict: PREREQUISITE_FIX_LANDED_BEFORE_VQ_VAE_INDICES_BLOB_VARIANT

The queued VQ-VAE directive targets the 192-byte `indices_blob` procedural
variant. Before building that L0 scaffold, Codex found a more basic
inflate-faithfulness blocker in the already-landed VQ-VAE procedural codebook
surface: `compose_with_procedural_codebook()` emitted a `VQVP` procedural
decoder envelope, but `parse_archive()` and `inflate_one_video()` still
expected the decoder section to be plain brotli-pickled state_dict bytes.

Empirical pre-fix proof:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/substrates/vq_vae/tests/test_procedural_variant.py
```

Result: `2 failed, 19 passed`.

Direct parse proof:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from src.tac.substrates.vq_vae.tests.test_procedural_variant import _make_synthetic_vqv1_archive_bytes, _CANONICAL_SEED_32B
from tac.substrates.vq_vae import compose_with_procedural_codebook, parse_archive
new = compose_with_procedural_codebook(_make_synthetic_vqv1_archive_bytes(), _CANONICAL_SEED_32B)
parse_archive(new)
PY
```

Result: `error brotli: decoder failed`.

## Fix

- `src/tac/substrates/vq_vae/archive.py`
  - Detects `VQVP` decoder-section sentinel.
  - Parses in-archive seed envelope.
  - Derives a bounded fp16 codebook tensor from deterministic seed bytes.
  - Injects the derived codebook into the decoder-only state_dict before
    returning `VqVaeArchive`.

- `experiments/train_substrate_vq_vae.py`
  - Vendors `src/tac/procedural_codebook_generator/` into emitted submission
    runtime trees, so procedural archives remain self-contained at inflate time.

- `src/tac/substrates/vq_vae/tests/test_procedural_variant.py`
  - Replaces brittle brotli/pickle section-length assumptions with byte-section
    invariants.
  - Adds a direct parse test for composed procedural archives.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/substrates/vq_vae/tests/test_procedural_variant.py
```

Result: `22 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/substrates/vq_vae/tests/test_vq_vae_roundtrip.py
```

Result: `15 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/local_pre_deploy_check.py --strict --trainer experiments/train_substrate_vq_vae.py --recipe substrate_vq_vae_k_sweep_modal_a10g_diagnostic_dispatch
```

Result: `ALL 9 CHECKS PASSED. Safe to dispatch.`

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from pathlib import Path
from experiments.train_substrate_vq_vae import _write_runtime
out = Path('.omx/tmp/vq_vae_runtime_vendor_smoke_20260521T050252Z/submission')
_write_runtime(out)
assert (out / 'src/tac/procedural_codebook_generator/seed_derived_codebook.py').is_file()
PY
```

Result: `runtime_vendor_smoke_ok`.

Review tracker policy checks passed for all touched `.py` files with no
`REVIEW_GATE_OVERRIDE`.

## Indices-blob directive status

The 192-byte `indices_blob` procedural variant is not claimed as complete by
this fix. The current repair makes the existing procedural codebook surface
inflate-faithful so a later indices-blob design can be evaluated honestly.

Adversarial classification for the queued indices work:

- Pure `REMOVAL` remains refused because the matrix classified the
  `indices_blob` as score-affecting.
- Pure equation #26 replacement is not yet proven because `indices_blob` stores
  per-pair decoder addresses, not a learned codebook/LUT.
- The safer next design is likely
  `RESIDUAL-CORRECTION-DOWNSTREAM`: procedural predictor plus residual stream,
  routed through `procedural_predictor_plus_residual_correction_savings_v1`.

## 6-hook wire-in declaration

- Hook #1 sensitivity-map: N/A for this parser/runtime fix.
- Hook #2 Pareto constraint: ACTIVE via future codebook/indices byte-vs-distortion
  comparison.
- Hook #3 bit-allocator: ACTIVE once the indices residual stream exists.
- Hook #4 cathedral autopilot dispatch: ACTIVE; this fix restores runtime
  consumption truth for downstream VQ-VAE procedural candidates.
- Hook #5 continual-learning posterior: deferred until a paired empirical
  anchor exists.
- Hook #6 probe-disambiguator: ACTIVE in the next indices-blob design; compare
  pure replacement vs residual-correction vs refusal.
