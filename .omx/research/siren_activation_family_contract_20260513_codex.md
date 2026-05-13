# SIREN activation-family archive contract (2026-05-13)

lane_id: `lane_substrate_siren_20260512`
scope: local SIREN/INR activation-family surfaces only
score_claim: `false`
promotion_eligible: `false`
gpu_dispatch: `false`
research_only: `true`

## Landing

This landing adds a typed activation-family interface for the full SIREN
substrate without changing the default dispatch contract. The trainer still
defaults to `naked_siren_replacement` plus `--activation-family siren`.

New comparable modes:

- `siren`: canonical sine activation.
- `finer`: FINER-style variable-periodic sine probe.
- `wire`: WIRE-style windowed/Gabor periodic probe.
- `bacon`: BACON-style band-limited sine schedule probe.

These non-default modes are activation probes under the SRV1 archive contract,
not claims of full-paper FINER/WIRE/BACON architecture parity. The selected
family is serialized in SRV1 metadata, parsed by `parse_archive`, and consumed
by inflate via `SirenConfig`.

## Byte-closure contract

- Header remains SRV1-compatible.
- New archives write `activation_family`, `wire_scale`, and
  `bacon_bandwidth_scale` into the metadata blob.
- Archives missing those metadata fields parse as `activation_family=siren`,
  preserving existing naked SIREN behavior.
- Unsupported activation families fail closed at config/archive pack time.
- Runtime state dict keys remain stable because the MLP layer contract still
  stores trainable tensors under `hidden.*.linear.*` and `output_layer.*`.
- The emitted contest runtime vendors `activation_family.py` alongside
  `architecture.py`, `archive.py`, and `inflate.py`.

## Six-hook wire-in

1. Sensitivity-map contribution: N/A for this landing; no empirical anchor or
   scorer sensitivity map was generated.
2. Pareto constraint: declared only as a comparable local activation axis;
   no Pareto frontier row is promoted without exact archive evidence.
3. Bit-allocator hook: N/A; no quantization or byte allocator changed.
4. Cathedral autopilot dispatch hook: fail-closed. The activation family is
   trainer/manifest-visible, but no provider dispatch was created.
5. Continual-learning posterior update: N/A; no score anchor.
6. Probe-disambiguator: the typed activation family is the local
   disambiguation interface for SIREN vs FINER/WIRE/BACON-style modes.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/substrates/siren/tests src/tac/tests/test_siren_substrate_readiness.py -q
```

Result: `50 passed in 3.19s`.

```bash
.venv/bin/python tools/audit_siren_substrate_readiness.py --json --fail-if-not-ready
```

Result: passed. The readiness payload remains
`ready_for_remote_dispatch=false`, `ready_for_exact_eval_dispatch=false`,
`score_claim=false`, and `promotion_eligible=false`.

```bash
.venv/bin/python experiments/train_substrate_siren.py --video-path upstream/videos/0.mkv --output-dir .omx/tmp/siren_activation_family_finer_smoke_20260513_codex --epochs 1 --device cpu --smoke --activation-family finer --skip-archive-build --skip-auth-eval
```

Result: CPU smoke only; no scorer load and no score claim.

```bash
.venv/bin/python -m ruff check src/tac/substrates/siren/activation_family.py src/tac/substrates/siren/architecture.py src/tac/substrates/siren/archive.py src/tac/substrates/siren/inflate.py src/tac/substrates/siren/tests/test_activation_family.py src/tac/substrates/siren/tests/test_dispatch_contract.py experiments/train_substrate_siren.py
```

Result: passed.

```bash
git diff --check -- src/tac/substrates/siren experiments/train_substrate_siren.py
```

Result: passed.

## Remaining blockers

1. No exact SIREN-family replacement archive has been scored on
   `[contest-CUDA]` or `[contest-CPU]`.
2. The current full substrate still has no learned per-pair/content-adaptive
   temporal embedding; scalar `t` remains the main structural risk.
3. No post-training quantization or rate-gradient proxy has landed for these
   activation modes.
4. No scorer-aware atom dispatch should be inferred from this landing; it only
   creates the byte-closed local comparison interface.
