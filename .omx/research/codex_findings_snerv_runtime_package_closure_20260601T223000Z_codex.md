# Codex Findings — SNeRV SNAR1 Runtime Package Closure — 2026-06-01T22:30:00Z

## Verdict

SNeRV SNAR1 now has a real executable receiver package surface for tiny full-frame
packets: `archive.zip` + `submission/inflate.sh` + vendored receiver modules +
machine-readable receiver proof. This is receiver-runtime custody only, not score
authority.

Authority flags remain false:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Landed Surfaces

- `src/tac/substrates/snerv_inverse_steg_carrier/inflate.py`
  - scorer-free, torch-free SNAR1 `0.bin` consumer
  - decodes `(n_pairs, 2, 3, H, W)` receiver frames
  - NumPy bilinear camera-size raw writer
- `src/tac/substrates/snerv_inverse_steg_carrier/archive_candidate.py`
  - writes SNeRV contest runtime via canonical deterministic ZIP helper
  - emits shared archive-bound candidate package
  - records false-authority blockers
- `tools/prove_snerv_receiver_archive.py`
  - new `--full-frame-packet`
  - new `--package-dir`
  - package-aware top-level blockers
- `src/tac/substrates/snerv_inverse_steg_carrier/receiver_proof.py`
  - full-frame proof packet mode with `n_pairs`, `frames_per_pair`, `channels`

## SSD Smoke

Command:

```bash
.venv/bin/python tools/prove_snerv_receiver_archive.py \
  --bins 4 \
  --levels 1 \
  --height 16 \
  --width 24 \
  --package-dir /Volumes/VertigoDataTier/pact/snerv_receiver_runtime_package_smoke_codex_20260601T223000Z \
  --out /Volumes/VertigoDataTier/pact/snerv_receiver_runtime_package_smoke_codex_20260601T223000Z/snerv_receiver_archive_proof.json \
  --packet-out /Volumes/VertigoDataTier/pact/snerv_receiver_runtime_package_smoke_codex_20260601T223000Z/snerv_receiver_archive_packet.snar \
  --package-timeout-seconds 120
```

Artifacts:

- `/Volumes/VertigoDataTier/pact/snerv_receiver_runtime_package_smoke_codex_20260601T223000Z/archive.zip`
  - bytes: `27075`
  - sha256: `1d8d0bba77ca9d418008824b689823884a61dfba0baace73dc3b58a835ad2368`
- `/Volumes/VertigoDataTier/pact/snerv_receiver_runtime_package_smoke_codex_20260601T223000Z/0.bin`
  - bytes: `2609`
  - sha256: `5623623b5100a804f5a755cb0e79d38d878e747a5991288739c5e03983659f17`
- `/Volumes/VertigoDataTier/pact/snerv_receiver_runtime_package_smoke_codex_20260601T223000Z/archive_bound_candidate_adapter_package.json`
  - receiver proof passed: `true`
  - receiver output bytes during proof: `6104016`
  - receiver output retained: `false`

Canonical blockers after packaging:

- `snerv_packet_not_full_600_pairs`
- `paired_contest_cpu_cuda_auth_eval_missing`
- `pywavelets_runtime_dependency_not_contest_proven`

## Validation

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_receiver_proof.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_archive_candidate.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_inflate.py -q
```

Result: `10 passed`.

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_receiver_proof.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_archive.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_advisory_archive_replay.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_inflate.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_archive_candidate.py -q
```

Result before final CLI blocker-print adjustment: `22 passed`.

```bash
.venv/bin/python -m ruff check \
  tools/prove_snerv_receiver_archive.py \
  src/tac/substrates/snerv_inverse_steg_carrier/receiver_proof.py \
  src/tac/substrates/snerv_inverse_steg_carrier/inflate.py \
  src/tac/substrates/snerv_inverse_steg_carrier/archive_candidate.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_inflate.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_archive_candidate.py
```

Result: `All checks passed!`

## Next Required Work

1. Add a real advisory/full-run packet export path that writes the trained
   full-600 SNAR1 packet into this archive-bound package builder.
2. Replace or certify the `pywt` receiver dependency. Best next hardening target:
   NumPy db2 periodization inverse/analysis primitives or a contest-runtime import
   proof that PyWavelets is present on the exact replay substrate.
3. Run charged 16-pair/full-600 advisory packages through this archive-bound
   runtime path.
4. Claim lane before any contest CPU/CUDA exact replay dispatch.
5. Only after byte-closed full-600 package plus paired exact replay, compare
   against PR95/PR101 on the same evidence axis.
