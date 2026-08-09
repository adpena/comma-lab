# OP1R executable reproduction producer

## Objective

Implement one narrow, read-only-on-scientific-inputs Python CLI at
`tools/reproduce_ddm_op1r_measurements.py` that reruns the
three scorer-free OP1R measurements and writes one small atomic JSON receipt.
This closes the provenance gap left by the original one-off `python3 -`
commands. Do not change any scientific input, public/intake checkout, evaluator,
or existing OP1R prose/JSON artifacts.

## Governing inputs

Read `OP1R_REPRODUCTION.json` as the hash-pinned contract. Fail closed before
measurement if any configured input/source/archive hash or required shape/dtype
does not match. The CLI must support `--config`, `--section` with
`target-cache|receiver|xz|all`, `--resume-from`, `--dependency-root`,
`--pycache-prefix`, and `--output`. The output must
be written by temporary sibling plus `fsync` plus atomic rename. Scientific
reductions use no RNG; seed any overwritten Torch constructor state explicitly
and record that it cannot affect the restored model.

## Required sections

1. `target-cache`: reproduce the official-DALI versus local-macOS-AV-like
   segmentation content diff, four-neighbor DALI boundary overlap, per-DALI-class
   counts, PR130-original comparisons, M1 target identity, and the DALI versus
   PR130-original pose reductions exactly as specified in the config.
2. `receiver`: import the pinned PR130 source without modifying it, parse the
   canonical CPR1 archive as `inflate.main` does, load complete n600 model state,
   then set only module `N=1`, decode/render the real first pair on CPU twice in
   fresh process state or an equivalently isolated subprocess, and require the
   configured token/output hashes and repeat equality. Put any scratch raw file
   under `/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/reproducer_scratch`,
   hash it, then success-clean it automatically; never use `/tmp` or local bulk.
3. `xz`: reproduce both complete configured LZMA grids, exact raw-model/token
   parse-back, deterministic stored-ZIP construction metadata, and the configured
   candidate hashes/bytes. This is a sweep: atomically checkpoint after every
   completed row at `--resume-from`, resume without repeating completed rows,
   preserve a per-row size ledger plus its digest, and preserve distinct
   grid-1/grid-2 stage-complete receipts. Fail closed on the configured Python,
   lzma-module, XZ, and liblzma runtime fingerprint before row zero and bind that
   fingerprint into every cursor/stage identity. Decode and compare every grid
   stream to the input. Do not overwrite the retained candidate archive.

## Provenance in the receipt

Record, both top-level and per section where relevant: actual UTC start/end,
the current workspace HEAD, producer path and producer SHA-256, exact argv,
config path/SHA-256, config schema, platform/arch, Python/build, NumPy, Torch,
constriction, lzma module path, linked liblzma/xz version if discoverable,
hardware label supplied by `--hardware-label`, device/axis, thread count and
determinism flags, all source/input/archive hashes and bytes, elapsed seconds,
and pass/fail against every configured expected value. Do not collect or emit a
serial number, hardware UUID, or other host identifier.

The receipt must explicitly say the original one-off command provenance is
`UNDETERMINED`; its own results are a fresh deterministic rerun, not a claim
that this was the historical producer.

## Constraints

- Scorer-free only. Never invoke `upstream/evaluate.py`, SegNet, or PoseNet.
- No network, CUDA, MPS, remote jobs, training, or full n600 video inflate.
- Writes are limited to the configured final output, SSD scratch, XZ cursor and
  distinct stage receipts, receiver worker receipts, and exclusive-lock files.
  Scientific inputs and the public/intake checkout remain read-only.
- Hold fail-closed exclusive locks over the receiver scratch namespace and the
  entire XZ resume namespace; concurrent writers are an error.
- Do not import or copy mutable code into the pact tree; source execution is
  hash-gated and read-only.
- Preserve all unrelated dirty work.
- Keep output machine-readable; diagnostics go to stderr.

## Acceptance

Run from `/Users/adpena/Projects/pact`:

```bash
python3 tools/reproduce_ddm_op1r_measurements.py \
  --config .omx/research/ddm_op1r_20260809/OP1R_REPRODUCTION.json \
  --section all \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/reproducer_state_v3/state.json \
  --dependency-root /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/runtime_py314 \
  --pycache-prefix /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/reproducer_pycache_v3 \
  --hardware-label 'MacBook Pro Mac17,6; Apple M5 Max; 18 cores; 128 GB' \
  --output .omx/research/ddm_op1r_20260809/OP1R_REPRODUCTION_RECEIPT.json
python3 -m json.tool .omx/research/ddm_op1r_20260809/OP1R_REPRODUCTION_RECEIPT.json >/dev/null
```

The first command must exit 0, all three section verdicts must be `PASS`, every
configured expected hash/count/reduction must match, and the success path must
leave no receiver scratch raw file.
