# PR106 Stacked Archive SHA Custody

Date: 2026-05-06
Agent: Codex
Evidence grade: empirical

## Finding

`experiments/build_pr106_stacked.py` already validates that each sister archive
embeds the exact PR106 anchor bytes before extracting sidechannel sections, but
the emitted `build_metadata.json` mostly recorded archive paths and byte counts.
That left stacked-candidate provenance weaker than the sister-lane reducers:
reviewers could not verify from the metadata alone that the named input and
output archive files matched the files used at compose time.

## Change

The stacked builder now emits:

- `manifest_schema = pr106_stacked_build_metadata_v2`
- `pr106_archive_sha256`
- `archive_sha256`
- `input_archives.{pr106,latent,yshift,lrl1,wavelet}.{path,bytes,sha256}`
- `output_archive.{path,bytes,sha256}`

The older flat path/byte fields remain in place for compatibility.

## Guard

`src/tac/tests/test_pr106_stacked.py::test_builder_metadata_records_archive_hash_custody`
runs the CLI in passthrough mode and asserts that the output metadata hashes
match the actual anchor and output archive bytes.

## Dispatch Status

No exact eval or remote GPU dispatch was attempted. This is a custody and DX
hardening change only. `ready_for_exact_eval_dispatch` remains false until
sister sidechannels have exact CUDA evidence and the composed archive passes
the existing dispatch blockers.
