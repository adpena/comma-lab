# Codex Findings: Real PR101 Grouped Runtime Campaign

UTC: 2026-06-01T17:23:57Z
Author: Codex
Axis: `[macOS-CPU byte-profile / receiver-runtime-tree-materialization-only]`

## Campaign

Ran the new PR101 grouped grammar materializer on the real public PR101 archive, not a synthetic state dict.

Durable artifact root:

```text
/Volumes/VertigoDataTier/pact/pr101_real_grouped_runtime_campaign_20260601T172357Z
```

Source archive:

```text
experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/archive.zip
source_archive_bytes=178258
source_archive_sha256=b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e
inner_member_bytes=178158
inner_member_sha256=5f1948f9572e65f71c614d2ff15764ee416522e25cb1b06c8b1299c1306e8aaf
decoder_blob_bytes=162164
decoder_blob_sha256=836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6
```

## Result

The grouped packet solver found no real decoder-byte win on the actual PR101 decoder:

```text
selected isolated bytes=162229; current isolated bytes=162260; floor=159877.3; status=entropy_saturated
grouped selected bytes=162164; current grouped bytes=162164; saved=0; runtime=stock_pr101_runtime
```

Campaign summary:

```text
candidate_archive_bytes=178262
candidate_archive_sha256=e0c81623aaa4ebe08817fadbe2847aed460bafaaf7e75446f1c3842cc5b268a6
grouped_delta_bytes_vs_current_stock=0
grouped_saved_bytes_vs_current_stock=0
runtime_blockers=full_frame_inflate_parity_missing,contest_cpu_cuda_exact_eval_not_executed
archive_blockers=full_frame_inflate_parity_missing,contest_cpu_cuda_exact_eval_not_executed,receiver_runtime_source_not_emitted
```

The candidate archive is 4 bytes larger because this run intentionally used the `u32_decoder_len_adapter` layout. Since the real selected grouped decoder is stock-runtime-compatible and byte-identical in length, the u32 adapter is unnecessary for this exact real-PR101 branch.

## Verdict

Real PR101/fec6 grouped decoder grammar is saturated. The new materializer path is correct infrastructure for future substrates and non-stock grouped layouts, but it does not lower the current PR101 decoder rate by itself.

Planner consequence: keep this branch as an exact byte-closure/runtime-adapter capability, not as a score-lowering candidate for the current PR101/fec6 substrate. Grammar remains high-EV for unsaturated substrates such as Z8/detail-coefficient and future train/export/archive lanes.
