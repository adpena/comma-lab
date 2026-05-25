# Codex Findings: Packet Member Merge v1 Portable Materializer

UTC: 2026-05-24T23:34:49Z

## Summary

Landed `packet_member_merge_v1` as a portable, family-agnostic materializer for
merging selected ZIP members into one deterministic receiver-visible packet.
The materializer is deliberately false-authority: it emits a byte-closed
candidate archive and reconstruction proof, but exact readiness remains refused
until a cooperative contest runtime adapter consumes the merged member and
proves source/candidate inflate parity.

## Durable Wiring

- Core materializer: `tac.optimization.family_agnostic_materializers.materialize_packet_member_merge_candidate`
- CLI: `tools/run_family_agnostic_materializer.py --target-kind packet_member_merge_v1`
- Registry: `comma_lab.scheduler.byte_shaving_materializer_registry`
- Queue compiler: `comma_lab.scheduler.byte_shaving_campaign_queue`
- Final-byte context compiler: `comma_lab.scheduler.final_byte_operation_contexts`
- Compiler target metadata: `tac.optimization.byte_shaving_campaign`

## Proof Contract

- Candidate schema: `packet_member_merge_candidate.v1`
- Parser proof kind:
  `packet_member_merge_original_member_reconstruction_receiver_proof.v1`
- Compiled-runtime proof kind:
  `packet_member_merge_runtime_adapter_consumption_proof.v1`
- Merge payload formats:
  `raw_member_payload_v1`,
  `source_zip_compressed_stream_v1`,
  `source_zip_compressed_stream_binary_table_v1`,
  `fixed_order_raw_deflate_sequence_v1`
- Portability contract:
  `family_agnostic_materializer_portability_contract.v1`
- Required runtime modules: Python stdlib `json`, `struct`, `zipfile`, `zlib`
- GPU/MLX/Metal/CUDA required: false

The proof verifies selected-member reconstruction by parsing the merge table,
slicing the concatenated payload, hashing reconstructed original members, and
checking non-selected member payload and ZIP-compressed-stream identity. The
top-level receiver contract stays false until a compiled receiver runtime is
built and consumed; parser-only reconstruction is recorded separately as
`reconstruction_proof_satisfied`.

## Real Archive Smoke

Final artifact:
`experiments/results/packet_member_merge_v1_smoke_20260525T000149Z/candidate.json`

- Source archive:
  `submissions/robust_current/archive_correct.zip`
- Source SHA-256:
  `4dd46fed78ed064bc97c9b3205088e82838c03667394f7936c8ae8d20f9837ab`
- Candidate SHA-256:
  `9baa1208be1fb0c4e62be63e18b20b00e2c9101e16a85721ddbc87db1d0f8b0d`
- Selected members:
  `renderer.bin`, `masks.mkv`, `optimized_poses.pt`
- Merged member:
  `merged.packet`
- Source bytes: `345802`
- Candidate bytes: `345514`
- Saved bytes: `288`
- Selected payload codec: `fixed_order_raw_deflate_sequence_v1`
- Merge table bytes charged in payload: `72`
- Receiver contract satisfied: true
- Runtime adapter ready: true
- Exact readiness authority: still false; no auth eval score claim in this memo
- Real receiver smoke:
  `experiments/results/packet_member_merge_v1_smoke_20260525T000149Z/receiver_smoke_real_file_list.json`
  passed through the lightweight generated receiver and robust renderer runtime.

Adversarial progression on the same robust archive:

- JSON raw merge: `346868` bytes, `-1066` saved.
- Source ZIP compressed streams plus JSON table: `346429` bytes, `-627` saved.
- Source ZIP compressed streams plus uvarint binary table: `345533` bytes,
  `269` saved before wrapper hardening.
- Fixed-order raw-deflate sequence (`DFL1`): `345514` bytes, `288` saved.

Interpretation: the negative first result was an engineering/config bug class,
not a method negative. Carrying decoded payloads and JSON metadata destroyed the
rate gain. Preserving source ZIP deflate streams, eliding per-member offsets and
hashes from charged bytes, and letting raw deflate stream termination delimit
members crossed the rate boundary. The remaining long-term opportunity is a
runtimeless/source-runtime-native packet such as `renderer_payload.bin`/`p` or a
PacketIR cooperative grammar, so the archive uses a contest-runtime-native
unpack path instead of a generated wrapper.

Runtime custody note: the first compiled receiver copied the full
`robust_current` directory when sidecars were allowed and generated multi-GB
runtime artifacts. The compiler now copies code/config only, skips rebuildable
archive/data sidecars, and reconstructs payload members from the candidate
archive. The final runtime tree is under 1 MB. Earlier bloated runtime/raw
outputs were deleted after preserving candidate/proof/smoke JSON.

## Next Integration Notes

1. Promote the DFL1 representation into PacketIR/cooperative receiver grammar
   instead of leaving it as an ad hoc packet dialect.
2. Add a runtimeless/source-runtime-native materializer that targets existing
   `renderer_payload.bin`/`p` unpack paths where the source runtime already
   consumes packed renderer members.
3. Add grouped-search acquisition rows for partial member subsets and
   interactions with header elision/recompression.
4. Add exact-readiness refusal tests requiring parser proof, compiled receiver
   proof, and receiver smoke/full-frame parity before auth dispatch.
