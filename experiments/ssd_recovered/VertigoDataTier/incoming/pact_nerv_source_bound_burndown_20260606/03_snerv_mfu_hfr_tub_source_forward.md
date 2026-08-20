# 03 — SNeRV official MFU/HFR/TUB: where proof is still metadata-only

## Finding

The current artifact says:

```json
"official_mfu_hfr_source_fixture_forward_parity_passed": true,
"official_mfu_hfr_tub_forward_parity_passed": false,
"full_tub_source_forward_parity_proven": false
```

Top blockers include:

```text
snerv_official_trained_checkpoint_state_dict_not_loaded
snerv_official_tub_encoder_decoder_weights_not_loaded
snerv_official_snerv_t_full_tub_source_forward_replay_missing
snerv_official_pytorch_wavelets_runtime_dependency_missing
snerv_official_tub_portable_temporal_encoder_weight_mapping_missing
snerv_official_tub_portable_output2_decoder_weight_mapping_missing
snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing
```

MFU and HFR have primitive fixture parity, but each row still says `full_stack_source_forward_parity_proven=false`. The receiver runtime rows say `receiver_runtime_decode_proven=true`, but `receiver_source_forward_replay_bound=false`. That is the metadata-only gap: the receiver can run primitives, but the full trained official graph is not bound to the payload.

## Shortest real closure

Patch target:

```text
src/tac/analysis/snerv_official_tub_source_forward_replay.py
tools/build_snerv_official_tub_lf_hf_replacement_authority_gate.py
src/tac/substrates/snerv_inverse_steg_carrier/tests/test_official_tub_full_source_forward.py
```

Implementation:

1. Load one official `snerv_t` checkpoint/state dict fixture.
2. Extract:
   - `self.encoder[1]`
   - `self.encoder[2]`
   - `self.decoder[self.decoder_len-1]`
3. Map these into portable payload keys:
   - `tub.encoder1.*`
   - `tub.encoder2.*`
   - `tub.output2_decoder.*`
4. Run official Torch `model/snerv_t.py` full TUB source forward on a deterministic LF triplet.
5. Run portable NumPy/MLX receiver path from the encoded payload.
6. Compare:
   - full output pixels,
   - output2 fusion tensor,
   - optional scorer atoms: last-frame SegNet logits, PoseNet YUV6 pair tensor.
7. Only then set:
   - `full_tub_source_forward_parity_proven=true`
   - `receiver_source_forward_replay_bound=true`

## Failing test

`tests/test_snerv_tub_full_source_forward.py` should assert no `unmapped_temporal_encoder` or `unmapped_output2_decoder` in `official_weight_keys`.

Current artifact fails this by construction.

## Passing target

```bash
pytest -q src/tac/substrates/snerv_inverse_steg_carrier/tests/test_official_tub_full_source_forward.py
python tools/build_snerv_official_tub_lf_hf_replacement_authority_gate.py --require-full-tub-source-forward
```

## Do not run long SNeRV before this

A local receiver-safe adapter is not official SNeRV unless it is source-forward bound. Rate experiments on LF/HF are still useful as codec probes, but not as official SNeRV long-run authority.
