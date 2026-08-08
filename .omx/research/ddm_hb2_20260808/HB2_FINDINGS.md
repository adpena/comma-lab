# HB2 findings - HPAC self-compress pack repair

## Verdict

MEASURED [byte-only scorer-free, macOS CPU]: tq1c stage 3 and stage 4 now pass on the real
`hpac_selfcompress_e60.pt` checkpoint. Stage 3 writes an exact self-compressed model blob;
stage 4 encode/decode round-trips all 600 tq1c label frames with `verified_exact=true`.

No scorer slot was used. No exact contest score was measured. The own-vehicle frontier is
unchanged: S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory].

## Reproduction

Standalone rerun of the driver stage-3 command reproduced the fail-closed failure exactly:

- checkpoint: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/tq1c/hpac_selfcompress_e60.pt`
- command surface: `pack_hpac_self_compress.py --channels 64 --patch 64 --delta 2 --frame-dim 8 --weight-bound 127 --activation-bound 127 --weight-exponent-min -6 --device cpu`
- rc: 1
- error: `RuntimeError: self-compressed round trip changed logits by 0.25`

## Mechanism

The mechanism was a deployable-range mismatch, not the 0-bit or 2-bit serializer edges.

Per-module, per-channel compare on the real checkpoint found exactly one differing channel:
`conv_a` channel 30, deployed bit depth 8. The self-compression-enabled source allowed a
deployed weight code of `-128` from the 8-bit two's-complement range. The plain restored
consumer model uses `weight_bound=127`, so its runtime `codes()` clamps that value to `-127`.
That one integer-weight difference produced the observed max logit delta `0.25` and 480
nonzero logit differences on the seeded packer check input. Fixed parameters had zero diffs.

After the fix, the same diagnostic reports `raw_model_bytes=18838`, `max_logit_diff=0.0`,
`nonzero_logit_diffs=0`, and zero differing weight channels.

## Fix

Repro repo:

- repo: `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`
- start HEAD: `2f94596bb0136d342254022a5c9584756eae0468`
- fix commit: `e34f31bc4969042c0051ac81aa3c56884419a231`
- patch mirror: `.omx/research/ddm_hb2_20260808/0001-Fix-HPAC-self-compress-deploy-bounds.patch`
- patch SHA-256: `cffb093faa22666cfb8d596a81f45c8e2f1462650cdfa91775b8e3b4c30e38c6`

Changed behavior:

- `hpac_self_compress._quantized_weight` now clamps the learned bit-depth range by the
  module's declared `weight_bound`, so the training/deployed source and plain consumer agree.
- `pack_hpac_self_compress.py` serializes only the intersection of the bit-depth signed range
  and the deployable `weight_bound` range.
- both pack-time and deployed `IHS1` deserializers fail closed if decoded weights exceed the
  consumer's deployable range.
- regression test added: an 8-bit channel with a raw `-128` weight serializes/restores as the
  deployable `-127` and matches source/consumer weights exactly.

Borrowed substrate accounting: PR130 supplied the IntegerHPAC architecture, self-compress
training/packing shape, `IHS1` model format, and arithmetic token codec. This arm changed only
the deploy-bound consistency repair, the regression test, and the tq1c rerun on our label
payload. No PR130 score or external HPAC byte number is claimed as ours.

## Rerun receipts

Stage 3 pack, same driver argv:

- rc: 0
- blob: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/tq1c/hpac.bin.xz`
- report: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/reports/tq1c_hpac_pack.json`
- raw model bytes: 18,838
- compressed model bytes: 14,116
- metadata bytes: 259
- verified exact: true
- max logit diff: 0.0
- model SHA-256: `6c44216e8f79bd7d04e998b898d5bf0dc16bae6e3763f8bc19ce4ec8ebdabb40`
- report SHA-256: `d4b708e70753ce740f63a49d0886120b1e69595bd349469630f958a93a78b3b6`

Packed model bytes vs the stage-2 estimate: 14,116 actual vs 18,753 estimated, 4,637 bytes
lower than estimate.

Stage 4 encode, same driver argv:

- rc: 0
- tokens: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/tq1c/tokens.bin`
- report: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/reports/tq1c_tokens_encode.json`
- token bytes: 97,928
- token bpp: 0.006641167534722222
- ideal bpp: 0.006501749009840285
- logit hash encode: `e63bd314afe44da45fc9ade6b3e58cc8855d242e7da1f6a64169f82cb43e827a`
- tokens SHA-256: `fc2e4d30df701877cb81d2aeeefa3079e5a1a4d1c3c24db627d7932ab780559f`
- report SHA-256: `d49351da957174af34c96d1e9920cbb2230ceed1eb51f5b8f4719f1cbf79caac`

Stage 4 decode, same driver argv plus `--require-exact`:

- rc: 0
- report: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/reports/tq1c_tokens_decode.json`
- verified exact: true
- logit hash decode: `e63bd314afe44da45fc9ade6b3e58cc8855d242e7da1f6a64169f82cb43e827a`
- raw token SHA-256: `3480e4af913f7f706c907a7ab2322b1a973728749d55e5f916beef6b4cde5db1`
- report SHA-256: `2e8131ce49e9418e328831a39893250b1a10b3830a0508fa9735011de4979b50`

Actual model plus token bytes: 112,044 B. This is 3,374 B lower than the stage-2 joint estimate
of 115,418 B, with exact decode equality attached.

## Recall Evidence

Sources searched before adjudicating/fixing:

- memory registry: `rg -n "ddm_hb2|hb2|20260808|common_contract|DDM|ddm" /Users/adpena/.codex/memories/MEMORY.md`
- memory registry: `rg -n "pr130|self-compress|self_compress|hpac|batch_shape|instrument|round trip|round-trip" /Users/adpena/.codex/memories/MEMORY.md`
- repo corpus: `rg -n "HPAC|hpac|self[-_ ]compress|PR130|pr130|pack_hpac_self|bit_depth|0-bit|2-bit|round trip|round-trip" .omx/research .omx/state reports docs experiments src tools ...`
- canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`, searched for HPAC/self-compress/PR130/bit-depth/integer terms
- DAG/index surfaces: `.omx/research/CANONICAL_RESEARCH_INDEX*` and `.omx/research/sub015_DAG_*`

Found beyond the charter: canonical equation `ddm_hb1_semantic_label_incumbent_transfer_v1`
requires target-payload training plus exact decode equality before adopting HPAC/SMEVR bytes on
our tq1c/GT label payloads. That changed the plan only by tightening the receipt standard:
the fix had to end with real tq1c pack plus encode/decode exact equality, not just a unit test.

Scoped negative: I did not find an existing HB2 packer fix or prior diagnosis of this `-128`
deploy-bound mismatch in the searched memory/repo/index/canonical-equation scopes.

## Boundaries

- CPU only, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `VECLIB_MAXIMUM_THREADS=4`.
- No Metal, MPS, CUDA, or scorer slot used.
- Did not touch `checkpoints/gt/`.
- Did not kill or signal driver pid 9316.
- Did not modify the hb1 driver script while it runs.
- Did not edit `upstream/` or Pact protected files.
- The gt arm will inherit the fix automatically at its future stage-3/stage-4 process boundary
  because those driver steps invoke the shared repro repo scripts after training. The live gt
  stage-2 process was not touched.

Follow-ons disposition: tq1c stage 3 and stage 4 were FIRED and completed; gt inheritance is
QUEUED-BY-EXISTING-DRIVER-FIRE-ORDER, not a new orphan follow-on.
