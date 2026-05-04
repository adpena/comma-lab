# Public PR91 HPAC Hybrid Intake - 2026-05-04

Evidence grade: external static profile plus empirical local prefix smokes.
No score claim here. No remote GPU dispatch was performed.

## Source

- PR: https://github.com/commaai/comma_video_compression_challenge/pull/91
- Title: `Hpac coder hybrid`
- Author: `ottokunkel`
- Head: `ottokunkel:hpac-coder-hybrid`
- Head SHA: `77f958d24e55980d95e01e3e9767b5a94320ed43`
- Submitted archive: `experiments/results/public_pr91_intake_20260504_worker/archive.zip`
- Archive bytes: `222404`
- Archive SHA-256: `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- External PR text reports PoseNet `0.00018940`, SegNet `0.00057185`, exact score `0.24879480490416128`; this ledger does not promote or verify that score.

## Archive Anatomy

`archive.zip` has one stored member named `x`.

- ZIP member `x`: `222304` bytes, SHA-256 `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- Bundle format: PR85/PR91 v5 24-bit length header, fixed `bias=223` and `region=273`
- Header bytes: `24`

Segment layout:

| segment | bytes | SHA-256 | note |
| --- | ---: | --- | --- |
| `mask` | `145087` | `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc` | `HPM1` |
| `model` | `57074` | `c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc` | identical to PR85 |
| `pose` | `1487` | `2b6e03e02fb17f662e3f3f9ce47daf2fc33bf1f76c96e34ffac005f1c5c4054c` | identical to PR85 |
| `post` | `1400` | `c3dc88e0fb5a1e48aec49eeacefeea20685e3c02598933deb1ff22dc09892575` | identical to PR85 |
| `shift` | `226` | `48a6ab5972eb594112c5e9c8ef2a35a757a3df818d489c893e30772f148e4dc4` | identical to PR85 |
| `frac` | `106` | `a5039f64850da646535df761447cecd580703d1cc0c56040006ce1a50bc9a992` | identical to PR85 |
| `frac2` | `149` | `3a01c7546d89d9d3ca846da37a20add7d456421e9211a6339850d946b856026b` | identical to PR85 |
| `frac3` | `154` | `d725b01b604566c5a76a06b821c644c8d5d5bb9a57ab0597d3bdabc8389858b5` | identical to PR85 |
| `bias` | `223` | `8910af036b9fa9f9bee2dbf241dede5afc9720ff9e7e5bcef784a7a3b3e4309b` | identical to PR85 |
| `region` | `273` | `4a7c16094fab281c30e27621a58577c095c4b1153c232518c9adbaba2a2b2392` | identical to PR85 |
| `randmulti` | `16101` | `c624372e60a0851c4c427dc333a60dcc5d6657ba8ed56951612e7c9d7be7629f` | identical to PR85 |

The PR91 byte delta against PR85 is exactly the mask segment: PR85 `236328` bytes to PR91 `222404` bytes, delta `-13924`; PR85 mask `159011` bytes to PR91 HPM1 mask `145087` bytes.

## HPM1 Contract

The `mask` segment is:

- Magic: `HPM1`
- Header bytes: `48`
- `N=600`, `H=384`, `W=512`, `P=32`, `delta=2`, `ch=64`
- `use_spm=1`, `hpac_d_film=8`
- `tokens_len=116796`, uint32-aligned, `29199` words
- `tokens_sha256=541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- `hpac_len=28243`
- `hpac_ppmd_sha256=de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`
- `ppmd_order=4`
- `tail_bytes=0`

That HPAC model SHA matches the known PR86 `hpac.pt.ppmd`; PR91 appears to reuse PR86's HPAC model and supplies a new token stream for the PR85 masks.

## Code Differences

PR91 changed four files:

- `submissions/hpac_coder_hybrid/inflate.py`: PR85-like QZS3 inflater plus `HPM1` branch.
- `submissions/hpac_coder_hybrid/inflate.sh`: same entrypoint shape as PR85.
- `submissions/hpac_coder_hybrid/pr86_hpac.py`: copied PR86 HPAC inflate-side decoder.
- `submissions/hpac_coder_hybrid/range_mask_codec.cpp`: modified PR85 QMA9 range codec.

Runtime dispatch order in `inflate.py` is `HPM1` -> `QMA6/QMA7/QMA8/QMA9` -> Brotli OBU. There is no `try/except` fallback around HPAC entropy decode. If `HPM1` decode fails, inflate aborts; it does not fall back to old PR85 at runtime. The PR comment's "falls back to old pr when compressor fails" is therefore treated as builder-side/external context, not a submitted runtime guarantee.

PR91 `pyproject.toml` declares `brotli` but not `constriction` or `pyppmd`, while `pr86_hpac.py` imports both. The cloned `uv.lock` did not contain `constriction` or `pyppmd`.

## Local Smokes

- `pr91_hpm1_frame0_decode_smoke.json`: empirical CPU prefix decode failed with `AssertionError: Tried to decode from compressed data that is invalid for the employed entropy model.`
- `pr91_range_codec_pr85_qma9_decode_smoke.json`: PR91 `range_mask_codec.cpp` compiled and decoded the PR85 QMA9 segment to `117964800` bytes with class range `0..4`, but the decoded SHA did not match the recorded PR85 token source SHA. Do not use the PR91 range codec as the old-PR fallback decoder until parity is repaired.

## Transfer Decisions

1. Do not dispatch PR91-derived HPAC locally until HPM1 full-stream decode and decode->reencode parity pass under pinned `constriction`, `pyppmd`, and `torch`.
2. Port `HPM1` only as a typed mask segment behind fail-closed preflight: header parse, uint32 alignment, HPAC model SHA, dependency versions, prefix decode, full token SHA, and runtime output parity.
3. Keep our existing PR85 QMA9 decoder as fallback. Do not replace it with PR91 `range_mask_codec.cpp` until PR85 token-source parity is repaired.
4. To beat PR91 directly, reduce mask bytes below `145087` or stack an independent non-mask reduction on the PR85-identical model/pose/post side channels. PR91 itself provides no non-mask byte improvement.

## Artifacts

- `experiments/results/public_pr91_intake_20260504_worker/pr91_archive_anatomy.json`
- `experiments/results/public_pr91_intake_20260504_worker/pr91_vs_pr85_segment_diff.json`
- `experiments/results/public_pr91_intake_20260504_worker/pr91_transfer_decisions.json`
- `experiments/results/public_pr91_intake_20260504_worker/pr91_hpm1_frame0_decode_smoke.json`
- `experiments/results/public_pr91_intake_20260504_worker/pr91_range_codec_pr85_qma9_decode_smoke.json`
- `experiments/results/public_pr91_intake_20260504_worker/pr91.patch`
- `experiments/results/public_pr91_intake_20260504_worker/pr91_api.json`
- `experiments/results/public_pr91_intake_20260504_worker/pr91_files_api.json`
