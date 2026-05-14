# HNeRV HLM2 Codec-Specific Repack Probe - 2026-05-14

## Scope

Target: `PR106-R2-HDM4-HLM2-XMEMBER` exact-CUDA anchor.

Archive:
`experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip`

Archive SHA-256:
`2c6e5f8d71f687227a28a9a378dc5edfc3215b762015042203b6bf58bfee9378`

Score axis: `[contest-CUDA]`

Known exact-CUDA score:
`0.20637231876787215`

## Probe Results

### Generic brotli repack route

Command:

```bash
.venv/bin/python tools/build_hnerv_lowlevel_repack_candidate.py \
  --source-archive experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip \
  --source-label PR106-R2-HDM4-HLM2-XMEMBER \
  --output-dir experiments/results/pr106_r2_hdm4_hlm2_brotli_repack_probe_20260514_codex \
  --json-out experiments/results/pr106_r2_hdm4_hlm2_brotli_repack_probe_20260514_codex/manifest.json \
  --jobs 8
```

Verdict: `ready_for_archive_preflight=false`.

Blockers:

- `decoder_packed_brotli` is an `HDM4 decoder codec`, not a brotli stream.
- `latents_and_sidecar_brotli` is an `HLM2 fixed-latent codec`, not a brotli stream.
- No rate-positive generic brotli section recode exists for this anchor.

Engineering change landed with this probe: `tac.hnerv_lowlevel_packer` now handles
wrapper-style `inner_*` scorecard sections for PR106 sidecar packets and reports
codec-specific routing blockers instead of misleading generic brotli failures.

### Decoder structural recode / HDM5 search

Command:

```bash
.venv/bin/python tools/profile_hnerv_decoder_structural_recode.py \
  --source-archive experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip \
  --source-label PR106-R2-HDM4-HLM2-XMEMBER \
  --include-hdm5-search \
  --hdm5-max-parts 8 \
  --hdm5-workers 8 \
  --hdm5-top-k 16 \
  --json-out experiments/results/pr106_r2_hdm4_hlm2_decoder_recode_profile_20260514_codex/profile.json
```

Verdict: current implemented HDM search does not beat HDM4.

- Source decoder section codec: `hdm4_q_brotli_split`
- Source decoder section bytes: `169990`
- Best implemented variant: `hdm4_q_brotli_split_fixed_recipe_dp4_plus_raw_scales`
- Best implemented byte delta: `0`
- Best HDM5 self-describing candidate: `170024` bytes, `+34` bytes vs HDM4
- Fixed-recipe HDM5 projection: `169990` bytes, `0` bytes vs HDM4

The entropy model still shows theoretical decoder headroom:

- Per-tensor zero-order q entropy plus raw scales: `167682` bytes (`-2308` vs source)
- Per-tensor previous-symbol entropy plus raw scales: `156562` bytes (`-13428` vs source)

The existing range/Huffman implementation does not realize that floor because
model overhead dominates. Next decoder work should be an HDM6-style
runtime-specialized context model, not another generic brotli repack.

Expanded partition check:

```bash
.venv/bin/python tools/profile_hnerv_decoder_structural_recode.py \
  --source-archive experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip \
  --source-label PR106-R2-HDM4-HLM2-XMEMBER \
  --include-hdm5-search \
  --hdm5-max-parts 28 \
  --hdm5-workers 8 \
  --hdm5-top-k 32 \
  --json-out experiments/results/pr106_r2_hdm4_hlm2_decoder_recode_profile_20260514_codex/profile_hdm5_parts28.json
```

Result: no byte-positive split was found even with up to `28` parts.
The best self-describing candidate remains `170024` bytes (`+34`), and the
best fixed-recipe projection remains `169990` bytes (`0` vs HDM4).

### Sidecar recode

Command:

```bash
.venv/bin/python tools/profile_pr106_latent_sidecar_recode.py \
  --sidecar-archive experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip \
  --member-name x \
  --json-out experiments/results/pr106_r2_hdm4_hlm2_sidecar_recode_profile_20260514_codex/profile.json \
  --md-out experiments/results/pr106_r2_hdm4_hlm2_sidecar_recode_profile_20260514_codex/profile.md
```

Verdict: current PR101 ranked sidecar is already the best implemented sidecar
grammar for this archive.

- Current charged sidecar bytes: `533`
- Best runtime-implemented row: `pr101_ranked_no_op_sidecar_format_0x02`
- Byte delta vs current: `0`
- Next closest non-implemented raw vocab bitpack: `539` bytes (`+6`)

## Routing

Do not dispatch a generic brotli repack of this HLM2 anchor.

Next byte-closed route:

1. HDM6 decoder-context candidate: runtime-specialized low-overhead context
   coder targeting the measured `156562-167682` byte decoder floor band.
2. Keep HLM2 fixed-latent section as the active latent codec unless a new HLM3
   proof beats it with raw-roundtrip and runtime-consumption evidence.
3. Keep PR101 ranked sidecar unless a runtime-implemented grammar beats `533`
   charged bytes with PacketIR consumed-byte proof and same-runtime parity.

No score claim is made by this probe.
