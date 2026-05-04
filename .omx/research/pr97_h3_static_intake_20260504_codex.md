# PR97 H3 Static Intake And Byte Opportunity Ledger - 2026-05-04

Scope: PR97 `vibe_coder_final_boss (0.23)` no-dispatch deconstruction only.
No remote GPU job was launched in this turn. Existing exact T4 replay
`exact_eval_public_pr97_vibe_coder_final_boss_t4_20260504T0923Z` remains the
score truth if/when harvested.

## Source Custody

- Archive inspected:
  `experiments/results/leaderboard_intel_20260504_codex/pr97_archive.zip`
- Archive bytes: `197160`
- Archive SHA-256:
  `6785a84879d3e3395bbf990b980fe32182fca7255c5b8559dcdaac9da7516642`
- Runtime inspected:
  `experiments/results/leaderboard_intel_20260504_codex/pr97_runtime/`
- Writeup inspected:
  `experiments/results/leaderboard_intel_20260504_codex/pr97_source_tmp/WRITEUP.md`
- Static preflight: passed, no blockers, charged member allowlist passed.
- Evidence grade in this ledger: `external_archive_byte_intake_only_until_exact_cuda_replay`.

The archive is a strict one-member ZIP with member `p`. The member is stored,
not deflated. ZIP envelope overhead is exactly `100` bytes.

## Payload Anatomy

Machine-readable profile:
`experiments/results/pr97_h3_intake_20260504_codex/profile_pr97_h3_intake.json`

Markdown profile:
`experiments/results/pr97_h3_intake_20260504_codex/profile_pr97_h3_intake.md`

Payload `p` is `197060` bytes, SHA-256
`3ff68e3b463498a6e054cd69ff286e5098333a24c4e187c70acd585af794a32a`.

Length-prefixed split:

| Part | Bytes | SHA-256 |
|---|---:|---|
| mask | 135120 | `bdf2f7a54a245e8a11d5b1ca74e7a7099c4754b3f5f8faa57e71418208031642` |
| pose | 2310 | `c7a62b5a8feaa43ad2aa42dafec98774adea78a82c31cebb9530326302993f54` |
| model | 57238 | `0ac6ff812fe3cda523960d62ddca11e96703deb44b3c212110b16923322fc472` |
| sidecar | 2376 | `b8bb46e5f4e5966489ff7ea12f2b814c78548c98c886fab25818b8c71ad93045` |

Subformat parse:

- Mask: H3 tiled range-mask stream, `22` chunks, `92` chunk-header bytes,
  largest chunk `22785` bytes, smallest chunk `39` bytes.
- Pose: brotli-compressed per-dim packed pose, raw `2612` bytes, `600 x 6`,
  bits per dim `[14, 4, 4, 4, 4, 4]`.
- Model: brotli-compressed flat-FP4 H3 model, raw `61164` bytes, `102` schema
  entries, `79544` FP4 params, `8172` FP16 params.
- Sidecar: xz-compressed `BPGD` sidecar, raw `3482` bytes, `455` pair records.
  Parsed counts: x2 `272` pairs / `272` patches, CMA-ES `33` pairs / `60`
  patches, pattern `69` pairs / `138` patches, pose deltas `109` pairs, warps
  `227` pairs.

The writeup's high-level byte table matches the real archive. The exact
sidecar parse clarifies that the shipped final X2 count is one X2 patch per
X2 pair in this archive, not the larger average patch count discussed in an
earlier explanatory table.

## Runtime And Compliance Notes

Static runtime preflight passed. Runtime files are:

- `inflate.sh`
- `inflate.py`
- `flat_fp4.py`
- `model.py`
- `schema_h3.py`
- `sidecar.py`
- `range_mask_codec.cpp`

Runtime dependency/compliance risks to preserve for exact replay:

- `inflate.sh` attempts `pip install brotli` if brotli is missing. The local
  repo venv has brotli, and current repo `pyproject.toml` already requires it,
  but exact replay should pin dependency closure rather than depend on network.
- `inflate.py` compiles `range_mask_codec.cpp` at inflate time and requires
  `c++`, `g++`, or `clang++`.
- Local compile check succeeded with Apple clang, and the resulting binary
  printed the expected usage text.
- No score claim is made from these checks. CUDA replay remains required.

## Byte-Only Candidate Artifacts

No-runtime-change candidates generated locally:

1. `experiments/results/pr97_h3_intake_20260504_codex/archive.pr97_deflated_p.zip`
   - bytes: `196954`
   - SHA-256: `82650a873be918537593bc9b9165c938a663012815a3a1c5c2dbd0e37d5ef213`
   - delta vs PR97 source: `-206` bytes
   - rate score delta: `-0.0001371669443431673`
   - `zf.read("p")` is byte-identical to source `p`.

2. `experiments/results/pr97_h3_intake_20260504_codex/archive.pr97_pose_model_br10_deflated_p.zip`
   - bytes: `196914`
   - SHA-256: `33a70eb0cdfddb7b362c4942bd53bb2851a60dd2e7689e53db701956e83070bf`
   - delta vs PR97 source: `-246` bytes
   - rate score delta: `-0.00016380130246805414`
   - Runtime source can remain unchanged.
   - Decoded raw mask, pose, model, and sidecar hashes all match PR97 source.

Runtime-change-only byte idea, not emitted as a runnable candidate:

- Recompress the sidecar raw `BPGD` stream with brotli quality 11 and update
  `decode_sidecar_blob` accordingly. Combined with pose/model brotli-q10 and
  deflated ZIP, estimated archive bytes are `196803` (`-357` bytes vs source,
  rate delta `-0.00023771164626461516`). This requires runtime-source review
  and exact CUDA replay before any promotion.

## Frontier Comparison And Recommendation

User-provided confirmed frontier at intake time: PR95 stemperm A++ exact T4
score `0.23089404465634825`, bytes `178277`, archive SHA-256
`e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`.

PR97 source is `18883` bytes larger than PR95 stemperm, a rate penalty of
`0.012573867491378653`. Therefore PR97 must win materially on SegNet/PoseNet
terms to beat PR95; the public writeup claim `0.22878` is external until the
queued exact T4 replay lands.

Decision rule:

- If exact T4 replay for source PR97 is not competitive with PR95, do not spend
  exact-eval slots on the byte-only candidates.
- If source PR97 exact T4 replay is competitive or beats PR95, follow-on exact
  eval candidates should be queued in this order after taking a dispatch claim:
  1. `archive.pr97_pose_model_br10_deflated_p.zip` with the original PR97 runtime.
  2. `archive.pr97_deflated_p.zip` only if a pure `p`-identity control is useful.
  3. The sidecar-brotli runtime patch only after local output-parity proof and
     runtime custody review.

This is a narrow byte opportunity, not a new score claim.
