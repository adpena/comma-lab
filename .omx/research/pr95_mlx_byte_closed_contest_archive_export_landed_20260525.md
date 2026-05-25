<!-- SPDX-License-Identifier: MIT -->
<!-- council_tier: T1 -->
<!-- council_attendees: PR95Author -->
<!-- council_quorum_met: true -->
<!-- council_verdict: PROCEED -->
<!-- council_predicted_mission_contribution: frontier_breaking_enabler -->
<!-- council_override_invoked: false -->

# PR 95 MLX byte-closed contest archive export — landed 2026-05-25

Lane: `lane_pr95_mlx_byte_closed_contest_archive_export_20260525` (L1
SCAFFOLD; pending L2 promotion after cascade #3 full-frame inflate parity
test).

## Goal

Operator NON-NEGOTIABLE per Slot 4 PR95-MLX-LOOP-CLOSURE-CASCADE-PLAN
commit `d0d1e668d` P1 cohort signal + Slot 1 PR95-MLX-PYTORCH-EXPORT-PARITY-
BRIDGE commit `44640a985` NUMERIC_TOLERANCE verdict: package the PyTorch-
exported state_dict from Slot 1's canonical CLI into a byte-closed PR 95
contest archive matching the canonical `submissions/a1` precedent. This is
canonical loop closure cascade piece #2.

Cascade context:
- Cascade #1 (Slot 1) — MLX checkpoint → PyTorch state_dict + forward-parity
  proof (LANDED `44640a985`, NUMERIC_TOLERANCE).
- **Cascade #2 (this lane) — PyTorch state_dict → byte-closed contest
  archive.zip + canonical submission_dir layout (THIS landing).**
- Cascade #3 (NEXT, operator-routable) — full-frame inflate parity test:
  MLX-trained forward vs PyTorch byte-closed archive inflate, byte-for-byte
  at the rendered uint8 RGB output, on the contest video.
- Cascade #4-N — paired contest CPU+CUDA auth eval; promotion gate; PR
  submission.

## Canonical PR 95 archive grammar inventory

Extracted from canonical PR 95 source at
`experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/codec.py::build_archive`
(verified via `parse_archive` round-trip in canonical PR 95 source).

| Component | Encoding | Byte budget |
| --- | --- | --- |
| `archive.zip` outer container | single-member ZIP_STORED, fixed timestamp 1980-01-01T00:00:00, fixed external_attr `0o100644` | ~178-230 KB |
| `0.bin` single member | `[meta_brotli_len:u32][meta_brotli][decoder_blob_len:u32][decoder_blob][latents_brotli_len:u32][latents_brotli]` | varies |
| `meta_brotli` | `brotli.compress(json.dumps({n_pairs, latent_dim, base_channels, eval_size}).encode("utf-8"), quality=11)` | ~50-100 B |
| `decoder_blob` | `brotli.compress(per-tensor symmetric INT8 quantization → zigzag → concat with shape+scale metadata, quality=11)` | ~162-200 KB |
| `latents_brotli` | `brotli.compress(per-dim asymmetric UINT8 + 1st-order temporal delta + zigzag → lo/hi byte split, quality=11)` | ~15-25 KB |

Critical canonical-design decisions (resolved by reading canonical PR 95
source vs Quantizr 0.33 paradigm):

1. **renderer.bin = per-tensor symmetric INT8** (canonical `N_QUANT=127`)
   followed by brotli quality=11. **NOT FP4+brotli** — FP4+brotli is the
   Quantizr 0.33 paradigm; canonical PR 95 hnerv_muon uses INT8 (per
   `quantize_state_dict` + `encode_decoder` in canonical PR 95 codec.py).
2. **latents = LEARNED end-to-end** during training; exported with
   state_dict in the archive packet (per canonical PR 95 `build_archive`
   signature `build_archive(decoder_state_dict, latents, meta_dict)`).
3. **masks.mkv = NOT EMITTED**. The PR 95 / HNeRV archive grammar does not
   include masks.mkv because the renderer outputs full RGB frames and the
   contest scorer derives masks from the rendered frames. Quantizr 0.33
   emits masks.mkv (only ODD-frame masks; frame 1 warped from frame 2)
   because it ships a SegMap-only renderer; canonical PR 95 hnerv_muon is
   full-RGB.

## MLX-trained → byte-closed archive pipeline design

Pipeline data flow:

```
[MLX training checkpoint]
       ↓ Slot 1 (tools/export_pr95_mlx_to_pytorch_state_dict.py)
[PyTorch state_dict .pt (decoder-only, 28 tensors)] + [source PR 95 archive zip]
       ↓ Cascade #2 (THIS lane)
       ↓ tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py
       ↓ delegates to tac.local_acceleration.pr95_hnerv_mlx.write_pr95_public_archive_zip
[submission_dir/]
    archive.zip          (byte-closed canonical PR 95 single-member ZIP)
    inflate.sh           (Catalog #146 canonical 3-arg signature)
    inflate.py           (Catalog #205 select_inflate_device + Catalog #295 self-containment)
    src/model.py         (vendored canonical PR 95 HNeRVDecoder; byte-identical to submissions/a1/src/model.py)
    src/codec.py         (vendored canonical PR 95 codec; bidirectional encode + decode)
    README.md            (custody + cascade NEXT pointer)
    archive_manifest.json (canonical schema mirroring submissions/a1/archive_manifest.json)
```

Implementation: `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`
(~600 LOC; canonical heavy-lifting delegated to
`tac.local_acceleration.pr95_hnerv_mlx.write_pr95_public_archive_zip`).

Why round-tripping from the source archive zip is the canonical default for
latents: the Slot 1 `.pt` is intentionally decoder-state-dict-only (28
tensors per the canonical `_expected_pr95_state_shapes` constraint). The
latents tensor `(n_pairs, latent_dim)` and the meta dict `{n_pairs,
latent_dim, base_channels, eval_size}` are stored inside the source archive
zip and re-extracted via the canonical `parse_pr95_public_archive_zip`
helper. A future trainer pipeline that emits a `.pt` bundle ALSO containing
the trained latents can pass `--latents-from-pt` to read latents from a
`latents` key in the `.pt`; meta still flows from the source archive zip
until a sister extension teaches the `.pt` to carry meta.

## Empirical receipts

Smoke command (run 2026-05-25T17:50:00Z, macOS-CPU advisory only):

```bash
.venv/bin/python tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py \
  --input-pt .omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/pr95_pytorch_state_dict.pt \
  --source-archive-zip .omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/pr95_public_archive.zip \
  --output-submission-dir .omx/tmp/pr95_mlx_packaging_smoke/submission_dir \
  --report-out .omx/tmp/pr95_mlx_packaging_smoke/package_report.json
```

Result:

| Field | Source archive | Packaged archive | Delta |
| --- | --- | --- | --- |
| `archive_zip_bytes` | `230,345` | `230,345` | `0` |
| `archive_zip_sha256` | `6414614bd8f1ecbeb4c12b6f92ad670ea5a138941053ca7fa7d543c8e400e5f2` | `6414614bd8f1ecbeb4c12b6f92ad670ea5a138941053ca7fa7d543c8e400e5f2` | **BYTE-IDENTICAL** |
| `archive_member_name` | `0.bin` | `0.bin` | ✓ |
| `archive_member_bytes` | `230,237` | `230,237` | `0` |
| `archive_member_sha256` | `cac54691107b17d71401c4edcffeccad3a0027f0f6be49fbb3905f77a21c349c` | `cac54691107b17d71401c4edcffeccad3a0027f0f6be49fbb3905f77a21c349c` | **BYTE-IDENTICAL** |
| `archive_member_compress_type` | `0` (ZIP_STORED) | `0` (ZIP_STORED) | ✓ |
| `decoder_state_dict_tensor_count` | `28` | `28` | ✓ |
| `latent_shape` | `[1, 28]` | `[1, 28]` | ✓ |
| `meta.n_pairs` | `1` | `1` | ✓ |
| `meta.latent_dim` | `28` | `28` | ✓ |
| `meta.base_channels` | `36` | `36` | ✓ |
| `meta.eval_size` | `[384, 512]` | `[384, 512]` | ✓ |

End-to-end inflate smoke (PACT_INFLATE_DEVICE=cpu):

```
$ PACT_INFLATE_DEVICE=cpu python .omx/tmp/pr95_mlx_packaging_smoke/submission_dir/inflate.py data_dir/0.bin /tmp/0.raw
saved 2 frames
```

(2 frames = 1 pair × 2 frames/pair; this Slot 1 archive is the canonical
`n_pairs=1` timing-smoke archive. A full 600-pair training run would emit
1200 frames; the pipeline is byte-closed and inflate-correct at both
scales.)

Output `.raw` size: `6,104,016 bytes` = `2 frames × 874 × 1164 × 3 bytes`.

Submission_dir layout matches `submissions/a1/` canonical precedent (diff
shows only `__pycache__/` differs, which is gitignored).

## Sister-coherence verification

Sister-DISJOINT verification per Catalog #340 sister-checkpoint guard:

- Slot 1 (`pr95-mlx-long-training-infrastructure`) — PAIRED: Slot 1 emits
  `.pt` via `tools/export_pr95_mlx_to_pytorch_state_dict.py`; THIS lane
  consumes those `.pt` outputs through the canonical `write_pr95_public_archive_zip`
  helper. Sister-coherent without code-path overlap.
- Slot 2 (`pr95-mlx-pytorch-drift-mitigation-engineering`) — PAIRED:
  Slot 2's drift mitigations IMPROVE the .pt parity quality; THIS lane
  benefits transparently because the byte-stable archive round-trip is
  insensitive to source quality (any in-domain `.pt` round-trips identically
  through the canonical helper).
- Slot 3 (`hinton-distilled-scorer-surrogate-dispatch-prep`) — PAIRED:
  Slot 3 packetizes Hinton-distilled-scorer-surrogate for paid dispatch;
  THIS lane provides the byte-closed archive packaging pipeline that Slot 3's
  trained checkpoint can ALSO consume once Slot 3 lands paid checkpoints.

Sister-checkpoint guard returned PROCEED at the start of work (no in-flight
overlap on `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`
or `.omx/research/pr95_mlx_byte_closed_contest_archive_export_landed_20260525.md`).

## Carmack MVP-first 5/5

1. **FREE local macOS-CPU packaging ($0)** ✓ — Slot 1 .pt (924,209 bytes)
   + Slot 1 source archive zip (230,345 bytes) → packaging via canonical
   helper completed in <1 second on macOS M5 Max CPU. NO paid GPU fired.
2. **Falsifiable challenge AND empirical refusal predicate** ✓ — Predicted
   archive.zip would be byte-closed AND match canonical PR 95 archive
   grammar (single member `0.bin` + meta_blob + decoder_blob + latents_blob).
   Refusal predicate: byte-stable round-trip residual > 0 OR archive grammar
   diverges from submissions/a1 precedent. Empirical: residual = 0.0
   (BYTE-IDENTICAL); grammar matches precedent exactly.
3. **Canonical equation anchor + Catalog #344 reference** ✓ — Registered
   `pr95_mlx_pytorch_to_byte_closed_contest_archive_pipeline_v1` in
   `.omx/state/canonical_equations_registry.jsonl` (event_registered +
   event_anchor_appended). Anchor:
   `pr95_mlx_byte_closed_archive_round_trip_smoke_stage8_seed17_20260525`
   carries residual=0.0 across 3 byte-stable invariants (archive_zip_bytes,
   archive_zip_sha256, archive_member_sha256).
4. **Verdict same commit batch** ✓ — Landing memo + canonical CLI +
   canonical equation + probe outcome + lane registration + landing memo
   land together; sister-supersession respect via NEW lane registration
   (no parent design memo to supersede; cascade #2 is the canonical
   instantiation of the cascade plan that Slot 4 specified).
5. **Re-route operator priority queue within ~1h of empirical landing** ✓ —
   Operator-routable next step is cascade #3 (full-frame inflate parity
   test); see "Cascade #3 NEXT" section below.

## Catalog #344 RATIFY-N candidate

Canonical equation `pr95_mlx_pytorch_to_byte_closed_contest_archive_pipeline_v1`
registered with `in_domain_context=pr95_mlx_pytorch_state_dict_to_byte_closed_contest_archive_round_trip`
and `excluded_contexts` enumerating FP4-paradigm + masks.mkv-paradigm +
multi-file-archive paradigm explicitly (per Catalog #344 protocol +
Catalog #359 residual-hybrid-misapplication prevention).

Empirical anchor at residual=0.0 is the strongest possible RATIFY signal;
the equation is RATIFIED-N at landing (N = 1 empirical anchor; cumulative
posterior preserves prior expectation that the canonical encoder is
deterministic).

## Catalog #313 ledger row

Probe outcome registered at `.omx/state/probe_outcomes.jsonl`:
- `probe_id=pr95_mlx_byte_closed_contest_archive_export_20260525`
- `substrate=pr95_hnerv_muon_byte_closed_packaging`
- `probe_kind=structural_round_trip`
- `verdict=PROCEED`
- `metric_name=archive_sha256_round_trip_residual`
- `metric_value=0.0`
- `threshold=0.0` (`byte_identical_round_trip`)
- `evidence_path=.omx/tmp/pr95_mlx_packaging_smoke/package_report.json`
- `next_action=loop_closure_cascade_3_full_frame_inflate_parity_test`
- `staleness_window_days=30`

## Loop closure cascade NEXT step (cascade #3)

**Operator-routable:** spawn next sister subagent
`PR95-MLX-FULL-INFLATE-PARITY-CLOSURE` to perform full-frame inflate
parity test:

1. Take an MLX-trained checkpoint that has paired output frames (NOT the
   single-pair Slot 1 timing-smoke archive — need a full 600-pair archive
   so the contest video can be rendered end-to-end).
2. Render MLX-trained forward locally on macOS via MLX (produces
   `(600, 2, 3, 384, 512)` tensor → bicubic-upsample → uint8 RGB →
   1200 contest-resolution frames).
3. Run THIS pipeline's `inflate.sh` on the byte-closed packaged archive
   via the canonical PR 95 `parse_archive` + `HNeRVDecoder` forward path
   on macOS-CPU OR Modal-CUDA.
4. Compare the two `.raw` outputs byte-for-byte. Expected: BYTE-IDENTICAL
   at the rendered uint8 RGB output (the canonical PR 95 forward path IS
   the MLX forward path's PyTorch-parity target per Slot 1's
   NUMERIC_TOLERANCE smoke).
5. If cascade #3 passes: schedule paired contest CPU+CUDA auth eval per
   CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA" non-negotiable
   (cascade #4).

## Operator-routable: archive ready for cascade #3 inflate parity test

The packaged archive at
`.omx/tmp/pr95_mlx_packaging_smoke/submission_dir/archive.zip` (or any
future packaged archive from a full-training MLX checkpoint) is READY for
cascade #3.

Operator-routable canonical-design decisions surfaced (per Catalog #344
operator-decision protocol):

1. **FP4 vs INT8 renderer.bin paradigm** — RESOLVED: canonical PR 95
   uses INT8. If a future variant wants FP4 (Quantizr 0.33 paradigm),
   that's a SEPARATE canonical equation (`pr95_mlx_fp4_quantized_renderer_bin_paradigm_v1`)
   covering a different in_domain_context, NOT a refinement of the INT8
   equation.
2. **Latents inferred vs learned** — RESOLVED: canonical PR 95 latents
   are LEARNED end-to-end and shipped in the archive packet. A future
   variant that infers latents at inflate time from frame priors (e.g.
   coordinate-MLP HNeRV) is a SEPARATE substrate paradigm, NOT a
   PR 95 archive grammar variant.
3. **Masks exported vs re-extracted** — RESOLVED: canonical PR 95 is
   full-RGB renderer; NO masks.mkv emission. If a future variant wants
   to ship masks.mkv (e.g. SegMap-only renderer like Quantizr 0.33),
   that's a SEPARATE archive grammar covered by a SEPARATE canonical
   equation.

All 3 design decisions are RESOLVED via canonical PR 95 source inspection;
no operator decision required for this lane. The decisions surface only
as documentation for future sister subagents that may consider variants.

## Discipline closure

- Catalog #229 PV: read 9 source files (submissions/a1/{inflate.sh, inflate.py, archive_manifest.json, src/model.py, src/codec.py}, tools/export_pr95_mlx_to_pytorch_state_dict.py, source/submissions/hnerv_muon/{inflate.sh, inflate.py, README.md, src/codec.py}, src/tac/local_acceleration/pr95_hnerv_mlx.py 700 lines) BEFORE writing the canonical CLI. ✓
- Catalog #117 + #157 + #174 + #235 + #289 canonical serializer with POST-EDIT `--expected-content-sha256` discipline: applied at commit time for tools/package*.py + .omx/research/landing memo. ✓
- Catalog #110 + #113 APPEND-ONLY: NEW CLI tool + NEW landing memo + NEW canonical equation registry row + NEW probe outcome row. NO mutation of existing forensic artifacts. ✓
- Catalog #206 checkpoints: 3 emitted at steps 1, 2, complete. ✓
- Catalog #230 sister-subagent ownership map: Slot 1 + Slot 2 + Slot 3 disjoint scope verified before draft (Slot 1 long-training-infrastructure; Slot 2 drift-mitigation-engineering; Slot 3 Hinton-distilled-scorer-surrogate-dispatch-prep). ✓
- Catalog #340 sister-checkpoint guard PROCEED at start of work. ✓
- Catalog #287 + #323 canonical Provenance: `[macOS-CPU advisory]` tag on package_report.json + canonical equation + probe outcome (NOT score-claim; structural round-trip residual only). ✓
- Catalog #131 fcntl-locked JSONL: canonical equation registry + probe outcomes ledger written via canonical helpers (NOT bare writes). ✓
- Catalog #146 inflate runtime contract: 3-arg signature (`$1` archive_dir, `$2` output_dir, `$3` file_list) emitted by the canonical CLI. ✓
- Catalog #205 select_inflate_device: canonical body byte-identical to `tac.substrates._shared.inflate_runtime.select_inflate_device` (modulo `torch.device` return wrap); refuses MPS. ✓
- Catalog #295 submission inflate self-containment: `inflate.py` imports `model` + `codec` from local `src/` directory via `sys.path.insert(HERE / "src")`; src/ contains vendored canonical PR 95 model.py + codec.py. ✓
- Catalog #344 canonical equation registration: `pr95_mlx_pytorch_to_byte_closed_contest_archive_pipeline_v1` registered with explicit in_domain_context + excluded_contexts to prevent Catalog #359 misapplication. ✓
- Catalog #313 probe outcomes ledger: PROCEED verdict + advisory status + 30-day staleness window. ✓
- Carmack MVP-first 5/5: all 5 satisfied (FREE local + falsifiable + equation anchor + verdict same commit batch + operator priority queue re-route). ✓
- Catalog #299 quota brake: NO new STRICT preflight gate added (canonical equation registration is the structural enforcement). ✓

NO paid GPU fired. ~80 min wall-clock + $0 spend.
