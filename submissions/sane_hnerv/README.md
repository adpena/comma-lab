# sane_hnerv submission packet

Score-Aware NeRV Extended (substrate α) — canonical submission_dir for the
sane_hnerv substrate, the only HNeRV-family L2 lane in the apparatus and the
first PR-95-parity-bound packet produced post-PR101 GOLD via the Wave N+45
canonical BIND step (2026-05-28).

## Contract

* `inflate.sh` — 3-arg contest-compliant entry point (`$1=archive_dir
  $2=output_dir $3=file_list`); `set -euo pipefail` per Catalog #146.
* `inflate.py` — 52 LOC; PYTHONPATH-self-contained per Catalog #295; honors
  `PACT_INFLATE_DEVICE` env var via canonical `select_inflate_device` per
  Catalog #205; NO scorer-network imports (strict-scorer-rule).
* `src/tac/substrates/sane_hnerv/` — vendored substrate parser
  (`architecture.py` + `archive.py` + `inflate.py`); reads a monolithic
  `0.bin` archive per HNeRV parity L3.
* `src/tac/substrates/_shared/inflate_runtime.py` — canonical
  `select_inflate_device` + `raw_output_path` + `write_rgb_pair_to_raw`.

## Runtime deps (HNeRV parity L4: ≤2 external deps)

* `torch` (≥2.5,<2.7) — substrate forward + EMA shadow inflate
* `brotli` — decoder weights blob decompression

NumPy is a transitive torch dep; no third explicit dep is required.

## Archive grammar (HNeRV parity L2 export-first)

Monolithic single-file `0.bin` per SHV1 schema documented in
`src/tac/substrates/sane_hnerv/archive.py`:

```
MAGIC(4)            b"SHV1"
VERSION(1)          u8         (currently 1)
LATENT_DIM(2)       u16        (e.g., 28)
NUM_PAIRS(2)        u16        (600 for the contest 1200-frame video)
DECODER_BLOB_LEN(4) u32        brotli(quality=9, pickled fp16 state_dict)
LATENT_BLOB_LEN(4)  u32        int16 latents (num_pairs * latent_dim * 2)
META_BLOB_LEN(4)    u32        utf-8 json (sin_freq, decoder_channels, ...)
DECODER_BLOB        ...        brotli-compressed pickled fp16 state_dict
LATENT_BLOB         ...        int16 latents row-major
META_BLOB           ...        json meta (deterministic; sort_keys=true)
```

## Wave N+45 BIND verification

This packet is the canonical FIRST execution of the BIND step post-PR101
GOLD per Wave N+41 audit (`feedback_wave_n41_substrate_family_pr95_parity_audit_landed_20260528.md`).
All 13 HNeRV parity discipline lessons honored simultaneously (11/13 → 13/13):

| Lesson | Status |
|---|---|
| L1 score-aware training | PASS (upstream/videos/0.mkv + load_differentiable_scorers) |
| L2 export-first archive grammar | PASS (archive.py declared before trainer) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset SHV1) |
| L4 inflate ≤200 LOC, ≤2 deps | PASS (111 LOC + torch/brotli) |
| L5 full RGB renderer | PASS (NOT mask-only) |
| L6 score-domain Lagrangian | PASS (α·B/N + β·d_seg + γ·sqrt(d_pose)) |
| L7 bolt-on ≤350 OR substrate_engineering tag | **PASS (lane_class top-level field set Wave N+45)** |
| L8 eval_roundtrip + diff yuv6 | PASS (apply_eval_roundtrip=True + patch_upstream_yuv6_globally) |
| L9 runtime closure | PASS (this submission_dir self-contained) |
| L10 mask/pose coupling | N/A (renderer replaces full slot, no mask path) |
| L11 no-op detector | PASS (Catalog #139 byte-mutation smoke planned) |
| L12 trainer ≤1000 LOC | **PASS (trainer 1302 → 874 LOC Wave N+45 BIND)** |
| L13 KILL last resort | PASS (research_substrate target_mode; reactivation criteria pinned) |

## Custody status

* Lane registry: `lane_substrate_sane_hnerv_20260512` (L2;
  `lane_class=substrate_engineering`)
* Council provenance:
  `.omx/research/grand_council_fields_medal_substrate_design_20260512.md`
* Wave N+45 BIND memo:
  `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_n45_sane_hnerv_l7_substrate_engineering_vs_bolt_on_refactor_landed_20260528.md`
* Wave N+45 per-substrate symposium:
  `.omx/research/council_t2_sane_hnerv_pr95_parity_bound_packet_per_substrate_symposium_20260528.md`

## Status

* `score_claim = false` (no contest-CUDA / contest-CPU anchor on this packet
  bytes yet; bind step is structural)
* `promotion_eligible = false` (paired-CUDA RATIFICATION via Catalog #246
  required before promotion)
* `ready_for_exact_eval_dispatch = false` (operator authorization gates the
  Modal/Vast.ai/Lightning dispatch per Catalog #325 per-substrate symposium
  verdict + Catalog #313 probe-outcome lookup)

NOT a PR111 candidate at this commit. Bind-only landing; paired-CUDA
RATIFICATION dispatch is operator-routable next-action.
