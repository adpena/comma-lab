# QBFLOW initialized packet schema v1

Status: **FROZEN BEFORE FIRST PAYLOAD** · arm: `ddm_qbflow_rate_first_rung` ·
axis: scorer-free byte gate only · score authority: none.

This schema is the counted initialized object required by
`.omx/research/charters/ddm_qbflow_rate_first_rung_20260827.md`. It changes both the basis and the
future objective relative to the measured dead predecessors:

- basis: a nonlinear step-reachable coordinate field emits all ten signed class interfaces; its
  second pass is conditioned on the decoder's own coarse Road probability and tangent, with a
  dedicated 8/16/24/32-cycle along-tangent comb so it does not inherit the measured 3.2x dash
  frequency deficit;
- objective: boundary features condition RGB directly and a separate interior head owns RGB and
  pose12. Any later training is scorer-in-loop through the real renderer and `R`; no fixed class
  paint or post-hoc pose mechanism exists in this object.

The initialized packet proves only whether this counted shape exists below the rate gate. It does
not prove that training reaches any distortion, score, runtime, or contest row.

## Counted/free boundary

Counted in `archive.zip`:

1. architecture/config section;
2. every learned model tensor after real role-specific quantization;
3. latent quantizer metadata;
4. every pair's boundary and interior latent codes;
5. packet headers, integrity fields, and deterministic single-member ZIP framing.

Free receiver computation under rule 118:

- normalized `(x,y,t)` coordinates and a generic perspective transform;
- deterministic Fourier/comb features;
- the fixed ten-interface-to-five-class incidence matrix;
- coarse Road probability, its finite-difference tangent, and all second-pass conditioning derived
  from the decoder's own output;
- packet parsing, dequantization, and NumPy forward code.

No GT, scorer weights, source masks, pair facts, source-derived basis table, or video-derived
constant may enter free code. Initialized random values are conservatively serialized and counted
even though a generic seed could reproduce them.

## Architecture and backward capacity choice

The source field has 1,625,624 four-neighbour interfaces over 600 pairs. At the 101,150-byte
reference rate and current pose, the no2 arithmetic allows `d_seg <= 4.4667138915998396e-4`, or
52,691.50 cell-errors: only 3.2413% of the interface count. This does not predict distortion; it
sets the representation burden.

The signed interface head has ten outputs, one for every unordered class pair. Four explicit
along-tangent frequencies require `10 interfaces * 4 frequencies * 2 phases = 80` degrees before
shared mixing; sixteen boundary-conditioning channels raise the flow width to 96. The separate
interior latent has twelve dimensions because the receiver emits a twelve-dimensional pose head,
while a sixteen-dimensional boundary latent controls interface flow and FiLM. Three residual flow
maps follow the input step map, giving four trainable step stages.

The chosen object has 79,513 learned scalars. The complete cap is 137,986 bytes. Reserving roughly
32 KB for 600 independently attributable latent records and framing leaves about 105 KB for learned
parameters; the role-based 8/10/12/16-bit menu places the 79,513-scalar object inside that envelope
without fp32-by-default. This is a falsifiable capacity portrait, not an optimal-width claim.

Precision roles are frozen as:

| role | bits | reason at this rung |
|---|---:|---|
| hidden mixing / FiLM | 8 | bulk learned capacity; predecessor rate gates show int8 objects can fit |
| coordinate and flow input maps | 10 | first projections set all downstream interface coordinates |
| interface/RGB/pose output maps | 12 | terminal signed and photometric outputs are decision-adjacent |
| step slopes, step centers, and biases | 16 | threshold placement and offsets are the sharp boundary controls |
| per-pair boundary latent | 10 | controls signed interface flow |
| per-pair interior latent | 12 | controls RGB interiors and the pose head |

This is role-derived initialization. No trained sensitivity exists yet, so it is explicitly not a
measured precision-waterfill optimum. The future fire order requires re-encoding every retained
checkpoint and may alter precision only from measured receiver/scorer sensitivity.

## Inner packet

All integers are big-endian. The packet header is:

```text
magic[4] = "QBF1"
version:u8 = 1
flags:u8 = 0
section_count:u16
```

Sections are strictly increasing by ID and use:

```text
section_id:u8
codec_id:u8           # 1 Brotli-q11, 2 LZMA-9-extreme/XZ, 3 zlib-9
reserved:u16 = 0
raw_len:u32
coded_len:u32
raw_sha256[32]
coded_crc32:u32
coded_payload[coded_len]
```

Each section is fully re-encoded through all three real coders; the smallest result wins with
codec ID as deterministic tie-break. Primary/repeat bytes for every candidate are retained.

Section IDs:

1. `config`: canonical JSON architecture description;
2. `model`: `QBT1` tensor table containing name, shape, precision, fp32 scale, and packed symmetric
   signed codes for every receiver-consumed parameter;
3. `latent_meta`: `QBM1` fixed ABI containing 10-bit/16-D boundary and 12-bit/12-D interior scales;
4. `latents`: `QBL1` ordered pair records; each `QBR` record carries pair ID, both precision IDs,
   packed code lengths, and its own CRC32.

Parser requirements: known version/flags/sections/codecs only; strict order; no duplicate tensor or
pair IDs; exact tensor-name/shape set; exact lengths; valid coded CRC, raw SHA-256, record CRC;
minimal complete bit payload; zero trailing bytes.

## Complete framing and n32 estimator

The contest-shaped artifact is a deterministic stored ZIP containing exactly one member `0.qbf`.
Timestamp is 1980-01-01 and mode is 0644. Because inner sections are already real-coded, ZIP does
not apply another compressor.

The same no2/qbw1 seeded stratified selection is reused: NumPy PCG64 seed 20260827, ten 60-pair
temporal blocks crossed with low/high Road-Lane crack count, two draws per half in blocks 0-5 and
one per half in blocks 6-9. Selected IDs must remain:

`4, 31, 49, 52, 62, 90, 100, 113, 128, 148, 173, 179, 186, 187, 214, 236, 256, 260, 268, 278,
326, 328, 341, 352, 368, 382, 444, 456, 483, 508, 563, 573`.

The independently reset record estimator is:

```text
B_var_hat = sum_h (N_h/n_h) * sum_i reset_record_bytes_i
B_hat = shared_archive_bytes(config + model + latent_meta) + ceil(B_var_hat)
```

The builder additionally materializes the full 600-record packet and complete `archive.zip`. The
gate requires both the preregistered projected `B_hat` and the stronger exact full archive to be at
most 137,986 bytes. Neither is a distortion or score measurement.

## Receiver and refusal

`experiments/ddm_qbflow_packet.py:reference_forward` must consume every declared architectural
branch on a real coordinate grid and emit:

- ten signed interfaces and five derived class logits;
- boundary-conditioned RGB for frame 0 and frame 1;
- a separate pose12 output from the interior head;
- the decoder-derived coarse Road probability and tangent used by the flow.

Behavioral tests must show that changing the interface, renderer, and pose tensors changes their
respective real outputs. Metadata-only tests are insufficient.

A one-bit mutation in every counted section is retained and must refuse at parse time. Primary and
repeat inner packets and ZIP archives must be byte-identical. Parse-back must reproduce every
quantized tensor and every selected/full latent code exactly.

## Scope and succession

Pass means only `RATE_SHAPE_EXISTS_INITIALIZED_UNTRAINED`. It authorizes MAIN to build the separate
scorer-in-loop training stage with chunk size at most 30, a live 116-GiB memory preflight, periodic
and per-stage byte-closeable EMA checkpoints, and a same-budget discrete QBW1 control. Every trained
checkpoint must be reserialized; initialized bytes never transfer as a trained rate claim.

Failure closes only this QBFLOW v1 initialized first-rung formulation. It does not prove an implicit
boundary-flow family lower bound.

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `static_packet_custody_byte_delta_score_savings_v1` — `tac.canonical_equations (registry: static-packet custody byte-delta)` (`tac.canonical_equations`). **Relation:** IN-DOMAIN (the schema's gate is this law's rate-only arithmetic).

The frozen packet's only gate is byte-side: both the pre-registered projected `B_hat` and the exact full archive must be ≤ 137,986 B, and the memo says plainly that neither is a distortion or score measurement — which is this law's `NOT a score claim until paired CUDA` clause stated at the schema surface.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
