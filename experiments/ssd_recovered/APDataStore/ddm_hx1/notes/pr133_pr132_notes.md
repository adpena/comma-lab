# ddm_hx1 intake — PR133 (cpr1_cbq_matched8) + PR132 (veigapunk_hpac_ft)
Date 2026-08-17. Read-only static intake. No launches, no scorer runs, no repo writes.
Custody: /Volumes/APDataStore/pact/ddm_hx1/intake/{pr133,pr132}/

## Headline
Both PRs are payload-only deltas on PR130 (CPR1, Fesal Fayed). Decoder source is
BYTE-IDENTICAL between PR133 and PR132 for all six shipped modules — so neither
changed the deployed architecture. All difference lives in archive.zip member `p`.

PR133 = OFFICIALLY EVALUATED + on the leaderboard: pose 0.00000896, seg 0.00029660,
190,212 B, S=0.165780 [contest-CUDA T4, maintainer eval bot 2026-08-08].
PR132 = NEVER EVALUATED by anyone. Closed with zero measured metrics.

## Payload decomposition (MEASURED by me, parsing archive member `p`)
payload = u32 models_xz_len | xz(models) | token_stream
models = u32 sem_len | u32 carrier_len | semantic | carrier | hpac

| section | PR133 | PR132 | note |
|---|---:|---:|---|
| semantic renderer | 40,252 | 40,252 | PR132 DIFFERS (20.77% of bytes); PR133 = CPR1 |
| pose carrier | 22,304 | 23,054 | PR133 DIFFERS; PR132 = CPR1 |
| HPAC model | 20,179 | 20,179 | identical, sha b07fff73fac41c5f |
| models_xz | 73,128 | 73,944 | |
| token stream | 116,980 | 116,980 | identical, sha 948379872ff81a4e — 61.5% of payload, untouched by both |
| payload | 190,112 | 190,928 | |
| archive | 190,212 | 191,028 | +100 B zip overhead, single stored member `p` |

PR133 carrier sha256 080aaf3206e1... matches its verification.json exactly.

## CPR1 pose carrier mechanics (inflate.py + carrier_codec.py)
CARRIER_DIM=12, CARRIER_H,W=24,32, CARRIER_AMPLITUDE=64.0, N=600.
- basis: 12 atoms x 3ch x 24 x 32 = 27,648 signed codes, BASIS_BITS=5 (carrier_codec.py:15),
  zigzag -> ONE GLOBAL canonical Huffman over a 32-symbol alphabet (:54-140). No context model.
- per-atom fp32 basis_scales (12) + fp32 coeff_scales (12).
- coefficients: 600x12 signed int12 (COEFF_BITS=12, :16), delta along TIME + zigzag
  (inflate.py:265-270), then per-DIMENSION Rice; k picked by exhaustive argmin of actual
  bit count over k in [0,12) (carrier_codec.py:221-224).
- render (inflate.py:601-643): basis bicubic->eval res, per-atom zero-mean + RMS-normalize,
  carrier = einsum("bk,kchw->bchw", coeff, basis)/sqrt(12); slave frame = 127.5 + 64*carrier.
So the PoseNet-scored slave frame is neutral gray + a rank-12 linear image basis.

## PR133 "cbq" / "matched8" — what they ACTUALLY are
- cbq = "Compensability-aware Basis Quantization" (README.md, CREDITS.md, verification.json
  payload_change.method).
- matched8 = "matched"-EFFORT CONTROL at "8" full-600 coefficient-search passes.
  NOT an 8-bit quantizer, NOT an 8-level codebook, NOT a matched filter.
- Mechanism: coarsen basis atoms 2, 5, 9 from 5-bit to 4-bit signed support, then RE-SOLVE
  the already-transmitted int12 coefficients so the free downstream channel absorbs the
  induced error; accept against exact forward PoseNet AND actual packaged bytes.
- Format did NOT change: BASIS_BITS is still 5 in carrier_codec.py. Restricting 3 of 12 atoms
  to a 4-bit lattice shrinks the ENTROPY of the shared 32-symbol Huffman stream; the Huffman
  coder converts that into bytes (828 B). Pure histogram shaping inside a fixed container.
- Quantizer is plain UNIFORM: int code x one fp32 per-atom scale. No Lloyd-Max, no k-means,
  no codebook, no dead zone, no asymmetric or non-uniform level spacing, no trellis/TCQ,
  no soft/annealed assignment. Grep for lloyd|kmeans|codebook|trellis|tcq|deadzone across all
  5 modules: 0 hits.
- CO-DESIGN: NOT RD-Lagrangian. But rate IS in the accept loop (real packaged bytes) and
  distortion is measured AFTER compensation by the coefficient channel. The transferable idea
  is the pricing rule, not the quantizer.

## PR133 score decomposition (arithmetic verified against official rows)
CPR1  0.029660 + 0.0152676 + 0.1272136 = 0.1721412
PR133 0.029660 + 0.0094657 + 0.1266543 = 0.1657800
dS = -0.006361 = pose -0.005802 (91.2%) + rate -0.000559 (8.8%) + seg 0.
Author's own effort-matched control (local MPS, advisory): control pose 8.0833e-6 /191,040 B
/205 pass-8 accepts vs candidate 7.8771e-6 /190,212 B /140 accepts -> CBQ = 2.55% of the pose
gain + 828 B. So CBQ's honest share ~ -0.000666 (10.5%); the coefficient re-solve ~ -0.00570
(89.5%) AT ZERO BYTES. Neither branch converged (both still accepting on pass 8).

## PR132
- HPAC in their usage = the inherited PR130 entropy model (arXiv:2511.10991 Hierarchical
  Parallelism + Progressive Adaptation). The HPAC model bytes are UNCHANGED. The name is a
  misnomer for what the PR does.
- "ft" = 800-step AdamW fine-tune of the int4 SEMANTIC RENDERER against SegNet CE at eval
  resolution (README.md line 3; meta.json steps:800), + a lossless xz repack worth -24 B.
- Loss: SCORER (SegNet CE), per the README string only. NO training code shipped; no loss
  line exists in the artifact. Assertion, not a receipt.
- The ft is REAL and non-trivial: semantic blob differs from CPR1 in 8,362/40,252 bytes
  (20.77%), same size, still int4 codes + fp16 per-channel scales (inflate.py:171-196), so
  they re-quantized onto the int4 lattice after the ft.
- Did it help? UNKNOWN. Zero measurements of THIS archive exist. The only numbers in the PR
  are the PARENT's, on the author's own RTX 5070. Their local parent seg 0.00029598 vs the
  official T4's 0.00029660 for the same archive = 6.2e-5 score of device drift, so any local
  claim below that is noise. Only measured delta: -24 B = -0.0000160 score.
- Order-of-operations: quantize -> ft -> re-quantize -> lossless repack, with NO compensation
  loop and no accept/reject gate. PR133 explicitly found in the same lineage that this shape
  fails ("direct quantization alone looked promising on a small screen, but it broke on larger
  batches").

## Bonus mechanism (integer_model_io.py, inherited, not claimed by either PR)
"IHS1" self-compressed integer model: a NIBBLE PER OUTPUT CHANNEL carrying that channel's bit
depth, then the weights bit-packed at variable per-channel depths (:15-92). Per-channel
adaptive bit-width with a 4-bit-per-channel side table. bits=0 means a zeroed channel.

## Falsifiers stated plainly
- "matched8" is not an 8-bit anything.
- CBQ's quantizer is plain uniform per-atom int-x-fp32-scale. The novelty is the SELECTION
  RULE, not the quantizer.
- PR132 is an unmeasured assertion. Do not cite any PR132 number as evidence.
