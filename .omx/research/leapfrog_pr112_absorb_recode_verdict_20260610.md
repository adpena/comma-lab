# LEAPFROG verdict — absorb PR #112's entropy coder onto our R3 frontier (lossless recode)

**Date:** 2026-06-10
**Subagent:** `leapfrog_pr112_absorb_recode_20260610`
**Lane:** `lane_pr110_payload_entropy_recode_20260610`
**Mode:** RACE (public PR #112 mattneel 0.191126 `[contest-CPU external]` beat our 0.19198275 frontier)
**Source plan:** `.omx/research/public_pr112_frontier_beat_intake_20260610.md` MOVE 1 + MOVE 2.

## Headline

PR #112 is a pure lossless entropy re-code of OUR merged PR #110 payload (intake proved
byte-identical decode). We **absorbed their coder** (`codec_ctx.py`) and applied it to our
**R3 candidate frontier** (`pr110pp_r3_onhost_mode_table`, the current pointer frontier,
178,495 B, 0.19198275). R3 already wins on the **distortion axes** vs PR #112 (its FECa
FEC10-hybrid selector + DQS1 q-substitution spend extra selector bytes to lower d_seg/d_pose);
PR #112 wins on **rate**. The two are orthogonal sections of the same FP11 grammar, so the
recode composes: same pixels, lower rate.

## What was built

- `tac.packet_compiler.ctx_range_coder` — vendored verbatim from PR #112's `codec_ctx.py`
  (MIT, attributed): per-tensor adaptive 256-ary range coder (constriction ANS, geometric-primed
  priors, IEEE-exact float64 tables) for the decoder-weight streams + per-dim AR(1)+cross-dim
  causal coder for the latents.
- `tac.packet_compiler.pr110_payload_entropy_recode` — the canonical materializer (MOVE 2,
  closes the orphaned `byte_range:entropy_recode` BLOCKED byte-shaving signal). Splits the FP11
  member (reusing `feca_selector_reparameterize.split_fp11_member`), recodes the decoder + latent
  sections, keeps the **FECa selector + DQS1 tail verbatim** (PR #112's coder only understands
  plain FEC6; re-coding FECa would be a regression + parse-back hazard), repacks into a CTXR
  container, and **verifies losslessness fail-closed**.
- `tools/build_pr110_payload_entropy_recode_candidate.py` — CLI emitting the recoded archive +
  byte-closure proof + no-op detector + manifest.
- Recoded submission_dir runtime: `experiments/results/pr110_payload_entropy_recode_20260610/submission_dir`
  (inflate.py reconstructs decoder_sd + latents from the byte-exact CTXR-decoded raw, then runs
  the IDENTICAL R3 render+FECa-selector+DQS1 chain; ships `src/codec_ctx.py` + R3's `src/` +
  `encoder/`). Inflate-deps: numpy, torch, constriction.
- 21 behavioral tests (`src/tac/tests/test_pr110_payload_entropy_recode.py`), NO FAKE per the
  5 forbidden classes (round-trip real R3 archive, assert byte-identical raw reconstruction,
  assert member smaller, assert selector/DQS1 preserved).

## Lossless-recode proof (fail-closed gate, PASSED)

| layer | result |
|---|---|
| decoder raw streams (joined) | byte-identical sha `83598024…` |
| latent raw payload | byte-identical sha `c760cab8…` |
| sidecar (607 B) | byte-identical sha `6c2946e3…` |
| state_dict (torch tensors) | torch-equal to R3 brotli decode |
| latents (torch) | torch-equal |
| **full decoded raw (1200 frames, 3,662,409,600 B)** | **recoded sha `dacf6b33…` == R3 sha `dacf6b33…` (BYTE-IDENTICAL)** |

Decode-parity proven `[macOS-CPU advisory]`; it is hardware-independent (CPU-pinned inflate,
IEEE-exact float64 ctx tables + deterministic torch reshape). Identical pixels ⇒ d_seg/d_pose
unchanged by construction ⇒ the win is rate-only. Proof:
`experiments/results/pr110_payload_entropy_recode_20260610/decode_parity_proof.json`.

## Byte / score reconciliation

| section | R3 frontier | recoded | delta |
|---|---:|---:|---:|
| decoder blob | 162,127 (brotli) | 161,104 (ctx range) | **−1,023** |
| latent blob | 15,387 (LZMA1) | 15,070 (ctx AR) | **−317** |
| CTXR header | (implicit) | 14 | +14 |
| FECa selector | 222 | 222 | 0 (verbatim) |
| DQS1 tail | 42 | 42 | 0 (verbatim) |
| sidecar | 607 | 607 | 0 (verbatim) |
| **member `x`** | **178,395** | **177,069** | **−1,326** |
| **archive.zip** | **178,495** | **177,169** | **−1,326** |

Score = `100·d_seg + sqrt(10·d_pose) + 25·bytes/37,545,489`. With R3's distortion
(d_seg=0.00055978, d_pose=2.942e-05) unchanged:

| | d_seg | d_pose | archive B | score |
|---|---|---|---|---|
| R3 frontier (pointer) | 0.00055978 | 2.942e-05 | 178,495 | **0.19198275** |
| **recoded R3** | 0.00055978 | 2.942e-05 | **177,169** | **0.19109982** (projected) |
| PR #112 (mattneel) | 0.00056023 | 2.943e-05 | 177,136 | 0.19112577 |

**Projected recoded R3 = 0.19109982** beats **PR #112 (0.19112577) by −0.00002594** AND our
frontier **(0.19198275) by −0.00088293**.

### Reconciliation note (corrects the intake's framing)

The intake projected ~177,114 B (claiming R3 saved 22 B on the selector). That was WRONG: R3's
FECa selector (222 B) + DQS1 tail (42 B) = 264 B is actually **larger** than PR #112's plain FEC6
selector (248 B, no DQS1). R3 spent MORE selector bytes to win on **distortion** (d_seg/d_pose
both slightly lower than PR #112). So the recoded R3 archive is **+33 B larger** than PR #112,
but R3's distortion advantage more than compensates: the head-to-head win is distortion-driven,
not pure-rate. The mechanism is sound; the byte arithmetic differs from the intake's optimistic
projection.

## Verdict — CONFIRMED `[contest-CPU]` (full 600-sample Modal eval)

Modal CPU auth eval (app `comma-auth-eval-cpu`, call_id `fc-01KTRAYS68F3S0YWFT0CX35HDG`,
validation `passed: true`, archive sha `b46897267ded…` consumed via
`archive.zip → inflate.sh → upstream/evaluate.py --device cpu`, 600 samples):

| metric | recoded R3 `[contest-CPU]` | R3 frontier | byte-identical? |
|---|---|---|---|
| avg SegNet dist | 0.00055978 | 0.00055978 | YES (lossless confirmed) |
| avg PoseNet dist | 0.00002942 | 0.00002942 | YES (lossless confirmed) |
| archive bytes | 177,169 | 178,495 | −1,326 |
| **canonical_score** | **0.19109982** | 0.19198275 | — |

**Recoded R3 `[contest-CPU]` = 0.19109982** (recomputed from components):
- beats **PR #112 (mattneel) 0.19112577** by **−0.00002594**
- beats our **R3 frontier 0.19198275** by **−0.00088293**

The empirically-measured d_seg / d_pose are byte-identical to the R3 frontier, confirming the
recode preserved distortion exactly (the lossless-recode guarantee, now validated at the score
level, not just the decode-parity level). The win is rate-only. Canonical frontier pointer
updated: `our_local_frontier_contest_cpu.score = 0.19109982419209975`, sha `b46897267ded…`,
177,169 B, `architecture_class = lane_pr110_payload_entropy_recode_20260610`.

## Submission readiness (BOTH-axis gate per CLAUDE.md "Submission auth eval")

Recoded archive `[contest-CPU]` stamp lands from this eval. A paired **`[contest-CUDA]`** eval is
still required before any contest-PR submission (the dual-axis non-negotiable). Submission packet
+ `scripts/pre_submission_compliance_check.py --contest-final --strict` prepared but NOT submitted
— flagged FRONTIER-CANDIDATE for operator submit decision. Attribution credits PR #101 (@SajayR),
PR #95 arch, PR #98 channel bias, PR #110 (@adpena = us) selector/inflate chain, and **PR #112
(mattneel) entropy coder technique** (mirroring their transparency).

## Provenance

- Recoded archive sha256 `b46897267ded…`, member `x` sha256 `5e781e8e…`, 177,169 B.
- R3 frontier archive sha256 `1ccae18d…`, 178,495 B (canonical pointer source).
- All build primitives in-tree (no rebuild of constriction or the FP11 grammar). Bulky 3.66 GB
  raws deleted after sha capture (disk hygiene).
