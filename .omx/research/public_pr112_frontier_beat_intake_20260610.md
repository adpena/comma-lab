# Public-frontier intake — PR #112 `rhnerv_comma` (mattneel, claimed 0.191126 [contest-CPU external])

**Date:** 2026-06-10
**Subagent:** `public_pr_frontier_beat_intake_20260610`
**Lane:** `public_pr112_intake_replay_20260610`
**Operator alert:** the latest public PR claims to beat our CPU frontier — "beaten us at our own game."
**Verdict headline:** TRUE, and **literally at our own game** — PR #112 is a *pure lossless entropy
re-coding of OUR merged PR #110 payload*. Decoded video is **byte-for-byte identical** to PR #110
(proven below). The entire win is **rate-only: −1,381 archive bytes → −0.00091 score**.

---

## 1. PR metadata (intake ledger)

| field | value |
|---|---|
| PR number | **#112** |
| title | `rhnerv_comma submission (0.191)` |
| author | **mattneel** (Matthew Neel) |
| URL | https://github.com/commaai/comma_video_compression_challenge/pull/112 |
| head SHA | `bdf4068f1e395d6d09764d2391632b15e38a96c8` |
| created | 2026-06-10T06:00:58Z |
| state | OPEN (eval workflow not yet maintainer-triggered) |
| claimed score | **0.191126** `[contest-CPU external]` (fork-CI ubuntu-latest report.txt) |
| archive URL | `…/releases/download/rhnerv-comma-v1/archive.zip` |
| archive sha256 | `dd4f3899b91f5b59df90b4bf4fc4d903099a286548339f5f65ff91e4b8146aa4` ✅ VERIFIED |
| archive bytes | **177,136** ✅ VERIFIED (downloaded, sha + size exact match) |
| claimed d_seg | 0.00056023 |
| claimed d_pose | 0.00002943 |
| claimed rate | 0.00471790 |
| GPU required? | No — CPU-pinned decode, < 3 min on 4 cores |
| lineage | PR #101 (@SajayR) content; PR #110 (@adpena = **us**) selector + inflate chain; PR #95 arch; PR #98 channel bias |

Intake artifacts (detached, NOT in working tree):
`experiments/results/public_pr112_intake_20260610/` (source/, archive/, replay_work/).

---

## 2. Score decomposition — WHERE the win comes from

| component | PR #110 (ours, merged) | PR #112 | delta | routes to |
|---|---|---|---|---|
| d_seg | 0.00056029 | 0.00056023 | ~0 (6e-8, reporting noise) | — |
| d_pose | 0.00002943 | 0.00002943 | **0** | — |
| archive bytes | 178,517 | **177,136** | **−1,381** | **RATE** |
| score | 0.192051 | **0.191126** | **−0.000925** | 100% rate |

vs our **current pointer CPU frontier** (178,495-byte `pr110pp_r3` R3 candidate, pointer score 0.19198275):
PR #112 is **−1,359 bytes / −0.00091 rate**. (My recompute of the R3 frontier with PR110's distortion is
0.192037; the pointer's 0.19198275 implies R3 carries a marginally different distortion — minor
reconciliation note, does not change the routing.)

**The single routing fact:** their entire advantage is rate. d_seg and d_pose are *unchanged* because the
decoded pixels are identical. This is a payload-entropy attack, not a fidelity attack.

---

## 3. Bit-level anatomy

### ZIP grammar (identical to ours)
Single member `x`, `ZIP_STORED`, no ZIP compression — the payload IS the entropy-coded blob (177,036 B
member + ZIP overhead → 177,136 B archive). Same single-member-stored grammar as PR #110.

### Container (the innovation — `codec_ctx.py`)
7-byte header: `u8 version|coder-bitmap` + `u24 dec_len` + `u24 lat_len`, then 3 sections + verbatim sidecar.

| section | PR #110 (info-identical) | PR #112 | delta | coder change |
|---|---:|---:|---:|---|
| decoder weights | 162,164 | **161,104** | **−1,060** | 7 Brotli streams → per-tensor adaptive 256-ary range coding (constriction ANS), geometric-primed priors, 2 B/model header |
| latents | 15,387 | **15,070** | **−317** | raw-LZMA1 → per-dim causal AR-prediction + discrete-Gaussian residual range coding |
| selector (FEC6) | 249 | **248** | **−1** | fixed-Huffman → adaptive 16-ary AC (was already at entropy) |
| sidecar | 607 | 607 | 0 | kept verbatim (measured < 7 B from optimal) |
| framing | 10 | 7 | −3 | tighter container header |
| **member** | **178,417** | **177,036** | **−1,381** | |

### What they did differently (the three techniques)
1. **Decoder weights (−1,060, the big one):** the INT8 weight byte-streams are **memoryless given tensor
   identity** — they report "Brotli q11 sits ~950 B above per-tensor order-0 adaptive cost, and every
   conditional context we tried (order-1, kernel-position, stored-axis neighbors, DeepCABAC-style
   binarization) loses." So the win is NOT a smarter context model — it is replacing Brotli's
   format/dictionary overhead with a tight per-tensor adaptive order-0 range coder. Per-model params
   (geometric decay ρ, strength M∈{4,16,64,256,1024}, increment, floor ε) chosen by exact simulated code
   length, transmitted in 2 B/model. 4 tiny tensors share one model. fp16 scale **high** bytes (redundant
   exponent cluster) are entropy-coded; low bytes raw. **IEEE-exact float64 table construction** → encoder
   and decoder build bit-identical probability tables on any platform (the key to lossless determinism).
2. **Latents (−317):** quantized latent codes have **negative** autocorrelation ≈ −0.45 → per-dim AR(1) on
   own deltas + optional lag-2 + up to 4 already-decoded cross-dims (integer-quantized least-squares
   coefficients) + discrete-Gaussian residual models (precomputed `Q_TABLE`, no `exp()` at decode).
3. **Selector/framing (−4):** adaptive AC on FEC6 modes + tighter container.

### Inflate chain (after reconstructed bytes) = OUR PR #110 chain, VERBATIM
`inflate.py` reconstructs PR #101's exact post-Brotli decoder streams + post-LZMA latent payload + PR #110's
exact 249-B FEC6 selector wire, then runs the **identical** fec6 chain: HNeRVDecoder, 16-pair batches,
bicubic 874×1164 `align_corners=False`, #98 channel biases (frame0 R−1/B−1, frame1 G−1), clamp/round, FEC6
K=16 selector (identical mode-IDs + Huffman code bits), uint8 NHWC. CPU-pinned. Inflate-deps: numpy, torch,
constriction (all in harness base env).

---

## 4. Replay verdict (decode-parity proof) `[macOS-CPU advisory]`

Ran PR #112 `inflate.py` on the real verified archive member, and PR #110 `inflate.py` on PR #110's archive,
on the same machine. Both produced 3,662,409,600-byte raw (1200 frames):

```
PR112 decoded 0.raw sha256: d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c
PR110 decoded 0.raw sha256: d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c
cmp: IDENTICAL (byte-for-byte)
```

**Confirmed:** PR #112 decode == PR #110 decode, byte-for-byte. The lossless-recode claim is true; pixels are
identical ⇒ d_seg/d_pose identical ⇒ the −0.00091 win is entirely the −1,381-byte rate reduction. Decode
parity is hardware-independent (deterministic float64 tables + CPU-pinned inflate); the claimed score
recomputes exactly (0.19112577 vs README 0.191126). Proof JSON:
`experiments/results/public_pr112_intake_20260610/replay_work/decode_parity_proof.json`.

The fork-CI report.txt is an ubuntu-latest (= contest GHA runner) `[contest-CPU external]` row; it remains
`external` until WE replay on our own contest-compliant path — but for a *lossless re-coding with proven
byte-identical decode and an exact byte count*, the score is deterministic and a paired Modal CPU replay
would only re-confirm 0.191126. (Modal budget preserved; paired replay deferred unless operator wants the
formal `[contest-CPU]` stamp — see §6.)

---

## 5. The leapfrog analysis — WE ALREADY IDENTIFIED THIS; WE NEVER MATERIALIZED IT

This is the structural finding (the "research-substrate trap" / orphan-signal failure mode per CLAUDE.md):

- Our **byte-shaving campaign** (`.omx/research/byte_shaving_campaign_master_gradient_..._portfolio_20260523Z.json`)
  flagged the **exact same decoder-weight byte spans** (`mg_byte_span_0162171…0178417`) as
  `materializer_target_kind_required: byte_range:entropy_recode`, candidate_saved_bytes 16,246 (optimistic UB).
  It sat **BLOCKED**: `master_gradient_byte_ranges_are_planning_coordinates_only` /
  `requires_archive_grammar_mapping_before_materialization`. We had the planner signal and never built the
  materializer that maps spans → PR101 tensor schema → re-codes → re-packs.
- We **already have every primitive PR #112 used**, in-tree and tested:
  - `constriction` is installed in our venv.
  - `tac.pr103_arithmetic_codec` — constriction `RangeEncoder` merged-AC + per-tensor `Categorical`
    histograms (PR103 L30 technique). PR #112's selector/static path is a strict subset.
  - `tac.shared_pmf_model` — "shared model across tensors chosen by exact cost" (== PR #112's
    `SHARED_MODEL_TENSORS={7,5,1,3}` idea).
  - `tac.arithmetic_qint_codec`, `tac.lossless.range_coder`, `tac.lossless.frequency_coder` — static + adaptive AC.
  - `tac.hnerv_decoder_recode` — "Planning-only structural recode probes" (the exact PR101 `FIXED_STATE_SCHEMA`).
  - PR #110's own `codec.py` exposes `decode_decoder_compact` (brotli→7 raw streams) and
    `decode_latents_compact` (LZMA→raw latents) — the exact inverse of what PR #112 re-codes.
- The ONE thing PR #112 added beyond our PR103 static-AC port: an **adaptive geometric-primed per-tensor
  model with grid-searched (ρ,M,inc,ε)** that beats static-histogram AC by the extra margin, plus the
  IEEE-exact-float64 deterministic table discipline for cross-platform losslessness.

### Top-3 leapfrog moves (ranked by predicted ΔS / cost)

**MOVE 1 — Absorb-and-recode our R3 frontier (DOMINANT; do this first).**
Apply PR #112's per-tensor adaptive recode to the decoder/latent content of our **R3 candidate**
(`pr110pp_r3_onhost_mode_table_20260610`, 178,495 B). R3 already shaves the **selector/framing** axis 22 B
below PR #110 via the on-host mode-table machinery — an axis PR #112 did NOT touch (their selector is the
plain FEC6 248 B). The two savings are on **orthogonal sections** (R3 = selector/framing; PR #112 =
decoder+latents), so they compose.
  - Predicted: decoder −1,060 + latents −317 (transfer cleanly — same PR101 INT8 weights/latent codes) on top
    of R3's −22 selector ⇒ ~**177,114 B** ⇒ score **~0.191117**, **beating PR #112 by ~+0.0000086** and our
    frontier by **−0.00092**. Pixels unchanged ⇒ d_seg/d_pose unchanged ⇒ zero fidelity risk.
  - Cost: ~$0 build (all primitives in-tree) + ~1 paired Modal CPU+CUDA replay (~$0.3) for the formal stamp.
  - Why we win the head-to-head: R3's selector axis is genuinely ours; PR #112 left it on the table.
    Re-deriving R3's selector via our on-host noise-floor tables is exactly the kind of hand-tuning their
    plain-FEC6 selector can't match.

**MOVE 2 — Build the canonical `pr110_payload_entropy_recode` materializer (closes the orphan, compounds).**
Promote the BLOCKED byte-shaving span to a real materializer: extract via `decode_decoder_compact` /
`decode_latents_compact`, recode with `pr103_arithmetic_codec` + `shared_pmf_model` + a NEW adaptive
geometric-primed per-tensor model (the only missing primitive), re-pack, byte-close, paired-eval. This is
the reusable surface that makes the win durable + lets us keep stacking entropy gains (their README admits
the sidecar is "< 7 B from optimal" and the selector "already at entropy" — but the **decoder order-0 floor**
is the live lever, and we can also test whether OUR distinct R3/PSV3 decoder weights have *more* recode
headroom than PR101's).
  - Predicted: ≥ MOVE 1's −1,377 B on any PR101-grammar archive; reusable across R3/PSV3/future lanes.
  - Cost: ~$0 (in-tree) + tests + 1 paired replay.

**MOVE 3 — Re-derive the latent/selector payload with our own tools (true leapfrog beyond pure recode).**
PR #112 recodes payloads *they inherited*. We can *regenerate* them better:
  - Latents: their AR(1)+cross-dim LS is a generic predictor. Our **null-basis / preimage compiler**
    (10–19.5%) and **invisibility basis** (22.7% / 80.67%) can push latent codes toward the SegNet/PoseNet
    null space *before* entropy coding — shrinking the residual entropy the coder must carry, a lever PR #112
    structurally cannot reach (it is locked to bit-exact reproduction of PR101's latents).
  - Selector: our **margin/cone targeting + composition algebra + frame-1 Class-3 atoms** can select
    per-pair modes that the plain FEC6 menu can't express, lowering distortion *and* feeding a tighter
    adaptive-AC selector stream.
  - Predicted: this moves the d_seg/d_pose axes (not just rate) — the only path to a *structural* lead rather
    than a byte-delta lead. Higher EV, higher cost (training/search), DEFER behind MOVE 1's quick ship.
  - Cost: substrate-engineering; queue after MOVE 1 ships (race-mode rigor: ship the smallest credible
    bolt-on first).

---

## 6. Recommended next actions (operator-routable)

1. **MOVE 1 now** (smallest credible bolt-on, beats PR #112): build the per-tensor adaptive recode, apply to
   R3, byte-close, run ONE paired Modal CPU+CUDA replay for the formal `[contest-CPU]`/`[contest-CUDA]` stamp,
   then `gh pr create` per `docs/submission_template.md`. Attribution must credit PR #101/#95/#98 + PR #112's
   coder technique (mirror their transparency).
2. **MOVE 2** as the durable materializer (closes the orphaned byte-shaving signal so the win is reusable).
3. **MOVE 3** queued as the structural (distortion-axis) leapfrog once rate parity is re-established.
4. Optional: formal paired Modal CPU replay of PR #112's archive (~$0.3) for an internal `[contest-CPU]`
   stamp — low value given the proven byte-identical decode + exact byte count, but available.

## 7. Custody / provenance

- All intake in detached `experiments/results/public_pr112_intake_20260610/`; NOTHING checked into the
  working tree of the contest clone.
- PR #112 archive sha256 + size VERIFIED against the release.
- Decode-parity proof recorded; 2×3.66 GB rebuildable raw decodes deleted after sha capture (disk hygiene).
- Claimed score stays `[contest-CPU external]` (fork-CI) until our own contest-compliant replay; the
  deterministic recompute (0.19112577) + byte-identical decode make the external claim high-confidence.
- Lane `public_pr112_intake_replay_20260610` claimed; close on memo landing.
