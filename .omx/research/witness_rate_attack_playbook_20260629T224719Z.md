# WITNESS RATE-ATTACK PLAYBOOK — the v2 witness `archive.zip` COUNTED payload

**UTC** 2026-06-29T22:47:19Z · **authority** `[$0 CPU design-audit / advisory]` · **pointer UNMOVED 0.19110**
**score_claim** false · **promotable** false · **ready_for_exact_eval_dispatch** false · this is a **PLAN (MEANS), not the end.**
A playbook lowers nothing. It pays ONLY at byte-close on a real representation, ratified by a paired contest-CPU/CUDA
exact eval. Every rate-delta below is labelled **GROUNDED** (cite the artifact) / **ESTIMATE** (derived band) / **UNGROUNDED**.

**Score law** (`upstream/evaluate.py:92`): `S = 100·d_seg + √(10·d_pose) + 25·B/N`, `N = 37,545,489`. NO time term.
Only `archive.zip` bytes `B` are counted (`evaluate.py:63`); inflate.py/inflate.sh code is FREE (rule-118), but
LEARNED/video-derived artifacts (weights, tables) MUST be in `archive.zip` and ARE counted (README:118 — incl.
PoseNet/SegNet). **Byte price ≈ 6.66e-7 score/byte** (GROUNDED, bolton memo). So 1,500 B ≈ 0.001 S.

**SEAM:** this memo owns RATE-TECHNIQUE → component mapping + classification + ranking. A sister agent owns the broader
config/arbitrariness audit; the **§4 FLAGGED training-time list** is the handoff it must pick up into the GPU-run config.

---

## §0. The v2 witness COUNTED components (what goes in archive.zip)

| Cx | Component | Nature | Grounded size (n→600) | Source |
|---|---|---|---|---|
| **C1** | canonical-scene descriptor (static base partition / keyframe) | int/label or RGB | seg_core lossless 895.7 B/frame; amortized base ~10–20 KB | composition map #11 (GROUNDED structure; amortized = ESTIMATE) |
| **C2** | pose stream (6-DOF × 600 pairs) | float→int | 7,200 B raw / <5,000 B zlib; →~hundreds B via #140 | `scorer_targets.py` docstring (GROUNDED raw); #140 PARTIAL (ESTIMATE) |
| **C3** | lane SDF / boundary contour descriptor | int coeffs / point-set | lossy poly deg2 2,288 B (6 px, re-incurs d_seg); deg6 5,247 B; honest point-set 46,516 B; lossless 55,650 B | budget_gate M3 (GROUNDED) |
| **C4** | bulk-jitter annulus store (waterfill output) | sparse position+symbol set | FULL margin-keyed store **177,926 B → rate 0.1185** (iid upper bound); earlier partial-waterfill MODEL 0.062–0.084 | budget_gate M4 (GROUNDED, latest); FEED-kd/kf (ESTIMATE, earlier/partial) |
| **C5** | movables residual | sparse | small (d_seg ~0.0008 contribution) | grok memo (ESTIMATE) |
| **C6** | one canonical margin field (smooth) | int tensor | one smooth field (small, brotli-friendly) | composition map #10 (ESTIMATE) |
| **C7** | trained lane-survival residual weights | NN weights | **the bulk** — FUTURE GPU output, the binding wall | not yet measured (UNGROUNDED) |

**Reference archives (GROUNDED):** base_ch20 existence proof = **89,244 B** (rate 0.0594); CPU frontier = 177,169 B
(rate 0.1185, lane_pr110); L13 non-RGB byte-close = 72,217 B, rate **−59%** vs frontier (8-pair parity, prior work).

**LOAD-BEARING STRATEGIC FACT (read before the matrix).** Two components dominate the rate budget: **C4** (bulk-jitter)
and **C7** (trained weights, ~91% of a NN archive). The M4 measurement is decisive: shipping C4 as an **explicit counted
dither store costs rate 0.1185 ≈ the WHOLE PR95 archive** → `S ≈ 0.26`, busts sub-0.15 by itself. **No post-hoc coder
fixes this.** The only route that closes is folding the jitter INTO the trained generator C7 (PR95 reaches d_seg 6e-4 at
~118 KB because a learned per-pair latent expands into the partition *including* the jitter — the jitter rides free with
content). So the #1 rate-attack on C4 is a **training-time move (don't store it explicitly)**, not a coder. The post-hoc
coders below are the **FINISHING KIT** that harvests the last fraction once C7 has descended near the frontier's
distortion; they cannot rescue a payload whose representation is wrong.

---

## §1. TECHNIQUE × COMPONENT MATRIX

Class: **PH** = POST-HOC (byte-close, no retrain) · **TT** = TRAINING-TIME (must be in GPU config or LOST) · **FE** = FRAME-EXPLOIT (decode-side).

| Technique | Class | Attacks components | In-tree? |
|---|---|---|---|
| L21 per-tensor byte-maps (zig/negzig/twos/off) | PH | C7, C6, C2, C1 (any int tensor) | yes (PR101 codec) |
| L22 conv4 storage perms | PH | C7 (conv weight tensors) | yes (PR101) |
| L23 split brotli streams (group by distribution) | PH | C7, any multi-tensor payload | yes |
| L24 raw-LZMA FORMAT_RAW (strip headers) | PH | C3, C2, C1 (contiguous streams) | yes (PR101 latent) |
| L25 temporal-delta uint8 + prefix-sum | PH | **C2 pose stream (PRIMARY)**, C3 across frames, C4 recurring positions | yes (PR100/101) |
| L26 canonical-Huffman length-ranked | PH | C7 code lengths, any symbol alphabet | **yes** `encode_huff_length_rank` |
| L29 fp16 per-tensor scales (int8 dequant) | PH | C7, C6 (the quant container itself) | yes |
| L30 range/arithmetic coding (constriction.Categorical) | PH | **C7, C4 symbols, C3, C2** (all int sections) | **yes** `RangeEncoder`/`RangeDecoder`; PR#112 `ctx_range_coder` SHIPPED |
| L31 colex-rank for sparse positions | PH | **C4 annulus positions (DIRECT), flip positions, C5 movables** | **yes** `encode_combination_colex` |
| L32 brotli-q11 max | PH | all sections | yes (deploy-time) |
| entropy-penalized loss (rate term in training) | **TT** | C7, C3/C4 representation entropy | partial (loss hooks) |
| latent-structure-inducing regularizer (#110) | **TT** | C7 latent code | research |
| variable-grid QAT (#111) | **TT** | C7 weights | research |
| int5 / LSQ / per-channel / outlier (#147) | **TT** | C7 weights | partial |
| weight-entropy reg (NVRC / NeuroQuant) | **TT** | C7 weights | research |
| T5 null-space-as-training-CONSTRAINT (bolton top) | **TT** | C7 (puts error in certified-invisible null) | basis built; constraint unbuilt |
| WRQ-as-QAT (score-aware bit alloc in loss) | **TT** | C7 | post-hoc harness built; QAT form unbuilt |
| T2 warp-residual frame0 head | **TT** (arch) | C1/scene (halves frame0 cost) | warp primitive exists; head unbuilt |
| L27 per-pair single-dim correction sidecar | FE | d_seg + d_pose (ships tiny sidecar) | yes (PR100/101) |
| L28 zero-byte channel postprocess | FE | d_seg + d_pose (0 bytes) | yes (PR98) |
| FECa / DQS1 selector exploits (#128) | FE | d_pose @ 0 d_seg (frame0 is SegNet-blind) | yes (selector) |

---

## §2. PER-TECHNIQUE CLASSIFICATION + GROUNDED-OR-ESTIMATED RATE DELTA

### POST-HOC (apply at byte-close; lossless unless noted; no retrain)

- **L30 range/arithmetic coding (= R1/R2 `ctx_range_coder`).** Attacks C7 weights + C4 dither symbols + C3 + C2.
  **GROUNDED:** on the FP11 frontier R1 measured **−1,023 B** (decoder 162,127→161,104) + R2 **−317 B**
  (latents 15,387→15,070). The coder primitive is grammar-agnostic and in-tree; only a **v2-grammar materializer**
  (~half-day) is missing. This is the single biggest *lossless* lever and the floor every structure move feeds.
- **L31 colex-rank for sparse positions.** Attacks **C4** (annulus/dither positions) DIRECTLY + flip set + C5.
  **GROUNDED (mechanism, in-tree):** `log2 C(N,K)` is the information-theoretic floor for an unordered position set; the
  coder is built (`encode_combination_colex`). **CAVEAT (honest):** the M4 store (177,926 B) is already a margin-keyed
  *entropy* estimate ≈ the colex bound, so colex is the *correct/cheap* way to hit that floor, **not a further win below
  it**. The real reduction on C4 comes from TEMPORAL correlation of recurring boundary flips (L25/T1-style) feeding colex
  a smaller set — and ultimately from not storing C4 explicitly (§0). PR101 `SIDECAR_NOOP_INFER_RANK_LEN = 3 B` is the
  GROUNDED anchor that position-sets compress to near-nothing when sparse.
- **L25 temporal-delta uint8 + prefix-sum.** Attacks **C2 pose stream (PRIMARY)** — a single near-stationary drive →
  6-DOF is a smooth trajectory; delta + prefix-sum collapses its entropy. **GROUNDED mechanism** (PR100/101 latent coding);
  **ESTIMATE magnitude** on C2 (C2 is only ~5 KB raw, so absolute B is small but near-free; compounds with #140 low-rank).
- **S12 resize-null preimage** (pixel DOF, AGNOSTIC, CERTIFIED 0 distortion). Attacks the coded-frame bytes feeding C1/C7.
  **−10 to −19.5% of coded frame bytes**, GROUNDED-ish (basis landed + certified); run BEFORE L30 as a force-multiplier
  (lower-entropy input). Note: applies only if the v2 witness ships decoded RGB frames into the archive.
- **L24 raw-LZMA FORMAT_RAW.** Attacks C3/C2/C1 contiguous streams. Strips format headers (~tens of B/stream).
  **ESTIMATE** (small, lossless; PR101 used it for the 15,387 B latent blob).
- **L21/L22/L23/L26/L29 brotli-friendliness bundle** on C7 (+C6/C2). Per-tensor byte-maps + conv-perms + split-brotli +
  canonical-Huffman length-rank + fp16-scales. Each **−100s B**, lossless, additive on **disjoint** tensors; **ESTIMATE**
  aggregate (PR101 shipped all of them; individually small). fp16-scales (L29) is the quant container (~56 B/28 tensors),
  not a "win" but the GROUNDED structural cost.
- **L32 brotli-q11.** All sections. **−5–10% on small payloads**, GROUNDED (offline cost free at deploy); zero-effort default.
- **WRQ score-aware per-tensor weight requant** (lossy on recon, in-cell). Attacks C7 (decoder = ~91% of bytes →
  largest single post-T1 lever). **UNGROUNDED magnitude** (needs its own exact-authority sweep); high ceiling, higher effort.
- **T9 global perm + cross-tensor cluster.** C7. **−100 to −500 B**, ESTIMATE (lossless, near order-0 floor).
- **T4 order-1 selector / RLE.** selector stream. **−50 to −100 B**, ESTIMATE (low EV; confirm-the-bound).

### TRAINING-TIME (MUST be in the GPU-run config or it is LOST — see §4)

- **entropy-penalized loss / weight-entropy reg (NVRC/NeuroQuant).** C7. Makes weights compressible by construction →
  multiplies every PH coder above. **ESTIMATE** (band; the make-compressible prior, not a post-hoc add).
- **T5 null-space-as-training-CONSTRAINT.** C7. **−0.01 to −0.04 compounding** (ESTIMATE; bolton "strongest synergy").
  Train error into the certified resize-null → certified-free error + lower-entropy residual for L30. Top TT promotion.
- **int5/LSQ/per-channel/outlier (#147) + variable-grid QAT (#111) + WRQ-as-QAT.** C7. Byte-minimal-for-the-scorer
  weights by construction (scorer tolerates far more weight error than recon). **ESTIMATE.**
- **latent-structure regularizer (#110).** C7 latent → unlocks T1 cross-pair dedup post-hoc. **ESTIMATE.**
- **T2 warp-residual frame0 head (arch).** C1/scene. **−0.01 to −0.03** by regenerating frame0 (SegNet-blind) from
  frame1+pose. **ESTIMATE.**

### FRAME-EXPLOIT (decode-side; mostly 0-byte; attack distortion, indirectly free rate budget)

- **L27 per-pair single-dim correction sidecar.** **−0.001 to −0.003 S** (GROUNDED anchor: PR100/101 "substrate-ceiling →
  medal-class jump"). Ships a tiny counted sidecar (~1.2 KB) but the d_seg/d_pose win dwarfs the bytes. Re-run on the witness.
- **L28 zero-byte channel postprocess.** **−0.0001 to −0.0005 S at 0 archive bytes** (GROUNDED, PR98/L28). Re-fit the
  3 bias constants on the v2 render-vs-GT. Free; do early.
- **FECa/DQS1 selector (#128).** d_pose at 0 d_seg cost (frame0 SegNet-blind by construction). Small bytes; GROUNDED (selector).

---

## §3. THE RANKED PLAYBOOK (rate-delta-per-effort, post-hoc lossless first)

**Sequencing law:** mutually-orthogonal LOSSLESS moves on DISJOINT sections are proof-by-construction additive (identical
pixels → identical d_seg/d_pose → byte savings sum). Moves on the SAME section SUBSUME each other (do NOT double-count).

1. **L30 range/arithmetic coder on every int section (C7+C4+C3+C2).** GROUNDED (R1 −1,023 / R2 −317 measured); coder
   in-tree; effort = one v2-grammar materializer (~half-day). **The floor every other move feeds. Build first.**
2. **S12 resize-null preimage** (if archive ships RGB frames). GROUNDED −10–19.5% of frame bytes; certified 0-distortion;
   run BEFORE L30 (force-multiplier). High EV, low effort, AGNOSTIC.
3. **L31 colex-rank on C4/C5 position sets + L25 temporal-delta on C2/C4.** GROUNDED mechanism, coders in-tree, cheap.
   Caveat: these hit the entropy floor, they don't beat it — their job is to feed L30 a correctly-structured stream.
4. **L21/L22/L23/L26/L29/L24 brotli-friendliness + raw-LZMA bundle on C7.** ESTIMATE −100s B each, additive on disjoint
   tensors; one batch at byte-close. Moderate aggregate, low effort.
5. **L32 brotli-q11.** Trivial, free at deploy. Always on.
6. **WRQ score-aware weight requant on C7.** UNGROUNDED magnitude but C7 is ~91% of bytes → highest *ceiling*; higher
   effort (exact-authority sweep). Run once C7 exists and has descended near-frontier.
7. **T9 perm/cluster (−100–500 B) + T4 order-1 selector (−50–100 B).** Confirm-the-bound; lowest EV.

**Distortion finishers (interleave; mostly free):** L28 (0-byte, do early) → L27 per-pair correction (re-run on witness)
→ FECa selector for d_pose.

**Order-of-magnitude harvest (NOT a sub-0.15 mover by itself):** the analogous lossless stack on the FP11 frontier was
**≈ −0.005 to −0.008 on the rate axis** (T1-dominated, GROUNDED band, bolton memo) + the WRQ/distortion finishers.
This is the **finishing kit**, not the breakthrough. **The breakthrough is C7 + the C4-fold (training-time, §4).**

---

## §4. ⚑ FLAGGED — TRAINING-TIME TECHNIQUES THAT MUST GO INTO THE GPU-RUN CONFIG (or they are LOST)

These cannot be applied post-hoc. If the GPU run ships without them, they are gone until a re-train. **Hand these to the
sister config/arbitrariness-audit agent for inclusion in the witness curriculum config.** Ranked by expected impact:

1. **⚑ FOLD C4 (bulk-jitter) INTO C7 — do NOT ship an explicit dither store.** The single most important config decision.
   Explicit C4 store = rate 0.1185 → S≈0.26 (GROUNDED, M4). The jitter must be emitted by the trained generator from a
   compact per-pair code (PR95 pattern). This is a representation choice baked at training time; losing it dominates everything.
2. **⚑ entropy-penalized / weight-entropy-regularized loss (NVRC/NeuroQuant) on C7** — make the decoder weights
   compressible by construction; multiplies every §3 post-hoc coder. ESTIMATE band.
3. **⚑ T5 null-space-as-training-CONSTRAINT** — train error into the certified resize-null. −0.01 to −0.04 compounding
   (ESTIMATE); bolton "strongest synergy". Highest single TT ceiling after the C4-fold.
4. **⚑ score-aware QAT (int5/LSQ/per-channel/outlier #147 + variable-grid #111 + WRQ-as-QAT objective)** — byte-minimal
   weights for the scorer by construction (scorer tolerates more weight error than recon).
5. **⚑ latent-structure-inducing regularizer (#110)** — induces cross-pair latent redundancy so post-hoc T1 dedup can fire.
6. **⚑ T2 warp-residual frame0 head (arch)** — regenerate SegNet-blind frame0 from frame1+pose; −0.01 to −0.03 (ESTIMATE).

(L28 bias constants and L27 per-pair corrections are FE/post-hoc and re-fittable later — they do NOT need to be in the
GPU config, but they should be re-derived on the v2 render once C7 exists.)

---

## §5. HONEST CAVEATS / NO-FAKE (this is a PLAN; it moves nothing)

- **Pointer UNMOVED 0.19110.** No exact eval here. The playbook pays only when a byte-closed v2 archive returns a lower
  paired contest-CPU/CUDA row. Every delta is `[advisory]`.
- **Do NOT sum a stacked total.** Interactions are not additive: disjoint-section lossless moves sum (proof-by-construction);
  same-section moves SUBSUME (L30 vs colex on the *same* positions; T1 vs R2 on the *same* latents — keep the stronger, use
  the weaker only as the within-cluster residual coder). Coding-structure moves (L25/L31) feed the entropy coder (L30) a
  lower-entropy input; they are not independent wins added on top of it.
- **The post-hoc kit cannot fix a wrong representation.** The binding rate cost is C7 (the trained weights) and the C4-fold
  decision — both TRAINING-TIME. On a base whose intrinsic d_seg/d_pose has not descended, the entire §3 harvest is noise.
- **C4 rate discrepancy resolved:** FEED-kd/kf 0.062–0.084 = an earlier *partial-waterfill* model (only flips clearing
  break-even / annulus-localized); budget_gate M4 0.1185 = the *full* margin-keyed store (iid upper bound). Both say explicit
  C4 store is rate-expensive; temporal correlation (untested) could lower M4, but the strategic answer is §4-#1 regardless.
- **WRQ magnitude is UNGROUNDED** (needs its own exact sweep); do not quote a number for it.
- **rule-118 boundary:** the coders/rasterizers/perms/prefix-sum are FREE generic algorithm in inflate.py; the LEARNED C7
  weights + any VIDEO-derived table (C1 keyframe, C3 coeffs, C4 store) are COUNTED. Do not smuggle a video-derived per-frame
  table into inflate.py as "code" (hide-data-in-code fake).

---

## §6. DAG FEED (tight summary)

**FEED — WITNESS RATE-ATTACK PLAYBOOK ($0 design-audit, advisory, pointer UNMOVED 0.19110).** Mapped all PR95-L21–L32 +
make-compressible + frame-exploit techniques onto the v2 witness COUNTED components C1–C7. **Decisive finding:** the rate
budget is dominated by **C7 (trained weights, ~91%)** and **C4 (bulk-jitter)**; shipping C4 as an explicit counted dither
store = rate **0.1185** (M4 GROUNDED) → S≈0.26, busts sub-0.15 alone → **C4 MUST be folded into the trained generator C7
at training time, not coded post-hoc.** The post-hoc PR95 coder stack (range/arithmetic L30 = R1 −1,023 B / R2 −317 B
GROUNDED + colex L31 on positions + temporal-delta L25 on pose + S12 −10–19.5% certified + brotli-friendliness bundle) is
the **FINISHING KIT**, order **≈ −0.005 to −0.008 rate** (GROUNDED band, bolton) — harvests the last fraction once C7 has
descended near-frontier; it is NOT the breakthrough and does not stack additively across same-section moves. **Coders are
all in-tree** (`RangeEncoder`/`ctx_range_coder`, `encode_combination_colex`, `encode_huff_length_rank`); the only build gap
is a **v2-grammar materializer (~half-day)** to lift them onto the witness container. **⚑ TRAINING-TIME techniques that
MUST enter the GPU-run config or are LOST (handoff to config-audit sister):** (1) fold-C4-into-C7, (2) entropy/weight-entropy
loss, (3) T5 null-space training-constraint (−0.01..−0.04 est), (4) score-aware QAT int5/LSQ/#147/#111, (5) latent-structure
reg #110, (6) T2 warp-residual frame0. Memo: `.omx/research/witness_rate_attack_playbook_20260629T224719Z.md`.

**Primary citations:** budget_gate_overturn M3/M4 (`...20260629T215729Z.md`, GROUNDED C3/C4 bytes); capstone_synergy
composition map (`...20260626.md`, component inventory + C1/C2 bytes); bolton inventory (`...20260612.md`, R1/R2/T1/S12/WRQ +
−0.005..−0.008 band); `src/tac/scorer_targets.py` (C2 raw 7,200 B); `src/tac/codec/pr101_polymorphic.py` (colex + Huff-rank
coders in-tree); `src/tac/lossless/range_coder.py` (RangeEncoder/Decoder in-tree); CLAUDE.md L21–L32 + L27/L28.
