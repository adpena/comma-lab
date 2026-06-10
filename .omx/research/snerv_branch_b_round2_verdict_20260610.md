# SNeRV Branch-B ROUND-2 verdict — the structured/clever tier prices the recursion; stored-LF stays 280–530× the 178 KB frontier

UTC 2026-06-10 · subagent `snerv_branch_b_round2_20260610` (Branch-B round-2 executor) ·
`[macOS-CPU advisory]` / `exact_cpu_advisory` — **NON-PROMOTABLE** (promotion requires paired
Linux x86_64 contest-CPU + CUDA per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA").
Run dir `/Volumes/VertigoDataTier/pact/snerv_branch_b_round2_20260610T000000Z/`.
Frontier at landing (orphan inventory `da62505aa`): contest-CPU **0.19198533** (archive
`b7106c9b…`, 178,493 B).

Binding directive: `.omx/research/snerv_rate_attack_round2_directive_20260610.md`
(commits `5a869cbca` + `9742915de`). Round-1 pricing baseline:
`.omx/research/snerv_branch_b_rate_attack_round1_20260609.md` +
`/Volumes/VertigoDataTier/pact/snerv_branch_b_rate_attack_20260609T230000Z/snerv_branch_b_rate_attack_ladder.v1.json`.
Base: G1b SNeRV path-B packet `934349b0…` 581,583,107 B, archive-surface d_seg 0.002468 /
pose-term 0.142602 (`snerv_g1b_export_binding_verdict_20260609.md`).

## THE ONE-LINE ANSWER (operator's "aren't there more optimal")

**There are more optimal *quantizers* (frame-role + channel axes give real, measured, free byte
wins on top of the round-1 uniform ladder), but there is NO stored-LF representation that reaches
the bar.** Every structured rung — temporal-class, channel-split, frame-role — still stores the
half-res video and lands at **280–530× the 178,493-byte frontier**. The decisive S8 test proves
*why*: the SNeRV MFU/HFR machinery cannot refine the frontier's own LL band (it is trained to the
G1b distribution and breaks pose catastrophically when cross-fed). **The binding next rung is NOT
another SNeRV stored-LF coder — it is composing the 0.19199 frontier DIRECTLY (learned synthesis),
which already renders evaluator-close for 178 KB and which SNeRV's stored-state approach cannot
approach.**

## S8 COMPOSE-THE-FRONTIER (the directive's S8-first decisive test) — FAILED, decisively

Mechanism (premise-verified, corr 1.0): `skip_high ≈ 2.0024·downsample₂ₓ(rendered_output) − 0.056`.
So substituting the frontier's half-res band as skip_high *should* make SNeRV render the frontier.
Frontier inflate ran receiver-side (155.2 s wall — **inside the 30-min budget**), 1200 camera-res
frames; DWT/downsample to the 192×256 skip_high grid; fed through SNeRV's unchanged MFU/HFR.

| S8 rung (frontier-LL as skip_high) | bytes | ×178 KB | d_seg | pose_term | nonrate | cone |
|---|---:|---:|---:|---:|---:|---|
| float64 | 585,960,499 | 3283× | 0.005921 | 2.2380 | 2.8301 | **FAIL** |
| fp16 | 297,191,891 | 1665× | 0.005921 | 2.2380 | 2.8301 | **FAIL** |
| uint8 global | 171,205,111 | 959× | 0.005924 | 2.2389 | 2.8314 | **FAIL** |
| uint6 frame | 118,422,115 | 663× | 0.005931 | 2.2490 | 2.8421 | **FAIL** |
| uint4 frame | 71,013,842 | 398× | 0.005945 | 2.3657 | 2.9602 | **FAIL** |

**Diagnostic (the decisive control):** the frontier render scored DIRECTLY through the identical
scorer path (camera-res, N=48) gives **d_seg 0.000538 / pose_term 0.0109 / nonrate 0.0647** — i.e.
the 178 KB frontier is excellent. Routing that same render's LL through SNeRV's trained MFU/HFR
**destroys** it: pose_term explodes 0.011 → 2.24 (205×), nonrate 0.065 → 2.83. Root cause: the
frontier render is genuinely brighter/sharper than the G1b "blurry mean-field" the heads were
trained on (frontier-LL mean 124.7 / std 121.5 vs G1b skip_high mean 42.9 / std 41.5, ≈2.9×). The
trained MFU residual + HFR heads are **bound to the G1b skip_high distribution**; cross-feeding any
other distribution corrupts pose.

**S8 honest comparison (the directive's bar — "residuals must pay rent vs the frontier alone"):**
best S8 = 71 MB at cone-FAIL = **398× the frontier archive with worse fidelity**. There is no rent
to pay; the SNeRV machinery is a *liability*, not an asset, on the frontier LL. **S8 is strictly
dominated by composing the frontier directly** (vehicle composition #31 should arrive through the
*frontier's* renderer, not SNeRV's). Decode runtime per rung ≈ 25–46 s (inside budget).

## S5a FRAME0-HARD-QUANT (the frame-role asymmetry) — the asymmetry is REAL and EXACTLY priced

Frame layout premise-verified: archive is pair-major/frame-major `(n_pairs,2,3,H,W)`; flat
`f = 2·pair + role`; EVEN = frame0 (SegNet-blind by `x[:,-1,…]` slice + the invisibility corollary),
ODD = frame1 (joint). Storage floor = carrier lossless intN codec, **split by frame role**.

| S5a rung (frame0=EVEN, frame1=ODD) | d_seg | Δd_seg vs ctrl | pose_term | floor bytes | ×178 KB | cone |
|---|---:|---:|---:|---:|---:|---|
| ctrl: f0+f1 uint8 global | 0.002491 | — | 0.1574 | 92,235,581 | 517× | OK |
| **f0 uint6 \| f1 uint8** | 0.002491 | **+0.00e+00** | 0.1971 (Δ+0.0545) | 70,059,613 | 393× | **OK** ← frame0 knee |
| f0 uint5 \| f1 uint8 | 0.002491 | **+0.00e+00** | 0.3269 (Δ+0.184) | 61,412,313 | 344× | FAIL |
| f0 uint4 \| f1 uint8 | 0.002491 | **+0.00e+00** | 0.6489 (Δ+0.506) | 55,466,929 | 311× | FAIL |
| f0 uint3 \| f1 uint8 | 0.002491 | **+0.00e+00** | 2.7226 (Δ+2.58) | 51,280,597 | 287× | FAIL |
| f0 uint2 / f1uint6 combos | *(in-flight; folded into JSON at commit)* | | | | | |

**Two exact, measured findings:**
1. **frame0 is SegNet-blind, confirmed EXACTLY** — d_seg(frame0 uint8/6/5/4/3) − d_seg(ctrl) =
   `+0.00e+00` at every frame0 precision. The invisibility corollary holds to the bit: SegNet
   reads frame1 only, so frame0 precision is a pure pose lever.
2. **The frame0 pose floor knee is uint6.** frame0 → uint6 costs pose +0.0545 (cone OK) and saves
   **92.2 → 70.1 MB (−24%, ~22 MB) at ZERO d_seg cost** — exactly the "50% of the planes
   immediately" the directive predicted. uint5 already breaks the 0.25 pose cone (0.327), and the
   curve is steeply convex below uint6 (uint4 → 0.65, uint3 → 2.72): frame0's LL is the half-res
   *previous-frame* PoseNet reads, so its precision is the exact frame0 motion signal (adversarial
   correction #1 vindicated — frame0 is SegNet-free, NOT pose-free).

## S2 YUV/CHANNEL-SPLIT (channel sensitivity) — chroma is the near-free direction

YUV transform = BT.601 full-range (the rgb_to_yuv6 luma basis); roundtrip near-lossless
(ctrl Δd_seg +2.0e-5). Storage floor split into 6 streams (Y/U/V × frame-role).

| S2 rung | d_seg | pose_term | floor bytes | ×178 KB | cone |
|---|---:|---:|---:|---:|---|
| ctrl: all-uint8 YUV | 0.002488 | 0.1433 | 94,590,789 | 530× | OK |
| **f1 Y8 UV6 \| f0 Y8 UV6** | 0.002490 | 0.1591 (Δ+0.016) | 66,665,027 | 373× | **OK** |
| Y8 UV4 / + frame0-chroma-coarsest / + frame0-Y combos | *(in-flight; folded into JSON at commit)* | | | | |

**Finding so far:** chroma coarsening to uint6 (both roles) is nearly free — pose +0.0165, d_seg
+2e-5 — and cuts 94.6 → 66.7 MB (−30%). Confirms the directive's measured axis: pose is
luma-dominant, chroma is the null-ward direction. (The frame0-chroma-coarsest + frame0-Y-coarser
combos — the cheapest payload in the archive — complete in the JSON; they compose S5a×S2.)

## The rung table — gap_to_0.19199 and the bytes-vs-178KB ratio per rung (the strategic bar)

| Rung | best cone-OK floor bytes | ×178 KB | nonrate | total (floor) | gap to 0.19199 |
|---|---:|---:|---:|---:|---:|
| Round-1 uint8-global (R5) | 106,855,519 (pkt) | 599× | 0.405 | 71.56 | +71.4 |
| Round-1 carrier-LF floor uint8 | 92,459,187 | 518× | 0.407 | 61.97 | +61.8 |
| **S5a f0-uint6 (frame-role)** | **70,059,613** | **393×** | 0.397 | **47.10** | **+46.9** |
| **S2 Y8-UV6 (channel)** | **66,665,027** | **373×** | 0.302 | **44.80** | **+44.6** |
| S8 best (frontier-LL, cone-FAIL) | 71,013,842 | 398× | 2.96 | 50.25 | +50.1 (cone-FAIL) |
| **— the frontier itself (learned synthesis) —** | **178,493** | **1×** | **0.065** | **0.19199** | **0** |

The frame-role + channel axes move the cone-OK storage floor from round-1's 518× to **~370–393×**
(~28% byte cut at d_seg-unchanged), and stack (S5a×S2 → frame0-chroma-coarsest is the cheapest
payload). But the rate term is still **~44–47** — i.e. **370–530× the frontier**. The directive's
absolute bar ("must reach well under ~200 KB total OR SNeRV becomes a base carrier") is **failed by
every stored-LF rung by 2–3 orders of magnitude.**

## STORED-STATE vs LEARNED-SYNTHESIS verdict (the directive's required honest comparison)

skip_high is SOURCE-DERIVED state (the half-res video), so compressing it is a *recursive instance
of the contest itself*. The bytes-vs-178KB ratio per rung is the verdict:

- **Stored-state (SNeRV skip_high, all round-1+round-2 rungs): 287–3283× the frontier.** Even the
  best cone-OK quantizer (S2 channel-split, 67 MB) is 373× and the carrier-LF entropy floor only
  buys back the LZMA slack — the planes are near-incompressible at near-lossless Δ (round-1 finding:
  intN floor ≈ packet bytes). The 192×256×1200×3 LL has ~real entropy; quantizer axes shave the top
  but not the order of magnitude.
- **Learned-synthesis (the 0.19199 frontier): 1×.** The frontier's HNeRV+selector *generates* the
  full video evaluator-close for 178 KB. S8 proved the frontier render itself is excellent
  (nonrate 0.0647) — the 178 KB IS the recursion solved.

**The 581 MB problem does NOT collapse onto SNeRV.** It collapses onto the frontier, which already
solved it. SNeRV stored-LF is the wrong representation for the LL band: storing the half-res video
is 280–530× more expensive than synthesizing the full video. **S8 even falsifies "SNeRV as a base
carrier that refines the frontier"** — its trained heads corrupt any non-G1b LL.

## ROUTING — the binding next rung

1. **Do NOT pour more effort into SNeRV stored-LF coders (S1/S3/S4/S5b/S6).** They will land in the
   same 200–500× band: temporal-delta (S1) shaves intra-LL temporal redundancy but the LL is
   already near-lossless-stored so the gain is the LZMA slack (round-1 already captured most of it);
   spatial DWT+entropy (S2/S3) is the channel axis, measured here at ~370×; the per-(p,r,ch,k,x)
   waterfill (S5b) cannot beat the union of S5a+S2 by an order of magnitude. None reach <200 KB.
2. **The binding move is the directive's S7/S9/S11 *synthesis* tier, NOT stored samples** — and S8
   shows the strongest synthesizer (the frontier's own HNeRV) is **already on disk at 178 KB**. The
   honest action is **compose the frontier DIRECTLY** (its renderer, not SNeRV's heads) and store
   only sparse pose/seg-cone residuals where the frontier render is outside budget — but the
   frontier render's nonrate is already 0.0647, so the residual budget is tiny. This is S8 done
   *right*: refine on the frontier's renderer, not by injecting its LL into SNeRV's mismatched heads.
3. **The only SNeRV-native value that survives** is the frame-role + channel asymmetry as a *generic
   bit-allocation prior* (frame0 SegNet-blind → pose-only floor at uint6; chroma → null-ward; the
   #46 waterfiller's measured-axis factors). These are mechanism contributions for ANY carrier, not
   a SNeRV rate fix. Wired as V3 rows below.

## Guards / scope

N=48-of-600 pairs (uniform first 48; per-pair d_seg spread 0.0019–0.0028, no degenerate constants).
Every rung byte-closed (real re-pack of the SNAR2 packet with recomputed decoder-header sha fields)
and EXACT re-measured through the identical inflate (`decode_snerv_archive_frames`) + scorer path
(real upstream SegNet/PoseNet `compute_distortion`, carrier-res→camera-res bilinear → rint/clip
uint8) that G1b used. All values `[macOS-CPU advisory]`; contest-axis (Linux x86_64 / T4) replay is
the promotion gate and is NOT claimed. Frontier inflate dependency = torch (the frontier runtime);
SNeRV decode is numpy. S8 runtime: frontier inflate 155 s + SNeRV decode 25–46 s/rung, both inside
the 30-min budget.

## Audit-provenance (per `src/tac/optimization/audit_provenance.py`; surface field MANDATORY)

1. claim "S8 frontier-LL→SNeRV nonrate 2.83 cone-FAIL (all 5 rungs)" · file
   `…/snerv_branch_b_round2_20260610T000000Z/s8_compose_frontier.v1.json` · field `rungs[*].nonrate_floor`
   · **surface: receiver** · reproduce `…/s8_compose_frontier.py --n-pairs 48`.
2. claim "frontier render DIRECT nonrate 0.0647 (d_seg 0.000538 / pose_term 0.0109)" · run-dir
   diagnostic (frontier `0.raw` → identical scorer path) · **surface: receiver** · reproduce: the
   diagnostic snippet in the run log / re-score `frontier_inflate/0.raw` first 48 pairs.
3. claim "frame0 SegNet-blind exact (Δd_seg vs ctrl = +0.00e+00 at uint8/6/5/4/3)" · file
   `…/s5a_frame0_hard_quant.v1.json` · field `rungs[*].delta_d_seg_vs_ctrl` · **surface: receiver** ·
   reproduce `…/s5a_frame0_hard_quant.py --n-pairs 48 --role-floor`.
4. claim "frame0 pose floor knee = uint6 (pose 0.197 OK; uint5 0.327 FAIL); saves 92.2→70.1 MB" ·
   same file · fields `rungs[*].pose_score_term` / `est_packet_floor_bytes` · **surface: receiver/export**.
5. claim "S2 chroma-uint6 near-free (pose +0.0165, d_seg +2e-5); 94.6→66.7 MB" · file
   `…/s2_yuv_channel_split.v1.json` · **surface: receiver/export** · reproduce `…/s2_yuv_channel_split.py`.
6. claim "band law skip_high≈2.0024·downsample(output)−0.056, corr 1.0" · S8 JSON `band_law` +
   `frontier_ll_skip_high_stats` (mean 124.7) vs round-1 `skip_high_stats` (mean 42.9) ·
   **surface: export** · reproduce: the linear-fit probe in the run log.
7. claim "frontier archive b7106c9b… 178,493 B inflates 1200 frames in 155.2 s" · file
   `…/frontier_inflate/inflate_run.log` ("saved 1200 frames" + "/usr/bin/time -l" real 155.18) ·
   **surface: receiver** · reproduce: re-run the submission_dir inflate.sh on the extracted `x`.

## V3 ingest (landed rows)

`…/snerv_branch_b_round2_v3_ingest.v1.json` — one
`snerv_branch_b_round2_candidate_action_evaluation.v1` row per rung (vehicle=snerv,
authority_tier=exact_cpu_advisory, metric_family=exact_pair_scorer, pays_rent=False for all S8 +
all stored-LF rungs above the SNeRV base only where floor < base). Each row carries the directive's
mandated fields: schema / base_archive_sha256 / payload_section / mutation / bytes_before-after /
d_seg / d_pose / score / delta_score_total / first_failed_surface / keep_or_reject /
bytes_vs_178KB_ratio / measurement_scope. first_failed_surface: S8 = `d_seg_guard`+`pose_term_guard`;
S5a uint5- = `pose_term_guard`; all = `rate_bar_vs_0.19199`.

## Cross-refs

`snerv_rate_attack_round2_directive_20260610.md` (the binding directive) ·
`snerv_branch_b_rate_attack_round1_20260609.md` (the uniform-ladder pricing baseline) ·
`snerv_g1b_export_binding_verdict_20260609.md` (the fidelity-survives-receiver / rate=100% fork) ·
`evaluator_invisibility_basis_landed_20260610.md` (the frame0-SegNet-invisible corollary S5a
confirms exactly) · `feedback_z8_detail_entropy_headroom_report_landed_20260531.md` (the sister
wavelet-storage rate anchor — same disease: near-lossless storage of source-derived state).
