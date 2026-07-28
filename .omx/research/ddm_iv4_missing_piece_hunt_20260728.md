# ddm_iv4 — THE MISSING-PIECE HUNT (the auditor's inverse of iv3, 2026-07-28)

**Arm:** `ddm_iv4` (codec scientist-artist, INVERTED — hunts absences), READ-ONLY. Base: `main@49224d51e9`.
**Directive:** operator 2026-07-28 — the complementary arm to iv3. iv3 composed what EXISTS (TOP-7 bridges,
the "organism" projecting ~90–150 KB vs the 154,522 B sub-0.15 line). YOU hunt what is MISSING: the absent
stages, the unmeasured couplings, the unbuilt math, the published techniques nobody imported — and *which
absence kills the projection*.

**NO-FAKE / calibration.** Contest S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489 vs FROZEN SegNet-argmax
(rank-4 linear head) + PoseNet-6-MSE; box ≤~200 KB @ d_seg ≤0.00116, d_pose ≤0.00161; effective bar =
min(0.15, official 0.172). Pointer UNMOVED at **0.19108 [contest-CPU]**; the live composed archive is
r6cal's **S=194.43** (99.73% RATE). `score_claim=false · promotion_eligible=false ·
rank_or_kill_eligible=false`. Every row labeled MEASURED / DERIVED / CONJECTURE / EXTERNAL(citation). No
number here moves the pointer; this memo names the holes and the cheapest fill for each. **Non-additive
pools bind** — absences that share a pool are co-measured, never summed.

The auditor's question, asked of iv3's beautiful organism: *what does it silently assume, what did every
real codec learn the hard way that it skips, and which single absence collapses ~90–150 KB to >300 KB?*

---

## §0 THE RANKED MISSING TABLE (gap · severity · why · cheapest fill · named consumer)

Severity: **FATAL** = its unfavorable resolution collapses the projection past the bar; **MAJOR** = a
hard-won codec stage/coupling the design skips, sized to matter; **MINOR** = real but small or bounded.
Fill verbs: **MEASURE** (instrument exists or is cheap), **BUILD** (new stage/solver), **IMPORT**(cite).

| # | Gap | Sev | Why it bites (label) | Cheapest fill | Consumer |
|---|---|---|---|---|---|
| A1 | **Stream E residual syndrome coded as raw brotli, not as a Slepian-Wolf/Wyner-Ziv SYNDROME** | **FATAL** | Stream E is EXACTLY the distributed-source-coding setting: transmit corrections against a decoder-side prediction (copy/warp/atlas = side information). Coding it as brotli'd residual leaves the entire Slepian-Wolf gain on the table. H(flip \| free context) is the OPEN, tier-moving scalar (pantheon F6 NULL); at ~1 bit/site over the 0.864% copy-support (1,019,467 sites) the raw flags alone are ~127 KB — this is collapse-driver #1. DSC codes the *coset*, not the residual. [DERIVED sizing + EXTERNAL theory] | IMPORT Pradhan-Ramchandran DISCUS / LDPC-syndrome (Wyner-Ziv); MEASURE H(flip\|context) through a real coset coder | r2s (Stream E), MAIN |
| A2 | **No composed multi-stream CO-MEASURE harness — every bridge measured in ISOLATION** | **FATAL** | The composition's `(d_seg, d_pose, bytes)` TRIPLE has never been measured together on one archive. Each stream touches the frames PoseNet reads; cb1 MEASURED a seg-motivated repaint (Lane carrier) adds **+22.7 d_pose** (opposite sign to MyCar's −0.18). A composed archive can fix d_seg and WRECK pose — collapse-driver #2 — and no instrument would see it. [MEASURED confound anchor: cb1 opposite-sign carriers] | BUILD the km1/fc1 assembly harness: realize all streams → one 384×512 plane → ONE frozen-scorer eval emitting the triple | fc1 assembly, MAIN |
| A3 | **No RDO MODE-DECISION stage — carriers route by CLASS LABEL, not by per-region λ-cost** | **MAJOR** | Every block codec's hardest-won lesson: bits are saved by choosing, per region, the CHEAPEST mode at a common λ (MPEG-4's 7 BAB modes; HEVC RDO). iv3 Bridge 4 routes each class to its native carrier unconditionally — but a temporally-static Movable patch wants SKIP (no-update), a ξ-predictable Lane wants inter-CAE, an uncovered boundary wants intra-CAE. Without the decision, static regions are over-coded and innovation under-coded. [EXTERNAL: ISO/IEC 14496-2 §BAB modes; MEASURED: crosswalk documents the 7 modes but the live pipeline has NO mode-decision node] | BUILD a per-region argmin-over-modes stage at the swept λ; IMPORT the 7-mode taxonomy (already crosswalked `mpeg4_shape_coding_intake_20260719`) | r2s (SUPPORT/CODING), fc1 |
| A4 | **Stratum-seam flips UNMEASURED — the per-stratum stitch has no in-loop restoration (deblock/SAO analog)** | **MAJOR** | Independent per-stratum PREDICT (Road plane / Lane / Movable / hood) stitched into one frame creates discontinuities at stratum seams — and real codecs deblock/SAO *inside the loop* because seam discontinuities hurt. Here the seams land on the codim-1 boundary annulus, already the dominant residual (boundary d_seg **0.427** vs interior 0.0235). Seam-induced flips would land on the most expensive pixels. [MEASURED: boundary=0.427 residual; UNMEASURED: whether stitching adds flips there] | BUILD a stratum-seam flip auditor (diff argmax at ±RF of every stratum boundary); IMPORT HEVC SAO in-loop offset as the per-seam repair | r2s (PREDICT), fc1 |
| A5 | **No BACKWARD/bidirectional prediction — the 81 MB frame_0 bootstrap (27.9% of the archive) has an EMPTY predictor node** | **MAJOR** | frame_1 carries ALL the d_seg content and is solved; frame_0 is seg-free (SegNet reads last frame only). The pose leg warps frame_1 FROM frame_0 (forward) — but nobody backward-predicts frame_0 FROM the solved frame_1 via the SAME ξ (inverse warp). The 2nd-largest byte carrier rides no predictor. This is the textbook B-frame move using only already-paid ξ. [MEASURED: frame_0 = 27.9%/81 MB dead-node; UNBUILT: reverse warp] | MEASURE: inverse-warp frame_1→frame_0 by stored ξ, code only the residual (keyframe_codec + xi_temporal_delta_coder both exist) | r2s (frame_0 crush), fc1 |
| A6 | **uint8 realization of each bridge UNMEASURED — the #532 wall (range(A) exactness breaks under uint8)** | **MAJOR** | Memory `realization_is_quantization_gated_minimal_writes_die_at_uint8` is a KNOWN killer: exact fp constructions die at uint8 rounding. Bridge 1 (depth→SE3 flow→sub-pixel resample) and Bridge 5 (ker(A) "free fiber" projection is exact in fp — but the stored representative must ROUND to uint8; does the free fiber survive?) both silently assume real arithmetic. v14 already MEASURED a mask promise 2.83e-4 → 0.0275 (97× worse) after fixed-prototype RGB projection under realization. [MEASURED analog: v14 paint gap; UNMEASURED per-bridge] | MEASURE each bridge through uint8-STE round-trip; report realized ΔS not fp ΔS | r2s (all stages), MAIN |
| A7 | **frame_0 crush × copy-PREDICT support-growth COUPLING unmeasured** | **MAJOR** | If frame_1's warp source is a CRUSHED frame_0, degrading frame_0 degrades the warp → MORE flip support in frame_1 → bigger Stream E. Named in the directive, never co-measured. Couples A5 and Stream E. [CONJECTURE coupling; both endpoints MEASURED separately] | MEASURE: sweep frame_0 crush quality vs frame_1 flip-support count jointly | r2s, fc1 |
| A8 | **DIBR disocclusion holes — the pose warp covers only 0.978 of the frame; 2.2% is UNDEFINED at object boundaries** | **MAJOR** | oc1 MEASURED "mean warp cover 0.978." The missing 2.2% is disocclusion: background newly exposed at depth discontinuities the source frame never saw — and it lands exactly at Movable/object boundaries where d_seg flips live. The warp silently leaves holes; standards fill them with depth-aware inpainting (VSRS). Un-filled holes → flips + pose error. [MEASURED: 0.978 cover; EXTERNAL: DIBR hole-filling; UNMEASURED: hole→flip cost] | IMPORT depth-aware inpainting (VSRS-style extrapolation); MEASURE flip mass in the 2.2% hole band | r2s (PREDICT), fc1 |
| A9 | **Plane+parallax is DEGENERATE exactly on Bridge 1's target (lead-car follow) — the free-3D-depth PREDICT will fail on the Movable band** | **MAJOR** | Irani-Anandan: residual parallax is radial about the epipole and DEGENERATE when the camera follows same-direction motion — i.e. forward-driving behind a lead car (pantheon's MEASURED lead_car_pass pair 452). Bridge 1's biggest target is Movable (27% of flip mass) — exactly where P+P is degenerate. iv3's "free 3D depth reverses the 2D-homography negative" claim is untested on the degenerate pairs. [EXTERNAL: Irani-Anandan degenerate motion; MEASURED: pair 452 lead-car; DERIVED risk] | MEASURE Bridge 1's Movable flip-support SEPARATELY on the lead-car/same-direction pairs before crediting it | r2s (PREDICT), MAIN |
| A10 | **Rate-control feedback loop MISSING — waterfill runs ONCE; no encoder-side per-unit closed loop** | **MAJOR** | Real codecs close the loop per-CTU (measure realized bytes, re-steer λ). Ours has only decoder-side quant feedback (`materializer_feedback`, `decoder_q_selective_runtime_feedback`) — no encoder loop that re-measures the composed archive's ACTUAL bytes and re-allocates. The 90–150 KB projection assumes every stream hits its bracket; without feedback, bracket misses compound silently. [MEASURED: only decoder-side feedback exists; DERIVED risk] | BUILD the one-λ cross-stream waterfill as a runnable solver consuming measured bytes (overlaps A2/§apparatus) | fc1, MAIN |
| A11 | **The archive still MATERIALIZES pixels — VCM/FCM's ~85% win is in the FEATURE domain, not pixel domain** | MAJOR(strategic) | MPEG standardization (2026 DIS) MEASURED: Feature-Coding-for-Machines beats pixel-domain Video-Coding-for-Machines by **~85% bitrate** at equal task accuracy. Our residual (Stream E) still lives in RGB pixel space (the 210 MB residual is pixel-domain). "Solve in description coordinates" (family-d) is the right instinct but the coder still prices pixels. laguerre/range(A) gesture at logit-domain; Stream E does not. [EXTERNAL: MPEG-VCM/FCM DIS 2026] | IMPORT the FCM principle: code the residual against SegNet's OWN margin/logit field, not RGB; the syndrome (A1) lives in flip-label space, not pixel space | MAIN (strategic), r2s |
| A12 | **Container/header overhead at multi-stream — bounded but a VOP-granularity trap** | MINOR | MEASURED: recursive ZIP overhead ~196 B/leaf (cc2, 27-leaf composition = 5,292 B); ~100 B/section (m6). At the 5-stream organism ≈ 1 KB (negligible). BUT if VOPs explode into per-object-per-frame leaves, overhead balloons. [MEASURED: cc2 5,292 B / 27 leaves] | Design constraint: keep VOPs a SINGLE tracked-object stream, not per-object leaves; already-measured, just honor it | fc1 |
| A14 | **DECODE WALL-CLOCK vs the 30-min budget — the composed codec's per-symbol adaptive-arithmetic + iterative-syndrome decode is unmeasured and lineage-critical** | **FATAL(conditional)** | The LIVE block-coded 291 MB archive decodes in **489.7 s** (cc3) / 738.98 s (r1b, 8 workers) — SAFE, because brotli/lzma decode in C. BUT the composed codec ADDS: per-stratum SE(3) warps ×1200 frames + analytic depth + **context-adaptive arithmetic decode over ~1M+ support sites** (iv3 Bridge 2 / `context_partition_codec`) + (if A1 imported) **iterative belief-propagation LDPC syndrome decode**. Per-symbol adaptive-arithmetic in PURE PYTHON is the classic 30-min killer; iterative BP over 1M sites likewise. A byte-winning coder that busts 1800 s is INADMISSIBLE — this is a collapse candidate (Collapse-4). [MEASURED: block-coder decode safe; UNMEASURED: composed adaptive/syndrome decode] | Use `constriction`'s Rust core (already a dep) NOT a hand-rolled Python CABAC; vectorize BP (numpy/torch) or vendor via `runtime-rs` (`residual-codec`/`qma-codec`/`tac-boundary-decode` crates EXIST); MEASURE composed decode wall-clock before crediting any per-symbol coder | r2s (CODING), MAIN |
| A15 | **iv3's byte projection targets the BOX distortion point — whose ZERO-RATE floor already EXCEEDS the bar; the design must realize the SOLVED distortion, not the box** | **FATAL** | The blocker receipt MEASURED the box-tolerance state (r6cal/MS2R) at **zero-rate lower bound 0.524** — 3× over 0.172 — because at box distortion (d_seg 0.00116 → 0.116, d_pose 0.0166 → 0.408) the distortion floor alone busts the bar regardless of bytes. Even d_seg 0.00116 + banked-R1 pose 0.00161 = **0.116 + 0.127 = 0.243 at zero rate** — STILL over 0.172. Sub-bar requires the SOLVED distortion (d_seg 1.52e-4, d_pose 1.02e-4 → zero-rate floor **0.047**, MEASURED C1_MS1) at ≤154,522 B. iv3 §0/§2 cite "d_seg ≤0.00116" as the organism's target — at that point the 90–150 KB projection can NEVER reach 0.172. The realization crux is not optional polish; it is the difference between a 0.047 and a 0.243 floor. [MEASURED: MS2R 0.524, C1_MS1 0.047 zero-rate LBs] | Re-anchor every stream's distortion target to the SOLVED point (1.5e-4 / 1e-4); the km1 confirming fit must land solved-grade distortion, not box-grade | MAIN, fc1, r2s |
| A13 | **Error resilience — correctly IRRELEVANT (stated to close the checklist)** | — | No transmission channel: single deterministic decode inside the 30-min budget. Determinism is the requirement (rule-118), resilience is not. No packet loss, no drift, no reference-frame corruption. Correctly absent. [DERIVED] | none | — |

---

## §1 THE TOP-3 COLLAPSE SCENARIOS (the arithmetic that turns ~90–150 KB into >300 KB)

**Collapse-1 — Stream E is incompressible (A1 unfavorable).** copy-PREDICT leaves 0.864% support =
1,019,467 flipped sites [MEASURED, oc1]. If the flip field is high-entropy at the boundary annulus,
H(flip\|free context) ≈ 1 bit/site → the residual FLAGS alone ≈ 1,019,467 / 8 ≈ **127 KB** before a single
value. Add the SegNet-RF dilation oc1's rung-2 falsifier warns of: holding argmax after a sparse residual
needs each flip dilated to its receptive field; if dilation exceeds ~10% of sites (11.8M) even at 0.1
bit/site that is **147 KB**. Arithmetic: 90 KB atlas + 127 KB syndrome + 15 KB VOP − 5 KB lossless ≈
**227 KB ≫ 154 KB line**. *This is the single most likely collapse* — and it is the OPEN term (F6 NULL).
The fix is precisely A1 (code the coset, not the residual) + A11 (in label space, not pixels).
[DERIVED from MEASURED support count; H is UNMEASURED — that measurement is the whole game.]

**Collapse-2 — composed d_pose blows up (A2 unfavorable).** Banked pose = d_pose 0.001610 → contrib
**0.127** [MEASURED, r1_dxi]. cb1 MEASURED the Lane carrier adds **+22.7 d_pose** [MEASURED advisory]. The
atlas/VOP/residual streams all repaint the pose-read frames. If composition perturbs pose by even Δd_pose
+0.05 (**454× smaller** than cb1's measured Lane blow-up), composed d_pose = 0.0516 → contrib √(10·0.0516)
= **0.718** → **+0.59 S**. The √-nonlinearity means any repaint that nudges pose off its tube is
catastrophic: +0.59 is **3.4× the entire 0.172 bar**, from a coupling no isolated measurement sees. *This
is why A2 (co-measure harness) is FATAL-missing, not just nice-to-have.* [MEASURED endpoints; composition
UNMEASURED.]

**Collapse-4 — a byte-winning coder busts the 30-min decode budget (A14 unfavorable).** The rate win of
Collapse-1's fix (Wyner-Ziv syndrome / context-adaptive arithmetic over the ~1M-site support) is realized
only if it DECODES inside 1800 s on contest CPU (4-core/16 GB) or T4. The live block-coded archive decodes
in 489.7 s [MEASURED, cc3] — but per-symbol context-adaptive arithmetic decode over 1,019,467+ sites in
pure Python, or iterative LDPC belief-propagation, can run 10–100× slower. If the composed decode lands at,
say, 3,000 s, the archive is INADMISSIBLE regardless of its byte count — the byte win is unrealizable. This
is not a distortion collapse; it is an admissibility collapse, and it is invisible to every byte/S
measurement. *A coder must be measured for wall-clock, not just bytes.* Fill: `constriction` Rust core (dep
exists) or vendor the `runtime-rs` codec crates; never hand-roll a Python per-symbol loop. [MEASURED:
block-coder decode; UNMEASURED: composed adaptive/syndrome decode.]

**Collapse-3 — Bridge 1 free-depth PREDICT fails on Movable (A8+A9 compound).** P+P is degenerate on
lead-car-follow pairs [EXTERNAL: Irani-Anandan] AND the depth warp opens 2.2% disocclusion holes at object
boundaries [MEASURED: 0.978 cover]. Movable = 1,083,972 flips = **27.0% of flip mass** [MEASURED], the #1
structural residual (conditional d_seg 0.988). If free-depth PREDICT fails there — both degenerate AND
hole-ridden — Movable reverts to a DENSE VOP carrier or falls into Stream E. VOP budget balloons 15 → 30+
KB, and 27% of the flip mass re-enters the syndrome → feeds Collapse-1. Arithmetic: iv3's "single largest
attack for zero bytes" (Bridge 1) is aimed at the one region where two independent literature failure modes
both fire. *Credit Bridge 1 only after measuring Movable flip-support on the degenerate pairs specifically.*
[DERIVED risk from MEASURED flip mass + EXTERNAL degeneracy + MEASURED cover.]

---

## §1.5 SIBLING-CAMPAIGN OBSTRUCTIONS — do their walls apply to OUR 6-stage design? (MAIN addendum)

A parallel campaign (`original_taskspace_inverse_witness_codec_20260725`, the g111/g120/g121 taskspace-
inverse-witness lineage) hit named obstructions chasing the SAME endpoint. Classified against our design:

| Their receipt (MEASURED) | Their wall | Lineage-specific or OURS? | Applies to our stage? Sev |
|---|---|---|---|
| **C1_MS1** exact two-plane raster: d_seg 1.5e-4, d_pose 1.0e-4 (SOLVED), **409.5 MB**, needs **2183× compression** | solved distortion exists but 3 orders over the rate ceiling | **OURS** (the exact lattice solve = our box realization) | **CONFIRMS THE CRUX** at stage 3/4 (realize solved distortion compactly). MAJOR — not a new wall, the numeric proof of ours |
| **MS2R** box-tolerance state (= r6cal): **zero-rate LB 0.524** | box distortion is not frontier-feasible even at zero bytes | **OURS** (r6cal is the live archive) | **FATAL — this IS A15.** The box distortion point can never reach 0.172; the design must hit the SOLVED point |
| **G20/G25/G29** ep725 recode: 80 KB but **d_pose 127** → S 36 | compact trained base ships un-solved pose | lineage-specific (trained INR base) BUT the class-lesson transfers | Reinforces **Collapse-2**: a compact atlas does NOT come with solved pose; pose is a REQUIRED separate solve/bank. MAJOR |
| **G51/G55/G57** direct task-layered Y1 + Y0\|Y1, receiver-closed 182 KB: **d_seg 0.179** (154× box), d_pose 45 | a fresh DIRECT scorer-plane temporal stream under-realized distortion badly | lineage-ADJACENT — this is close to our family-(d) "solve in description coordinates" | **WARNING for our confirming build (§5 km1):** a naive layered direct-scorer-plane stream FAILS realization. The joint fit must be genuinely joint with the real coder in-loss, NOT a layered Y1/Y0 stream. MAJOR |
| **g121 coldroot** v9-semantic trained base: seg-score LB 0.267 by pair 32, d_seg ~0.0067 by pair 80 | a trained-INR argmax base is 5.8× over the box on d_seg | lineage-specific (v9 witness/INR, BANNED-adjacent) | class-lesson only: a TRAINED base cannot substitute for exact realization. Confirms the crux; the specific number does not transfer |

**The convergent verdict (their negative space = our hunting ground):** across the ENTIRE sibling campaign,
every base is exactly one of — (a) solved-distortion but huge (C1: 409 MB, OURS), (b) compact but
un-solved distortion (G55 182 KB@0.179; G20 80 KB@d_pose 127; g121 trained@0.0067), or (c) box-tolerance
but not frontier-feasible even at zero rate (MS2R/r6cal 0.524, OURS). **No artifact in either lineage is
simultaneously compact AND solved-distortion.** That is the realization crux, independently re-proven from
two lineages — and it directly upgrades A15 to FATAL and warns family-(d) off the G55 layered formulation.
The `NO_EXISTING_FRESH_ORIGINAL_RECEIVER_CLOSED_FRONTIER_COMPETITIVE_ARCHIVE` verdict is not a lineage
quirk; it is the same wall our design must break, and it locates it precisely at stages 3–4 (realize the
solved distortion compactly) — the same place A1/A15/Collapse-1 already point. [All rows MEASURED per the
cited receipts; OURS/lineage-specific classification DERIVED.]

## §2 WHAT THE MASTERS WOULD ADD (the one technique from codec history this design most needs)

The design's dominant open term is Stream E — the residual/correction stream where the atlas, warp, and
VOPs all miss — and iv3's organism, r6cal's live archive, and every current plan code it as a **raw
brotli'd residual**. That is precisely the setting that **distributed source coding** was invented for.

The residual is a source X the encoder must transmit; the decoder already holds a highly-correlated
prediction Y (the copy/warp/atlas). Slepian-Wolf (1973) proved X can be sent at rate H(X\|Y) *without the
encoder ever seeing Y*, and Wyner-Ziv (1976) extended it to lossy reconstruction with side information —
the exact structure here, where Y is decoder-side and X is the flip correction. Pradhan-Ramchandran's
DISCUS (1999) and the practical LDPC/LDGM-syndrome constructions
([Wyner-Ziv via LDPC](https://www.researchgate.net/publication/224683114_Wyner-Ziv_video_coding_using_LDPC_codes),
[graph-based WZ code design](https://arxiv.org/pdf/1205.4332)) make it runnable: transmit the **syndrome**
(coset index) of X with respect to a channel code matched to the flip-error channel between prediction and
truth; the decoder finds the best estimate in the signalled coset. The gain over coding the residual
directly is exactly the Slepian-Wolf bound — the difference between H(X) and H(X\|Y), which for a 0.864%
flip field against a good predictor is large.

Fused with the FCM lesson (A11) — code in the machine's feature/label domain, not RGB, where MPEG measured
~85% bitrate savings at equal task accuracy — the masters' single addition is: **code Stream E as a
Wyner-Ziv syndrome over the argmax-flip *label* field, against the decoder's own prediction as side
information.** This is the one stage the entire ~90–150 KB projection rests on, it is the one term the tree
codes naïvely today, and it is the exact-fit theory nobody in this repo has imported. Build the co-measure
harness (A2) first so the syndrome's realized `(d_seg, d_pose, bytes)` triple is honest; then import DISCUS
for Stream E. Everything else — mode decision, seam repair, backward prediction, hole filling — is a
MAJOR-but-second-order refinement of a codec whose open term is a distributed-source-coding problem wearing
a brotli disguise.

---

## §3 APPARATUS GAPS (the instruments missing to CONFIRM the organism — ranked)

1. **Composed multi-stream CO-MEASURE harness** (A2) — the km1/fc1 assembly as runnable code: realize all
   streams into one 384×512 plane, run ONE frozen-scorer eval, emit `(d_seg, d_pose, bytes)`. Does NOT
   exist (tools/ has old substrate-cross-archive "composition" scripts, none for the ego-scene codec).
   **This is the instrument that fires Collapse-1 and Collapse-2.** Highest priority — without it every
   bridge stays isolated and the two FATAL couplings are invisible. BUILD.
2. **One-λ cross-stream waterfill SOLVER as runnable code** (A10) — `seven_home_stream_allocator` is
   self-disclaimed research-only; `region_merge` solves support not cross-stream; the KKT Sinkhorn
   waterfill over F1–F6 at swept λ is a CONCEPT. BUILD the solver that consumes measured per-stream bytes
   and re-allocates (the encoder rate-control loop).
3. **Stratum-seam flip auditor** (A4) — does not exist (`region_merge` is support-MDL, not seam-diff).
   BUILD: diff argmax at ±RF of every stratum boundary, report seam-induced flip mass. The in-loop-filter
   instrument.
4. **H(flip \| free decoder context) meter through a REAL coset coder** (A1) — the tier-moving scalar is
   NULL/UNMEASURED (pantheon F6). Not an entropy proxy (G4 already proved proxies ≠ real coder). BUILD as
   part of the DISCUS import.

---

## §4 HONESTY + WHAT THIS ARM DID NOT DO

- No score claim; pointer UNMOVED at 0.19108. Every collapse figure is DERIVED from MEASURED endpoints
  (support counts, cover, cb1 pose sign, ZIP overhead) with the OPEN term (H(flip\|context)) explicitly
  UNMEASURED — that is the point: the collapse hinges on the one number nobody has measured.
- I did NOT build any harness or run any measurement (READ-ONLY arm). The fills are named for the r2s /
  fc1 / MAIN consumers.
- I did NOT re-litigate iv3's bridges — the parts are real and MEASURED-to-exist; this arm audits the
  WIRING and the ABSENCES the composition silently assumes away. iv3 and iv4 are the same organism seen
  from composition (what fuses) vs audit (what's missing to make the fusion honest).
- Literature imports are cited inline; the MPEG-4 CAE machinery is already crosswalked (design) but NOT
  built into the live pipeline — A3's mode-decision stage is the load-bearing un-built piece.

## STORES CONSULTED

`CLAUDE.md` (NO-FAKE; §7.1 dynamical gates; inflate-free/rule-118; non-additive pools; pointer-only;
measured-scored-quantity axis) · `MEMORY.md` current-state (realization-quantization-gated;
frozen_scorer_exact_factorization; opportunity_pools_non_additive; distortion_byte_economics_upper_bounds)
· `.omx/research/pantheon_synergy_crux_synthesis_20260728.md` (F1–F6, §2 n600 evidence, §5 km1, §7) ·
`ddm_iv3_codec_artist_synergy_bridges_20260728.md` (the 7 bridges + organism this audits) ·
`ddm_oc1_xi_temporal_predict_measured_20260727.md` (2D-homography task-NEGATIVE 2.16×; 0.864% copy support;
0.978 warp cover; dead PREDICT node) · `ddm_iv1_plugin_inventory_sweep_20260728.md` +
`ddm_iv2_independent_plugin_sweep_20260728.md` (built inventory; 210 MB residual + 81 MB frame_0 bootstrap;
Movable 0.988 / Lane 0.437 residuals) · `mpeg4_shape_coding_intake_and_crosswalk_20260719.md` (7 BAB modes,
10/9-pixel CAE templates — crosswalked, NOT built) · `codex_findings_ddm_cc2_coder_races` (ZIP overhead
5,292 B/27 leaves) · `codex_findings_ddm_m6` (100 B/section) · cb1 (+22.7 Lane d_pose, −0.18 MyCar) ·
r6cal (S=194.43, 99.73% rate) · `upstream/{evaluate.py,modules.py}` (seq_len=2; SegNet last-frame argmax;
PoseNet two-frame YUV6) · sibling campaign (MAIN addendum):
`original_taskspace_inverse_witness_codec_20260725/{g121_coldroot_exact_monotone_obstruction_receipt,fastest_existing_score_path_blocker_receipt}_20260727.json`
(C1_MS1 0.047 / MS2R 0.524 / G20 d_pose 127 / G55 d_seg 0.179 — the two-lineage realization-crux re-proof) ·
`src/tac/optimization/ddm_runtime_receiver.py` (LIVE receiver = brotli/lzma block coders, C-fast) ·
`runtime-rs/crates/{residual-codec,qma-codec,stbm1br-codec,tac-boundary-decode,tac-levelset-inflate}`
(native decode crates exist) · cc3/r1b decode wall-clock 489.7 s / 738.98 s [MEASURED, <1800 s].
EXTERNAL literature (cited inline): MPEG-4 Part 2 CAE
([IEEE 723476](https://ieeexplore.ieee.org/document/723476/)); Wyner-Ziv/LDPC syndrome
([RG 224683114](https://www.researchgate.net/publication/224683114_Wyner-Ziv_video_coding_using_LDPC_codes),
[arXiv 1205.4332](https://arxiv.org/pdf/1205.4332)); plane+parallax degeneracy
([Irani-Anandan direct recovery](https://www.weizmann.ac.il/math/irani/sites/math.irani/files/publications/direct_recovery.pdf));
DIBR disocclusion
([EURASIP 10.1186/s13640-016-0109-6](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-016-0109-6));
MPEG-VCM/FCM ~85% feature-domain gain
([Mile-High Video 2026](https://dl.acm.org/doi/10.1145/3789239.3793282),
[arXiv 2001.03569](https://arxiv.org/pdf/2001.03569)); HEVC RDO/SAO in-loop
([SAO in HEVC](https://www.researchgate.net/publication/255568022_Sample_Adaptive_Offset_in_the_HEVC_Standard)).
