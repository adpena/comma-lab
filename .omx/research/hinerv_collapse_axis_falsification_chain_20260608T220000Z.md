# HiNeRV parse-back collapse — falsification chain + build-path localization (v7/v8/v9 smokes)

UTC: 2026-06-08T220000Z · claude (solo) · planning, no score claim. Authority: every
number below is `[macOS-MLX research-signal]` / counterfactual; NONE is a contest score.
The only authority terms are exact `100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489`
on the submitted archive bytes (not measured here).

## The phenomenon
A target-region SegNet birth wins ~12k region pixels live/fakequant; the runner's
authoritative selected-archive parse-back wins only **2**. This tranche diagnosed WHY,
letting the artifacts decide. Five hypotheses were tested; four were falsified.

## Hypothesis ledger (artifact-decided)

| # | hypothesis | status | deciding artifact |
|---|---|---|---|
| H1 | a single/pairwise DECODER SECTION collapses L | FALSIFIED (under int8) | v8 guilt sweep: every section ret_vs_fakequant=1.0, no collapse — but used wrong codec (int16_raw), so SUPERSEDED by H-codec |
| H2 | the LATENT codec int8_brotli_q11 collapses L | FALSIFIED | v9 grid: latent axis drop = 1376 (~11%, mild); int8_brotli_q11 keeps 10781/12157 |
| H3 | the DECODER codec int4_mixed collapses L | TRUE in grid, but NOT the selected cause | v9 grid: int4_mixed → 17 wins (catastrophic). BUT selected backend payload ~9197 ≈ int8 (9115), NOT int4 (5771) → selected is int8, so int4 is not what ships |
| H4 | EMA-vs-live selection collapses L | FALSIFIED | runner ema AND live candidate parsebacks BOTH = 2 |
| H5 | the runner's ARCHIVE BUILD PATH (sidecar wrap / coder-aware-qat export) collapses L | **LEADING** | same live weights + same int8 codec: my counterfactual pack (no sidecar, 9115B) = 10781; runner archive (8127B sidecar, 16326B zip) = 2 |

## v9 controlled codec grid (same compact live export, ONE builder — apples-to-apples)
```
decoder/latent                wins    retention_vs_live   payload_bytes
int8_mixed / int16_raw        12157   0.998               9115   (faithful)
int8_mixed / int8_brotli_q11  10781   0.885               9123   (latent mild: -1376)
int4_mixed / int16_raw           17   0.0014              5771   (decoder catastrophic: -12140)
int4_mixed / int8_brotli_q11     17   0.0014              5779
```
collapse_axis = decoder_codec (decoder drop 12140 >> latent drop 1376).

## Selected-archive custody (v9, sha a30859e1, 16325B zip, candidate ema)
- effective_decoder_codec = int8_mixed (telemetry); backend payload ~9197 ≈ int8, NOT int4.
- latent blob is tiny (1 pair); int16_raw vs int8_brotli differ by 8 bytes — latent codec is not the byte lever here.
- payload 17324 = int8 backend (~9197) + base64 target-region action sidecar (~8127).
- ema parseback = 2; live parseback = 2 (EMA non-causal).
- target_margin over region: p10 -1.78, p50 -1.15 (birth is well below the SegNet wall at parse-back).

## The apples-to-apples bug this tranche caught (twice)
1. My decoder-section sweep (v8) re-packed with int8_mixed+int16_raw → 12157 wins, contradicting the
   runner's authoritative 2. Cross-check against the runner's parseback row exposed it.
2. The PRIOR `archive_roundtrip_shadow` "latents_fine not the cause" finding used int16_raw, not the
   real int8_brotli_q11 — but v9 shows even int8_brotli_q11 is mild, so latents remain not-the-cause.

The structural lesson: a section/codec shadow is only evidence about the SELECTED archive if it packs the
SAME build path the runner ships. Counterfactual packs are alternative-codec candidates, not selected-archive verdicts.

## Decisive next probe (cheap, precise — NOT done this tranche)
Render the runner's live/archive.zip **backend (unwrapped, no sidecar)** vs **with-sidecar**, counting region
target wins with `hi_nerv_hard_region_miner_inputs.npz::target_labels_bhw` + SegNet:
- if backend = 2 → the export build path (coder-aware-qat / pack_archive at selection) collapses it →
  compare against my counterfactual int8 pack (10781) byte-for-byte to localize the transform.
- if backend ~10781 but with-sidecar = 2 → the SIDECAR WRAP is inverting the birth (critical Rule #8 bug).
File/function: `tac.substrates.hi_nerv.inflate.build_model_from_archive` (wraps sidecar via
`wrap_model_with_target_region_actions`); `tac.substrates.hi_nerv.archive_candidate.pack_archive_from_exported_state_dict`
(my counterfactual pack) vs the runner's export/selection pack.

## DECISIVE RESOLUTION (backend-vs-sidecar separation; GPT Case B confirmed)
The runner emits TWO parseback rows per candidate; reading both resolves the "2":
- `hi_nerv_birth_parseback_survival_{ema,live}.json`: `parseback_wrong_to_target = 2`,
  margin p50 = -1.15 — this is the **BACKEND** HiNeRV decoder render with **NO sidecar**.
  The backend birth is DEAD at parse-back (both ema AND live, int8 bytes). EMA-vs-live FALSIFIED.
- `hi_nerv_target_region_action_parseback_survival.json`: the sidecar overwrites
  **2286** pixels (`parseback_program_survived: true`, `total_action_pixels: 2286`,
  `base64_text_bytes: 8128`) BUT `scorer_effect_survival_measured: false` /
  `parseback_scorer_effect_survived: null` — **Rule #8: the sidecar overwrite's SegNet
  effect over L is UNMEASURED.**

So: my counterfactual int8 pack surviving at 10781 is on a DIFFERENT build path (sidecar-less +
possibly different export snapshot), NOT the shipped program. The shipped program is
{dead backend (2) + 2286-pixel sidecar rescue whose scorer effect is unmeasured}.

### Reframed lowering race (the two operative unknowns, in priority order)
1. **SIDECAR-SCORER ROW (top priority).** Does the 2286-pixel sidecar overwrite, after
   parse-back + SegNet over L, win the region? If yes → the sidecar IS the shipped working
   rescue (price its 8128 bytes by exact ΔS). If no → Rule #8 failure; the action does not
   survive as a sidecar either. Spec: `.omx/research/hinerv_sidecar_scorer_effect_row_blocker_spec_20260608T210000Z.md`.
2. **BUILD-PATH CODEC-UPGRADE CANDIDATE (secondary).** Why does the counterfactual int8
   backend survive (10781) while the shipped int8 backend dies (2)? Suspects: H5b
   coder-aware-qat export transform (nominal int8 but effective-aggressive), H5c the export
   selection snapshot differs from the in-loop `model.export_state_dict()`. If a realizable
   shipped backend can be made to survive, it competes with the sidecar by exact ΔS.

## P0 SIDECAR-SCORER PROBE OUTCOME — BLOCKED (GPT Case C) + runner-"2" now suspect
Built `tools/probe_hinerv_sidecar_scorer_effect.py` (renders the live archive backend
unwrapped + with-sidecar wrapped, scores both with the torch SegNet over the birth region).
Confirmed structurally: the receiver IS wrapped (`TargetRegionActionReceiver`, 1 action),
the model emits RGB in **[0,1]** (not [0,255]), and the sidecar **correctly applies 2286
pixels** (max_abs_diff 0.875, n_changed=2286). Scale-bug in the probe fixed (×255 before SegNet).

**BUT the probe's NO-FAKE validation gate FAILS:** its independently-scored backend =
**11297** region wins, while the runner's authoritative parseback backend = **2**, on the
SAME archive (sha ddd88b6c). `--expected-backend-wins 2` → `verdict:
UNVALIDATED_backend_scoring_path_mismatch`, `scorer_effect_survival_measured:false`,
`parseback_scorer_effect_survived:null`. No sidecar verdict is trusted.

**Meta-finding (the runner's "2" is now itself suspect):** TWO independent re-measurements
of the same archive both say ~11k wins —
- v9 codec grid (uint8 eval-roundtrip, MLX in-loop path): int8 cells = 10781–12157;
- this probe (float, torch archive backend, extract_gt_masks): 11297 —
while ONLY the runner's authoritative parseback says 2. When two independent paths agree
and the "authority" disagrees by ~11k, the authority is the outlier. Candidate causes of the
runner's "2": a different win definition (transitions vs absolute argmax), a different region
alignment, a different frame index, the eval uint8 roundtrip applied differently, or a
genuine scorer-path bug in `measure_birth_parseback_survival_from_report`.

### The reconciliation that unblocks P0 (exact, file/function/line)
Align ONE scoring path end-to-end and find where 11k → 2:
- runner authoritative: `tac.substrates.hi_nerv.birth_survival.measure_birth_parseback_survival_from_report`
  → its win definition + `_candidate_logits_np` (uint8 roundtrip via
  `_receiver_uint8_roundtrip_ste_nhwc01`) + region reconstruction (`reconstruct_birth_region_mask`).
- my probe: `tools/probe_hinerv_sidecar_scorer_effect.py` uses `tac.scorer.extract_gt_masks`
  (float, no uint8 roundtrip, absolute argmax==cls over GT==cls).
Make the probe call the SAME `_candidate_logits_np` + cert win definition + region mask the
runner uses, gate on reproducing backend=2, THEN the sidecar number over L is trustworthy.
Until then P0 is BLOCKED on scoring-path identity (not on artifacts — both are persisted).

## SCORING-PATH IDENTITY PROBE (P0) — two more axes FALSIFIED; scorer-path isolated
`tools/probe_hinerv_scoring_path_identity.py` holds the render FIXED (torch archive
backend) and varies only the scorer axes on the same v9 live archive:
```
float [0,255]  absolute argmax==4 over region:  11297
uint8 camera-res roundtrip, absolute:           10762   (eval-roundtrip drops only 535)
float, wrong->target transition:                11297   (== absolute; all region pre-wrong)
runner authoritative backend:                        2
```
- **uint8 eval-roundtrip FALSIFIED** as the 11297->2 cause (drops 535, not ~11295).
- **win-definition (transition vs absolute) FALSIFIED** (both 11297).
- region size already matches the runner (50568), so region is not it.
=> the 11297->2 divergence is isolated to the **SCORER PATH**: my `tac.scorer.extract_gt_masks`
(upstream torch SegNet, interpolate+preprocess_input+argmax) vs the runner's
`tac.substrates.hi_nerv.birth_survival._candidate_logits_np` ->
`teacher_logits_for_frames_nhwc01` (the harness SegNet teacher, possibly the MLX port).
Next sub-test (named): score the SAME torch backend render with `_candidate_logits_np`'s
teacher vs `extract_gt_masks`; if the teacher gives ~2 and extract_gt_masks gives 11297 on
the SAME render, the runner's scorer path is a different SegNet (port drift or a different
preprocess) and one of the two is unfaithful to upstream `modules.py` DistortionNet.SegNet.

## DECISIVE HIGH-SIGNAL FINDING — the backend does NOT collapse; the "2" is a different quantity
Same-render two-scorer test (`tools/probe_mlx_segnet_fidelity_vs_torch.py`) on the v9 live
archive backend render (region 50568 px, class 4):
```
torch upstream SegNet wins (evaluate.py authority): 11297
MLX-ported SegNet wins (runner harness path):       11306
region argmax agreement torch-vs-MLX:               0.99966 (17 px disagree)
full-frame argmax agreement torch-vs-MLX:           0.99991 (18 px / 196608)
backend_collapses: FALSE
```
**The backend birth SURVIVES parse-back at ~11,300 region wins under BOTH the authoritative
torch SegNet AND the MLX port.** This is consistent with the runner's OWN margin distribution
(target_margin p50 = -1.15 ⇒ ~22% of 50568 = ~11k win), but CONTRADICTS the runner's
`wrong_to_target_count = 2`. Therefore the runner's "2" measures a DIFFERENT quantity than
absolute region wins — almost certainly a TRANSITION against an initial/reference render
(`initial_in_region_target_count: 0` → wrong_to_target counts pixels CREATED relative to that
reference), OR the sidecar net effect, NOT a backend collapse. **The "parse-back collapse to 2"
premise that drove this whole diagnosis is, at the backend-absolute-win level, an artifact of
the runner's win-DEFINITION, not a real birth collapse.**

ALL prior collapse hypotheses (H1 section / H2 latent / H3 int4 / H4 EMA / H5 build-path /
H5a sidecar / uint8 / win-def / region / MLX-port) are now subordinate to this: there may be
no backend collapse to explain. The runner's `measure_birth_parseback_survival_from_report`
win-definition must be read line-by-line and reconciled against absolute SegNet d_seg.

## OPERATOR DIRECTIVE (2026-06-08) — MLX-port drift hardening is TOP PRIORITY
"hardening and fixing and engineering all drift away possible of MLX port is a top priority" +
"getting this as high signal and fidelity as possible is extremely important". Measured drift:
**18 px / 196608 (0.0092%)** full-frame argmax disagreement between the MLX port and the
upstream torch SegNet on this render. The MLX port is NEAR-faithful but NOT argmax-exact.
Hardening lane: `tac.local_acceleration.mlx_scorer_adapters.MLXSegNetAdapter` → drive
`full_frame_argmax_disagreement_px` to 0 (and per-class logit max-abs delta below a tight
band) via `tools/probe_mlx_segnet_fidelity_vs_torch.py` as the gate; wire the gate into a
STRICT preflight so any MLX-scored birth carries a torch-parity attestation (false-authority
risk: a birth the MLX port calls won/lost may differ from the contest torch SegNet).

## *** DEFINITIVE RESOLUTION — the backend SURVIVES; the SIDECAR destroys it; the runner conflates them ***
Reconciliation using the runner's EXACT win-def + region + scorer on the v9 live archive:
```
                                              scorer = MLX adapter (runner path); win-def = region_margin_stats.region_hard_won_pixels; region = reconstruct_birth_region_mask (50568 px, MATCHES runner)
BACKEND (unwrapped HinervSubstrate):          region_hard_won = 11306   <- SURVIVES
WRAPPED  (build_model_from_archive=runner):   region_hard_won =     3   <- collapses
runner authoritative wrong_to_target:                                2
```
**The backend HiNeRV birth does NOT collapse — it wins ~11306 region pixels under the runner's
OWN exact win-def + region + scorer.** The runner's authoritative "2" is the WITH-SIDECAR
(wrapped) render: `measure_birth_parseback_survival_from_report` scores
`build_model_from_archive`, which applies the target-region action sidecar via
`wrap_model_with_target_region_actions`. So the entire "parse-back collapse to 2" premise that
drove this whole session was the runner **conflating backend + sidecar** — a SURVIVING backend
mislabeled as collapsed because the shipped "rescue" sidecar is CATASTROPHIC.

### The sidecar is the bug (not the backend, not any codec/section/EMA)
The sidecar overwrites 2286 px (`exact_uint8_action_pixels_applied: 2286`) yet collapses
region_hard_won from 11306 -> 3 over the WHOLE 50568-px region — a blast radius ~22x its own
support. That is not plausible from 2286 local overwrites through SegNet's receptive field; it
points to a BUG in `tac.substrates.hi_nerv.target_region_actions.TargetRegionActionReceiver.forward`
(the overwrite writes `rgb_u8/255.0` into the [0,1] frame; render range goes 0.98 -> 1.00 with
the sidecar) corrupting the frame so SegNet's argmax flips region-wide. The "rescue" overlay is
DESTROYING an otherwise-surviving birth.

### Consequences (re-prioritized roadmap)
1. **The collapse chase is OVER**: there is no backend collapse to explain. Sections/latents/
   int4/EMA/build-path were all chasing a phantom created by the runner's backend+sidecar conflation.
2. **Fix the survival measurement**: `measure_birth_parseback_survival_from_report` must score the
   BACKEND (unwrapped) separately from the with-sidecar render, and report BOTH — a surviving
   backend must not be labeled collapsed by a harmful sidecar.
3. **Fix or disable the sidecar**: `TargetRegionActionReceiver.forward` overwrite collapses the
   render; until fixed, the sidecar is net-harmful and must NOT be bundled (it adds 8128 bytes AND
   destroys ~11k wins).
4. **Then compute exact d_seg/d_pose/bytes for the BACKEND-only archive** and feed LoweringRace —
   the backend may already be a strong frontier candidate (it preserves the birth).
5. **MLX-port hardening (operator TOP priority)**: backend wins 11306 (MLX) vs 11297 (torch) = 9px
   region drift / 18px full-frame; drive to argmax-exact via the fidelity gate.

## DO NOT
- name a guilty decoder section (H1 superseded; the selected codec is int8, faithful in the grid).
- implement section QAT or decoder-codec QAT (int8 is faithful; int4 isn't what ships; the gap is build-path).
- treat the v9 grid cells as selected-archive evidence — they are counterfactual same-export packs.
- call int16_raw "the fix" — its win-preservation is on a DIFFERENT (sidecar-less) build path than ships.
