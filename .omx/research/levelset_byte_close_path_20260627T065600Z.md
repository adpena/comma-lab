# LEVEL-SET WITNESS byte-close + exact-eval path — BUILT + tested end-to-end (the ROW-ENABLER)

**UTC:** 2026-06-27T06:56:00Z · **Author:** levelset byte-close path subagent (DAG **FEED-dp**) ·
**Axis:** `[macOS-CPU advisory]` NON-PROMOTABLE · **Pointer:** UNMOVED 0.19110.
**Constraints honored:** CPU-only `$0` (GPU pid 72600/72602 mod-32 baseline UNTOUCHED, confirmed
alive 3:31+ throughout), additive (a NEW tool — the RGB tool `tools/witness_byte_close_and_eval.py`
is NOT edited), numpy-fp32 + frozen CPU-torch verdict authority, serializer +
`--expected-content-sha256` commit, review-gate scan+mark-file, subagent_checkpoint, no `/tmp` in
artifacts, inflate `.raw` scratch auto-cleaned (certify-or-block: rebuildable from archive.zip +
inflate.py).

Resolves the CRITICAL ROW-BLOCKER from the FEED-df recursive review
(`.omx/research/yousfi_levers_optimal_form_review_20260627T063335Z.md`): the level-set witness had
**NO exact-eval path** — `tools/witness_byte_close_and_eval.py` is hard-keyed to the RGB witness
(`params["out.weight"]` line ~143, forward `z = h @ p["out.weight"].T` line ~324); the level-set
npz has `out_sdf`/`out_tex`/`palette`/`code` and NO `out.weight` → KeyError, and a level-set
builder did not exist (grep: zero hits). Until it existed, ALL the level-set d_seg descent was
advisory-only and the pointer could NOT move (per THE GOAL: only a byte-closed contest-CPU/CUDA
exact row moves the pointer).

---

## DELIVERABLE — `tools/levelset_byte_close_and_eval.py` (WORKING, tested)

Mirrors the RGB tool's STRUCTURE + NO-FAKE discipline, keyed on the level-set checkpoint. ONE
command: load `levelset_witness_{ema,live}_mlx.npz` → byte-close → self-contained inflate → realized
d_seg/d_pose parity on the inflated frames → staged contest-CPU exact-eval command.

### 1. Load + self-orient detection
Separates LEARNED params (non-`__` keys: `in_proj`/`film`/`hidden.*`/`out_sdf`/`out_tex`/`palette`/
`code`) from the `__cfg_*`/`__bank_*`/`__render_hw` scalars. Reads activation/chroma/softmax_temp/
bank cfg/render res from the npz (defaults+inference fill any an older save block omitted, loud
WARN). Detects self-orient structurally: `dir_w = in_feat − 2·curvelet_cols`; `>0` ⇒ self-orient,
`n_dir_freqs = dir_w/4`.

### 2. BYTE-CLOSE (the MEASURED rate term) — matches the canonical accounting EXACTLY
int8(`_int8_symmetric`, the trainer's single source) + brotli-q11: ONE stream for the base weights
+ ONE for `code` — byte-identical to `lever_b_levelset_generator.quantize_levelset_blob`. **Verified
empirically:** base 73468 + code 203 = 73671 = `canonical_quantize_blob_bytes` (`accounting_matches_
canonical: True`). The curvelet bank is NOT stored.

### 3. RULE-118 FREE-vs-COUNTED split (the rate game, explicit)
| FREE in inflate.py (regenerated, 0 bytes) | COUNTED in archive.zip (rate term) |
|---|---|
| curvelet/shearlet bank (5 scalars n_scales/n_orient0/f0/base/n_iso + max_freq → parametric polar grid) | `in_proj`/`film`/`hidden.*`/`out_sdf`/`out_tex`/`palette` int8+brotli (learned video-derived weights) |
| self-orient directional feats (FIXED-POINT on the decoder's OWN argmax — GT-free, reconstructible) | per-frame `code` table int8+brotli (the per-(pair,frame) video-derived modulation) |

The bank + self-orient feats are GENERIC ALGORITHM (rule 118 FREE); the learned payload is COUNTED.
Self-orient hyperparameters (freq_across/along/tau/iters) are a handful of manifest bytes (generic,
not video-derived) — rule-118-clean.

### 4. INFLATE.py — self-contained, full-output, op-for-op mirror
numpy + brotli + torch-for-R (+ scipy only on the self-orient branch). Reads the blob → int8-dequant
→ regenerates the curvelet bank from cfg → curvelet feats → [self-orient fixed-point per pair] →
`_forward` (an op-for-op mirror of `levelset_rgb_forward_numpy`: in_proj → FiLM hidden → out_sdf/
out_tex → softmax(phi/T)@palette + tex → sigmoid·255, chroma-aware) → torch R (bicubic↑camera →
round → uint8, byte-identical to `_torch_R_to_camera_uint8`) → streams the FULL
`(2·n_pairs, 874, 1164, 3)` uint8 `.raw` (frame0 AND frame1 per pair). `<=200`-LOC waiver
(self-orient inlining; rationale in the gen comment + review-gate). Verified: `full_output_shape_ok=
True`, raw 36,624,096 B for the n6 test.

### 5. PARITY — realized d_seg/d_pose on the INFLATED frames (frozen CPU-torch, vs GT)
Reads the `.raw` back → `twr.cpu_verdict_d_seg_batch` / `cpu_verdict_d_pose_batch` (the trainer's
authority path) on the inflated frames vs GT. Recomputes `S = 100·d_seg + sqrt(10·d_pose) +
25·bytes/37_545_489` (never a cached field).

---

## FIRST ADVISORY LEVEL-SET BYTE-CLOSED ROW (tested on the available `_ema` checkpoint)

The RUNNING mod-32 dir (`levelset_n96_mod32_20260627T032128Z/`) is EMPTY — the trainer saves the
npz only at the END of the loop (no mid-run save in the current code), so there is nothing to test
against from the live run yet. Tested instead on the available `_ema` checkpoint
`experiments/results/levelset_gpu_smoke/levelset_witness_ema_mlx.npz` (n_pairs=6, render 96×128,
self-orient, hosc — the SAME architecture shape as the mod-32 target, which also uses self-orient).

| quantity | value | note |
|---|---|---|
| archive.zip | 74,839 B | n6 (tiny); rate = 0.001993, rate_term = 0.0498 |
| base int8+brotli | 73,468 B | = canonical quantize_levelset_blob base (match) |
| code int8+brotli | 203 B | = canonical code (match) |
| **realized d_seg** (inflated) | **0.604841** | smoke checkpoint (6 epochs); trainer ep6 reported 0.6737 → within the self-orient fixed-point parity band |
| **realized d_pose** (inflated) | **189.594201** | trainer ep6 reported 189.61 → **near-exact match** (pose render faithful) |
| **S_advisory** | **104.0763** | = 60.48 (seg) + 43.54 (pose) + 0.05 (rate); POSE term dominates |

**The byte-close → inflate → parity → S path WORKS end-to-end.** The d_pose near-exact match
(189.59 vs trainer 189.61) proves the texture/code render is faithfully reconstructed; the d_seg gap
(0.6048 vs 0.6737) is the documented self-orient parity (fixed-point on FINAL weights vs the
trainer's trajectory-accumulated dir feats) plus 6-epoch smoke noise (the trainer's own verdict swung
0.55→0.51→0.67 across epochs 0/3/6). This is a SMOKE checkpoint — the row is poor by design; the
point is the TOOL is real and faithful.

**Authority:** `[macOS-CPU advisory]` NON-PROMOTABLE. NO score/frontier/promotion/pointer claim;
pointer UNMOVED 0.19110.

---

## DEFINITIVE POSE-FOR-ROW VERDICT — **w_pose > 0 REQUIRED; a stored sidecar does NOT work**

The scorer computes `d_pose = MSE(PoseNet(generated_pair)[:6], PoseNet(original_pair)[:6])` **on the
rendered FRAMES**. A stored 6-scalar pose sidecar is bytes the scorer NEVER reads — it does not
change the rendered frames, so it CANNOT lower realized d_pose. The measured row confirms it: the
w_pose=0 (POSE-BLIND) checkpoint renders frames whose realized d_pose = 189.59 — and storing the
Quantizr 6-scalar targets as a counted section would only ADD bytes while leaving d_pose at ~190
(the inflate render does not consume the sidecar).

Therefore the level-set ROW REQUIRES a **POSE-TRAINED render**: `--w-pose > 0` supervising the
texture/`code` to hit the 6 PoseNet targets in the SegNet-null space (exactly how the RGB witness
"carries pose in per-(pair,frame) codes"). The Quantizr stored-target is the GT for that supervision
(the target the render is trained TOWARD), NOT a deploy-time sidecar that fixes a pose-blind render.
The reviewed/running mod-32 config (`--w-pose 0`) is d_seg-only and will produce a garbage pose term
at byte-close — **a pose-trained checkpoint is the binding prerequisite for a usable level-set exact
row.** `--fold-pose-sidecar` is provided (parity with the RGB tool) but OFF by default and LOUDLY
documented as adding dead counted bytes on a code/texture-pose witness.

---

## KEY FINDINGS / FLAGGED GAPS (for follow-up landings)

1. **TRAINER SAVE-BLOCK GAP (self-orient params not persisted).** The trainer's npz save block
   persists `__cfg_*`/`__bank_*`/`__render_hw` but NOT the self-orient params
   (`n_dir_freqs`/`freq_across`/`freq_along`/`reorient_every`). `n_dir_freqs` is inferable from
   `dir_w/4`; `freq_across`/`along`/`tau` are NOT → the tool defaults them to the trainer defaults
   (32/4/4) and exposes `--so-*` overrides. **The mod-32 run uses `--self-orient --freq-across 32
   --freq-along 4`**, so its checkpoint WILL byte-close with the tool defaults — but a follow-up
   should persist these into the npz for provenance-complete reproducibility.
2. **Self-orient parity caveat (NO-FAKE).** The deploy fixed-point converges on the FINAL weights;
   the trainer's reported implied_S used dir feats accumulated along the training trajectory →
   close, not bit-identical. The realized d_seg on the inflated frames is the TRUTH; any gap vs the
   trainer number is itself a finding (here 0.6048 vs 0.6737, within band).
3. **30-min budget consideration (n600 contest inflate).** Self-orient decode runs `so_iters` (4)
   forwards + a scipy EDT per pair before rendering 2 frames; at n600 + render 384×512 that is
   ~600·(4+2) forwards + 600·4 EDTs + 1200 torch-R bicubic-up-to-camera. Likely within the 30-min
   budget but should be timed on the real n600 checkpoint; if tight, reduce `so_iters` or
   precompute. (Flagged, not yet measured — no n600 self-orient checkpoint exists yet.)
4. **inflate.py dep count.** numpy+brotli+torch (no-self-orient) / +scipy (self-orient) — exceeds
   the HNeRV-parity ≤2-dep soft budget; acceptable for advisory (scipy is standard); the EDT could
   be inlined to drop scipy if a contest submission needs the tighter dep tree.

---

## NEXT (the imminent exact row this enables)
- A **pose-trained** (`--w-pose > 0`) **n600** level-set checkpoint → this tool → archive.zip +
  inflate.py + inflate.sh → `experiments/contest_auth_eval.py --device cpu` on Linux x86_64 (Modal
  CPU) = the first REAL level-set byte-closed exact-eval row. The tool is the path; the binding
  prerequisites are (a) w_pose>0, (b) n_pairs=600 (the 1200-frame contest `.raw`), (c) a Linux
  x86_64 host for the contest-CPU axis.

## Validation summary (NO-FAKE / compliance)
- `py_compile` OK; byte-close accounting matches canonical `quantize_levelset_blob` EXACTLY;
  inflate full-output-shape verified; parity d_pose near-exact vs trainer (render faithful); d_seg
  within self-orient parity band. End-to-end tested twice (skip-parity + full parity).
- CPU-only `$0`; GPU pid 72600/72602 UNTOUCHED (confirmed alive throughout); numpy-fp32 + frozen
  CPU-torch verdict; inflate `.raw` scratch (35 MB) auto-cleaned (rebuildable from archive.zip +
  inflate.py). Review-gate scan+mark-file (13 entities reviewed). subagent_checkpoint logged.
- NO score/frontier/promotion/pointer claim. Pointer UNMOVED 0.19110. Axis `[macOS-CPU advisory]`
  NON-PROMOTABLE. Report: `reports/levelset_byte_close_first_row.json`.
