# #205 per-stage byte-close → EARLY exact-row plan (readiness + prepped dispatch)

**Date:** 2026-07-02 · **Track:** END-oriented (the only thing that moves the pointer is a byte-closed
`upstream/evaluate.py` n600 exact row < 0.19110). **Authority:** `[macOS-CPU advisory] NON-PROMOTABLE`.
Canonical frontier pointer **0.19110 UNMOVED**. `score_claim=false, promotable=false`. NO paid dispatch
fired; NO GPU launch; NO trainer edits (READ-only on the run). Every advisory S here is the frozen
CPU-torch mirror; the ONLY score is `upstream/evaluate.py` on the exact archive bytes (CPU+CUDA).

---

## 0. TL;DR (the headline first)

1. **⛔ BLOCKER — #205 OOM'd and DIED before any checkpoint.** The daemon run
   `experiments/results/levelset_n600_witness_20260702T210653Z` was **killed at the 90 GB RSS guard**
   (`status=oom exit=137 peak_rss=90300MiB elapsed=612.03s`) ~10 min in, in the INIT stage — it never
   reached epoch 25 (the first `--ckpt-every` checkpoint). **There is NO #205 checkpoint on disk.** The
   sealed config (n600 + `--pose-carrier` + `--lane-render-band` + persistence + island-amplify +
   structured-init all active) exceeds the 90 GB RSS ceiling. **A #205 early exact row is impossible
   until the run is relaunched and survives to epoch ≥25.** (Scale-measured/safeguarded non-negotiable:
   the guard did its job — no machine-kill — but the run produced 0 rows.)
2. **✅ Keyframe-accounting gap CLOSED (correctness fix, committed `f7c6abdea`).** `compose_witness_archive.py`
   now carries the real-luma pose-carrier keyframes as an EXPLICIT COUNTED rate line item (rule-118:
   keyframe bytes counted, warp decoder free). Honest known-store rate corrected from the borrowed
   partition figure (~0.006) to the MEASURED real-luma payload (~0.022–0.067; §2). 6 tests; phase_a +
   phase_b run clean.
3. **✅ Byte-close PATH verified end-to-end** on the closest n600 proxy checkpoint (`levelset_n600_v2_attrclean`):
   archive.zip → inflate (full 1200-frame output) → realized d_seg/d_pose parity → advisory S + honest
   rate breakdown all work. Advisory row below.
4. **⛔ SECOND BLOCKER (pose axis) for #205's sealed config:** the level-set byte-close tool
   (`tools/levelset_byte_close_and_eval.py`) does **NOT reproduce the `--pose-carrier` warp-real-luma-frame0
   decode**, nor count its keyframe payload. A #205 pose-carrier checkpoint byte-closed by this tool today
   would be **POSE-BLIND** (inflate renders frame0 from the INR, not the warp) AND under-count rate. The
   d_seg/rate half is ready; the pose half needs the carrier decode + keyframe section wired into the
   levelset inflate (see §4).

---

## 1. #205 run status (READ-only forensic)

- Run dir: `experiments/results/levelset_n600_witness_20260702T210653Z/` (only `launch.sh` + `run.log`).
- Launched: 2026-07-02T21:06:53Z via `tools/safe_run.py --rss-mb 90000 --timeout 1209600`.
- Died: OOM at 612.03 s (`SAFE_RUN ... status=oom exit=137 peak_rss=90300MiB`), still in the init stage
  (log stops after `island_amplify`; no `eval`/`ckpt`/epoch lines).
- Provenance from log: `git_sha=7521a49fe`, seed 0, curvelet+self_orient front-end, GROUPED_BACKWARD active,
  pose_carrier residual_mode=table (s_t=0.044, s_t_fit best 2.562 @0.044), lane_render_band start_epoch 300,
  persistence classes [1,3], w_seg 100 / w_pose 1.0, epochs 1000, muon_start 726, l7_start 1000, ckpt_every 25.
- **Consequence:** no rolling `resume_state`, no stage ckpt, no EMA npz. Nothing to byte-close from #205.
- **Operator action (not this track):** relaunch with either a higher `--rss-mb` on a machine with more RAM,
  or reduce peak memory (e.g. smaller `--accum-pairs`, defer lane-band/persistence init memory, stream GT)
  so the run reaches epoch ≥25 and writes the first checkpoint. Only then does the early-row loop below fire.

---

## 2. Task 1 — keyframe-accounting fix (committed `f7c6abdea`)

**What:** `tools/compose_witness_archive.py` accounted PARTITION keyframes (5-class argmax label maps in
`store_blob`) + the pose sidecar, but silently OMITTED the **real-luma warp keyframes** the `--pose-carrier`
path stores. `src/tac/boundary_math/warp_real_luma_frame0.py` (lines 46-51) explicitly flags this payload
as COUNTED and "the vehicle's ... concern ... NOT smuggled into this module's byte claim." This fix makes
the vehicle count it.

**How (rule-118 clean):**
- New helper `keyframe_payload_accounting(args)` → explicit `keyframe_blob_bytes` + `keyframe_blob_rate`.
  Source priority: `--keyframe-payload-path` (counts a real blob's `st_size`) > `--keyframe-payload-bytes`
  (explicit measured codec byte count) > default 0.
- phase_a byte-budget: honest `known_store_total_bytes_proj600` = partition store + pose sidecar + **keyframe
  payload** (kept `known_store_excl_keyframes_bytes_proj600` for continuity); break-even + sub-0.15 d_seg
  recomputed on the honest total.
- phase_b: `honest_rate_incl_keyframes` + `keyframe_in_this_zip=False` (the stored-sidecar archive does not
  pack keyframes; accounting-only, loud — no dead bytes fabricated).
- Default 0 with a NO-FAKE note: 0 = stored-sidecar pose path (dead bytes for d_pose); a real-luma
  `--pose-carrier` ROW REQUIRES payload > 0.

**Measured correction (phase_a, n96 cache, `--keyframe-payload-bytes 32875`):** partition store 11587 B +
pose 1160 B + **keyframe 32875 B** → honest known-store 45622 B, rate **0.0304** (vs 0.0068 excl-keyframes);
break-even d_seg tightened 0.00187 → **0.00142**. The real-luma keyframe payload dominates the known store.
Codec-measured range for 13 keyframes (memo `warp_keyframe_payload_rate_minimization_20260702` §2): rate
**0.006** (192×256 HEVC crf40, cliff-risk) → **0.022** (384×512 HEVC crf34) → **0.067** (384×512 HEVC crf28,
d_pose-safe). The cheap-vs-expensive branch is decided by the residual-compensation measurement (memo §4.2,
needs a trained #205 residual — hence blocked on the relaunch).

---

## 3. Task 2 — byte-close PATH verified (advisory proxy row)

#205 has no checkpoint, so the PATH was verified on the closest existing n600 checkpoint
(`levelset_n600_v2_attrclean_20260630T194549Z`: n600, self-orient, hosc, chroma — the CORE levers; w_pose=0,
no pose-carrier/lane-band). Command:

```bash
.venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir experiments/results/levelset_n600_v2_attrclean_20260630T194549Z \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --max-pairs 4 --keep-packet --out reports/levelset_byteclose_pathcheck_attrclean_20260702.json
```

**Result (advisory):** archive.zip **82853 B** (sha `6431609d…`), rate **0.002207**, rate_term **0.0552**.
Honest rate breakdown: base int8+brotli **55931 B** + code int8+brotli **25705 B** (+magic/prefix) = 0.bin
83550 B; curvelet bank + self-orient dir feats = FREE (rule-118). Inflate full-output (8,874,1164,3) OK.
Realized (4 pairs): **d_seg 0.00395**, **d_pose 101.63 (POSE-BLIND, w_pose=0)** → S_advisory 32.33
(pose-dominated garbage, expected). n_pairs=600 → contest-ready 1200-frame `.raw`.

**Takeaway:** the d_seg + rate byte-close path is proven end-to-end on the real level-set format. The pose
term is garbage here because it's a w_pose=0 checkpoint (and, for a pose-CARRIER checkpoint, would ALSO be
garbage — see §4).

---

## 4. The pose-carrier decode BLOCKER (why #205's sealed config is not yet a full exact row)

`tools/levelset_byte_close_and_eval.py` renders BOTH frames of each pair from the witness INR
(`levelset_rgb_forward_numpy`). It has **no reproduction of the `--pose-carrier` warp-real-luma-frame0
decode** (grep-confirmed: no `pose_carrier`/`warp_real`/`keyframe` path in the tool or in
`src/tac/local_acceleration/torch_levelset_inflate.py`). #205's SEALED config carries pose via the carrier
(even frames = warp(real-luma f0, ξ), odd = witness f1; `--w-pose 1.0 --pose-carrier residual-mode table`),
NOT via the INR texture. So a fully-trained #205 checkpoint byte-closed by this tool today would:
1. render frame0 from the INR (≈ frame1) → PoseNet sees ~no motion → **POSE-BLIND** (d_pose O(100)); and
2. **under-count rate** (no keyframe payload).

**To make #205's pose-carrier EXACT row honest, two wiring tasks (out of scope here, flagged):**
- (A) Reproduce the warp-real-luma-frame0 carrier in the levelset inflate: store real keyframes + per-pair ξ,
  warp frame0 at decode (the FREE homography/exp_se3/bilinear/R; the twist is DUAL-USE with the pose sidecar).
- (B) Count the keyframe payload in the levelset byte-close rate (the same explicit line item this task added
  to `compose_witness_archive.py` — port it to `levelset_byte_close_and_eval.py`).

**The d_seg + rate half is ALREADY row-ready** — a #205 checkpoint (once it exists) can produce an honest
d_seg + rate advisory + exact row TODAY via §5; only the pose term needs (A)+(B). (The compose/v2 hybrid
pipeline's residual-INR pose path is the sister route where pose lives in the INR, not the carrier.)

---

## 5. Task 3 — PREPPED (NOT FIRED) exact-eval commands

**Fire ONLY on a real, good #205 checkpoint (operator GO). CONTAINMENT: local advisory is $0; contest CPU/CUDA
is the pointer-mover — operator fires it.** All commands assume the relaunched #205 run dir `<RUN>` with a
checkpoint npz present.

**(a) Local advisory byte-close + realized parity (macOS-CPU, $0, NON-PROMOTABLE):**
```bash
.venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir <RUN> \
  --npz-name <levelset_witness_ema_mlx.npz | levelset_ckpt_<stage>_ep<N>.npz | levelset_witness_ema_BEST.npz> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --lane-render-band --lane-band-dash-forward-max 55.0 \
  --max-pairs 8 --keep-packet \
  --out reports/levelset_byteclose_n205_<stage>.json
# -> advisory d_seg + honest rate (+ POSE-BLIND until §4 (A)+(B) land)
```

**(b) REAL exact eval — CPU axis (contest-CPU, authoritative ONLY on Linux x86_64 / Modal CPU; macOS = advisory):**
```bash
.venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir <RUN> --npz-name <ckpt.npz> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --run-exact-eval --eval-device cpu --memory-tier decode_cpu_16gb \
  --uncompressed-dir upstream/videos --video-names-file upstream/public_test_video_names.txt \
  --keep-packet --out reports/levelset_exact_cpu_n205_<stage>.json
# forces all 600 pairs; ~1-2h CPU. Runs upstream/evaluate.py on the exact archive bytes.
```

**(c) REAL exact eval — CUDA axis (contest-CUDA, T4/A100 host):** same as (b) with
`--eval-device cuda --memory-tier decode_t4_16gb`. Or the staged
`experiments/contest_auth_eval.py --archive <packet>/archive.zip --inflate-sh <packet>/inflate.sh --device {cpu,cuda}`
(emitted as `contest_cpu_eval_cmd` in the byte-close report).

**Modal <$5 dispatch (buys the real rows):** dispatch the byte-closed packet to a Modal Linux x86_64 CPU
container running (b) for the contest-CPU axis, and a T4 for (c) — canonical dispatch path per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA". Both axes required for a submission claim; neither inferred from
the other. Fire only after the local advisory (a) confirms a sub-0.19-plausible d_seg + rate.

---

## 6. Task 4 — early-exact-row plan (which checkpoint → which row)

Once #205 (relaunched) writes checkpoints (`--ckpt-every 25`, `--stage-checkpoints`), each is an independent
early row. Priority by expected d_seg maturity (the pose term rides §4's wiring):

| checkpoint | epoch | why it's a row | advisory (a) | exact CPU (b) | exact CUDA (c) |
|---|---|---|---|---|---|
| rolling `levelset_witness_ema_mlx.npz` | latest | cheapest signal; watch d_seg descent | every 25 ep | — | — |
| `levelset_ckpt_tau_ep<300+>` | tau-converged (post curriculum tau @300) | first stable partition → first real d_seg row | yes | ✅ if d_seg < ~0.0012 | after CPU |
| `levelset_ckpt_muon_ep<726+>` | Muon finisher | orthogonalized finisher; PR95 places Muon last | yes | ✅ | ✅ |
| `levelset_witness_ema_BEST.npz` | best d_seg | the lowest-d_seg row to date | yes | ✅ (the headline row) | ✅ |
| final `ep1000` | end | full-curriculum row | yes | ✅ | ✅ |

**Promotion path:** local advisory (a) on every checkpoint → when a checkpoint's advisory d_seg + honest rate
projects sub-0.19 (S = 100·d_seg + √(10·d_pose) + rate_term), fire (b) contest-CPU (Modal, <$5) → if it holds,
fire (c) contest-CUDA → update `.omx/state/canonical_frontier_pointer.json` ONLY on a real exact row < 0.19110.
The pose term is NOT trustworthy until §4 (A)+(B) land; until then, treat the exact-eval S as a d_seg+rate
LOWER-bound witness (pose garbage), and prioritize (A)+(B) so the pose-carrier row's S is honest.

**Sequence:** [operator relaunch #205 → survive to ep25] → §5(a) advisory loop → wire §4 (A)+(B) for the honest
pose → §5(b)/(c) exact rows on tau/Muon/BEST/final. This track holds the byte-close + accounting ready; the
pointer moves only on the real exact row.

---

## 7. Provenance / cross-refs

- Fix commit: `f7c6abdea` (compose_witness_archive keyframe_blob line item + 6 tests; all pass; phase_a+phase_b
  run clean). Proxy byte-close report: `reports/levelset_byteclose_pathcheck_attrclean_20260702.json`.
- Sources: `warp_keyframe_payload_rate_minimization_20260702.md` (the measured keyframe rate surface + the §6
  step-5 accounting fix it requested), `warp_real_luma_frame0.py` docstring (the COUNTED keyframe dependency it
  flags), `tools/levelset_byte_close_and_eval.py` (the level-set exact-eval path), `tools/safe_run.py` (the RSS
  guard that OOM-killed #205). Pointer **0.19110 UNMOVED**; every number `[macOS-CPU advisory] NON-PROMOTABLE`.
