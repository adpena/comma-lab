# Branch-A runbook — byte-close → authoritative exact eval for the decisive run (STAGED, zero-latency)

**Purpose.** When the decisive run (`yousfi_r3_taper_marginhinge_e5_20260620`) finishes its finisher stages
and its **advisory** contest-CPU S crosses the frontier, convert it to an **authoritative** exact row and, if
it beats 0.19110, **update the pointer** — with no setup latency. The whole chain is PROVEN for this exact
vehicle (G3/#127, 2026-06-18), so this is a *reuse*, not a build.

**Authority recap (the de-risk):** G3 measured advisory(macOS in-Python) ↔ exact contest-CPU gap ≈ **0.001%**
on the real 600-pair video. So the run's **live advisory S already tells us whether it will cross** — fire the
paid eval only when advisory S is below 0.19110 (decide-don't-defer). contest-CPU is the leaderboard axis;
contest-CUDA drifts (G3: d_seg +1%, d_pose +41%) and is carried, not used for the pointer.

## Fire condition (read off the live dashboard / summary)
The binding term is d_seg. Using the run's current pose+rate (~0.108), the live break-even:
* **beat 0.19110 → advisory d_seg < ~8.1e-4** (≈2.5× below the current 0.00207 plateau)
* **sub-0.15 → advisory d_seg < ~4.0e-4**
Fire when `torch_vehicle_summary.json:last_eval.score` (or `best_score`) < 0.19110.

## The proven assets (all exist, verified 2026-06-22)
* **Payload:** `experiments/results/yousfi_r3_taper_marginhinge_e5_20260620/best/best_archive.bin` (EMA-shadow,
  byte-closed by the run; dry byte-close → archive.zip verified OK, member `0.bin`, ZIP_STORED, deterministic mtime).
* **Byte-close + NO-FAKE parity actuator:** `tools/verify_e2e_byte_close_eval.py` (G2 fixed-point parity contract).
* **Inflate runtime tree (PROVEN closure):** `experiments/results/g3_torch_vehicle_bc20_packet_20260618T012713Z/submission_dir/`
  = `0.bin` + `archive.zip` + `inflate.py` (66 LOC; deps torch+F+numpy) + `inflate.sh` + `src/` + `README.md`.
  Runtime-closure proof: `.omx/research/g3_torch_vehicle_bc20_runtime_closure_proof.json` (PASS, 1200 frames,
  `inflate.sh -> python inflate.py archive/0.bin inflated/0.raw`).
* **Authoritative host:** Modal — `experiments/modal_auth_eval_cpu.py` (CPU Linux x86_64) + the CUDA variant.
  Dispatch template: `.omx/research/g3_paired_dispatch_plan_20260618T013243Z.json`. Cost ~$1–3 for the pair
  (within the operator-approved $20 Modal budget). molt is NOT needed (and currently requires Tailscale auth).

## Steps (fire-time)
1. **Byte-close + parity** the finisher checkpoint:
   `.venv/bin/python tools/verify_e2e_byte_close_eval.py <args for the yousfi_r3 best/>` → archive.zip + parity_ok.
   (Or minimal: `tools/build_torch_vehicle_d2_archive_zip.py --input-bin .../best/best_archive.bin
   --output-zip <pkt>/archive.zip --member-name 0.bin`.)
2. **Assemble the submission packet:** clone the G3 runtime tree (`cp -r <g3 submission_dir> <new pkt>`), then
   drop in the finisher's `archive.zip` + `0.bin`. **VERIFY the inflate reconstructs THIS run's taper**
   `[16,16,17,19,19,14,10]` (if the G3 inflate.py hardcodes a different taper, update the taper param) — this
   is the one fire-time check.
3. **Local runtime-closure smoke (GATE, $0):** run `inflate.sh -> inflated/0.raw` locally and assert the
   inflated frames match the run's in-process render (parity). This catches any taper/grammar mismatch BEFORE
   spending. Scratch goes to the SSD tier (`/Volumes/VertigoDataTier/pact/.omx_tmp/...`) and auto-cleans
   (~3.66 GB raw, rebuildable).
4. **Claim the lane:** `tools/claim_lane_dispatch.py claim --lane-id lane_yousfi_r3_bc20_exact_eval_<date>_contest_cpu ...`
   (+ the CUDA lane). Cross-agent coordination per CLAUDE.md.
5. **Fire the paired Modal eval** (reuse the g3 dispatch commands, repointed to the new archive + sha):
   `.venv/bin/modal run --detach experiments/modal_auth_eval_cpu.py --archive <pkt>/archive.zip
   --expected-archive-sha256 <sha> --inflate-sh inflate.sh --output-dir experiments/results/modal_auth_eval_cpu/<...>
   --pair-group-id <...> --lane-id <...>` + the CUDA variant.
6. **Harvest + recompute** (the rounded `final_score` field LIES — recompute by hand):
   `S = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37_545_489` on BOTH axes.
7. **Verdict + pointer:** if **contest-CPU S < 0.19110** → run
   `scripts/pre_submission_compliance_check.py --contest-final --strict` (expected sha/size/auth-eval JSON),
   then update `.omx/state/canonical_frontier_pointer.json` (the FIRST ours-trained pointer move). If only
   close, record the calibration row + go to branch B (relaunch alt stage-8 finishers from a preserved
   `stage_snapshots/<stage>_end`).

## Operator-gated at fire-time (1 thing)
The paired Modal eval is a ~$1–3 paid dispatch. It's within the approved $20 Modal budget and is the RIGHT
default the moment advisory S crosses (per CLAUDE.md "spend to BUY exact rows"). I will surface the advisory
crossing + the fire-or-hold decision rather than auto-spend.

## NO-FAKE ledger
- Byte-close PROVEN on the live run's best_archive.bin (this turn). Inflate runtime + Modal dual-eval PROVEN
  for this vehicle class (G3/#127). Advisory↔contest-CPU ≈0.001% (G3).
- NOT claimed: no score moved; pointer UNMOVED 0.19110; the authoritative row only exists after upstream/
  evaluate.py runs on the byte-closed bytes. The fire condition (advisory d_seg < ~8.1e-4) has NOT been met
  (current 0.00207).

Cross-refs: `g3_torch_vehicle_bc20_first_exact_row_20260618T0135Z.md` (the proven dual row) ·
`tools/verify_e2e_byte_close_eval.py` · `tools/build_torch_vehicle_d2_archive_zip.py` ·
`decisive_run_161_config_and_deepmath_optimality_audit_20260622.md`.
