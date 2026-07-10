# Click-polish phase-2 — Modal CPU exact-gated search + eval runbook (task #399)

**Date:** 2026-07-10 · **Lane:** `lane_clickpolish_pr110_frontier_20260710` · **Axis:** contest-CPU (Linux x86_64)
**Pointer UNMOVED at contest-CPU 0.19110 — this runbook is MEANS toward a lower exact row (a defensive bank).**

Phase-1 harness (COMMITTED `21c209150`, all greens): `src/tac/click_polish.py` + `tools/click_polish_exact_search.py` + `src/tac/tests/test_click_polish.py`.

## Phase-1 greens (measured, this host, macOS-CPU)
- **Round-trip byte-exact** (`verify`): `encode(decode(archive))==archive`, sha `b46897267ded…` reproduced exactly (177169 B). No-op detector passes at full-archive level.
- **Fold-sidecar custody**: dropping the 607-B sidecar section = exactly **−607 B** (before sha = frontier sha; after sha recorded). Byte-exact reversible accounting.
- **Pair-locality** (`verify-render`): clicking pair_a leaves pair_b's frames BYTE-IDENTICAL in the same batch (no cross-talk) — MEASURED on our decoder, not assumed.
- **Diagonal-batch ≡ sequential**: seg-argmax EXACT across batch sizes; d_pose differs only by CPU batch-float nondeterminism (~8e-9, far below the click gate). The search's accept gate is EXACT re-render (canonical 16-batch layout), so batch-float never corrupts an accept.
- **n4 end-to-end smoke** [macOS-CPU advisory, NON-PROMOTABLE]: round-0 accepted 4 monotone clicks, banked a valid candidate archive, ledger fsync'd + resumable.
- **17 tests green** (13 fast + 4 render).

## Measured cost/time (macOS-CPU, UPPER bound for Modal)
- render+score ≈ **0.39 s/pair**; one full 600-pair eval ≈ **~4 min**; renders/round = `4 deltas × 28 dims + 1 = 113`.
- **≈ 7.4 hr/round** at n600 with the full ±1/±2 sweep (UPPER bound; Modal `cpu=8` is faster).
- **Cheaper knob:** `--sweep-deltas` ±1 only halves round cost (`2×28+1 = 57` renders → ~3.7 hr/round), do more rounds.
- Modal CPU class ≈ $0.01–0.03 / core-hr (cost_band_calibration.py) → an 8-core box ≈ **$0.08–0.24/hr**. A 2–3 round bounded run ≈ **8–22 hr ≈ $0.7–$2.7**. Cap rounds/wall-clock to stay **well under $2**; **abort if projection > $5** (per #381 envelope ≤$20).

## The dispatch (same-container determinism — MANDATORY, advisory §8.2)
`torch.interpolate(bicubic)` LSBs differ across CPU microarch (PR128's own `expected_output.sha256` is NOT portable). Therefore **SELECTION and EXACT EVAL run in the SAME Modal container/microarch.** Never assume a decode sha is portable; verify in-container. This is why the search + eval must be ONE dispatch (not "search on macOS, eval on Modal").

### Modal image (reuse the auth-eval CPU image pattern, `experiments/modal_auth_eval_cpu.py`)
Deps ALL already present: `torch` (CPU wheel), `numpy<2`, `constriction>=0.4,<0.5`, `safetensors`, `timm`, `segmentation-models-pytorch`, `av`, `einops`, `brotli`. **PIN** in the image (advisory §2 — constriction unpinned upstream): `constriction==0.4.2`, `torch==2.12.1` (or the image's `2.5.1` — self-consistent in-container is what matters), `numpy==1.26.4`. Mount (copy=True at build): `upstream/` (incl. `videos/` + `models/*.safetensors` + `public_test_video_names.txt`), the frontier `submission_dir/` (archive + `src/` + `inflate.py/.sh`), `src/tac/`, `tools/click_polish_exact_search.py`. Add local python source: `tac`.

### In-container GT build (container-safe, pure torch — NO MLX)
`precompute_gt` drags MLX; use the pure-torch path instead. Build GT targets aligned with evaluate.py's own GT decode:
```python
# decode 600 GT pairs via upstream AVVideoDataset (yuv420_to_rgb — the evaluator's path),
# then extract targets with the frozen scorers (tac.scorer):
from tac.scorer import load_default_scorers, extract_gt_masks, extract_gt_pose_targets
# gt_frames = flat list of (874,1164,3) uint8 tensors [f0_0,f1_0,f0_1,f1_1,...] for n pairs
posenet, segnet = load_default_scorers("upstream", device="cpu")
lstars   = extract_gt_masks(gt_frames, segnet).cpu().numpy()      # (n,384,512)
gt_poses = extract_gt_pose_targets(gt_frames, posenet).cpu().numpy()  # (n,6)
```
(GT-frame decode: iterate `upstream.frame_utils.AVVideoDataset(public_test_video_names, upstream/videos, batch_size, device=cpu, num_threads, seed)` collecting the seq_len=2 frames.) Feed `lstars`/`gt_poses` into `ClickPolishSearch(gt_lstars=…, gt_poses=…, axis_tag="[contest-CPU]")`.

### Search invocation (in-container)
```python
from tac import click_polish as cp
pkt = cp.FrozenPacket.parse("<submission_dir>/archive.zip", "<submission_dir>")
rnd = cp.Renderer(pkt, device="cpu")                 # drop_sidecar=False (safe monotone bank)
scorer = cp.Scorer("upstream", device="cpu")
search = cp.ClickPolishSearch(pkt, rnd, scorer, lstars, gt_poses,
                              out_dir="<vol>/clickpolish", axis_tag="[contest-CPU]",
                              max_rounds=<bounded>)
result = search.run()   # resumable: replays <vol>/clickpolish/accepted_clicks_ledger.jsonl
```
Set `torch.set_num_threads(<cores>)` fixed for cross-run determinism. `search.resume()` runs automatically at the top of `run()`. **Resumability (P0):** the ledger lives on a Modal **Volume** so a preempted run resumes without losing accepted clicks; candidate archive is re-banked every accepted round.

### Exact eval IN THE SAME container (the authoritative row)
After `search.run()` banks `candidate_archive.zip`, build a submission_dir (candidate archive + the frontier `inflate.sh/inflate.py/src/`) and run the proven chain:
```
bash <submission_dir>/inflate.sh <archive_dir> <inflated_dir> upstream/public_test_video_names.txt
python upstream/evaluate.py --submission-dir <submission_dir> --uncompressed-dir upstream/videos \
       --video-names-file upstream/public_test_video_names.txt --device cpu --report report_cpu.txt
```
**Output custody (advisory §5):** assert inflated raw is EXACTLY **3,662,409,600 bytes** (1200 frames × 874×1164×3) — a SHORT raw = evaluator truncation = NO-FAKE failure. Storage preflight ≥ that + archive before inflate (fail closed). Atomic writes (tmp+rename). Assert exact byte-consumption (our `inflate.parse_member` already refuses trailing bytes).

### Custody + harvest
- Lane claim (done): `tools/claim_lane_dispatch.py claim --lane-id lane_clickpolish_pr110_frontier_20260710 --platform modal --instance-job-id <job> --agent clickpolish-build`.
- Record call_id at spawn (`tac.deploy.modal.call_id_ledger.register_dispatched_call_id`); **HARVEST OR LOSE** — schedule harvest same turn: `tools/harvest_modal_calls.py --execute --from-ledger --call-id <fc-…>`.
- Output custody → `experiments/results/clickpolish_pr110_20260710/`: `candidate_archive.zip` + sha256 + `search_result.json` (per-component deltas + `borrowed_substrate_accounting`) + `report_cpu.txt` + `accepted_clicks_ledger.jsonl`.
- The row is a score claim ONLY after `upstream/evaluate.py` on the exact candidate bytes (Linux x86_64) returns; then ingest as `our_local_frontier_contest_cpu` via `tools/refresh_canonical_frontier.py` conventions (CUDA axis separately if promotion language is used).

## Waterline framing (advisory §6)
From 0.19110: **sub-0.19 ≡ −1651.74 archive bytes OR −1297.41 net seg cells** (1 cell = 8.4771e-7 S, 1 byte = 6.6586e-7 S). The safe monotone run (sidecar KEPT) buys the d_seg click gains (PR128 analog ≈ −0.0027 seg over many rounds); the sidecar-fold variant adds −607 B (−0.0004) but must exact-gate the distortion recovery. Frame every measured delta against these.

## Recommended sequencing (validation-first, de-risks the paid run)
1. **Cheap validation dispatch** (~$0.10): the SAME app at n8 / max-rounds 1 / ±1 → proves image build + in-container GT + search + candidate + `evaluate.py` row + custody, and measures REAL Modal sec/pair. (The n8 candidate is not a score claim.)
2. If green + timing projects **well under $2**, launch the **bounded n600** run (detached, resumable, `--sweep-deltas ±1`, capped rounds/wall-clock).
3. Harvest → exact-eval row → pointer ingest.

## Borrowed-substrate accounting (NO-FAKE #7) — DEFENSIVE BANK, not innovation
Mechanism = PR128 click-polish [external unverified]; substrate = OURS (PR110-lineage frontier). NO code lifted from the PR128 tree or haochen-rye/HNeRV|NeRV (those are unlicensed). See `tac.click_polish.borrowed_substrate_accounting()` + `dispatch_discipline()`.

## Triality
- **DSL leg = N/A** — this is a search/polish TOOL, not a training lever (no trainer flag; nothing to fold into `witness_dsl`).
- **Equations leg = deferred** to the first measured exact-eval row (then register the click-polish RD law + the byte/seg-cell waterline equivalents).
- **DAG leg = FEED-clickpolish-build** (appended).

## Dispatch record (phase-2 execution, 2026-07-10)
- `fc-01KX6D6991EY3TM8WPG51MN7YN` — n8 validation, **FAILED (operator lesson)**: spawned without `--detach`; ephemeral-app stop killed the call. Ledgered.
- `fc-01KX6D7KMB1VVQTQWDTTJR28B9` — n8 validation (detached). Search half GREEN in-container (baseline n8-mean S 0.18511673; round-0 accepted 8/8 monotone clicks → n8-mean d_seg 0.00054105→0.00052770; ledger + candidate banked to Volume). **Eval half FAILED**: eval-dir copy missed `submission_dir/encoder/` (src/fec10_hybrid_decoder.py re-exports the FECa decoder from ../encoder — single-source-of-truth pattern). Fix `69f83a865`. Ledgered.
- `fc-01KX6DKHTXQFW5SS85HJ6GNBX5` — n8 validation re-dispatch (detached, encoder/ fix, resumes the banked round-0 ledger from the Volume). IN FLIGHT.

Measured so far (in-container Linux x86_64, 8-core): the n8 round-0 clicks are worth ≈ −1.78e-7 full-set d_seg (≈21 seg-cells ≈ −1.8e-5 S). Naive one-round full-set extrapolation ≈ −0.0013 S (upper bound; PR128 banked −0.0019..−0.0027 over many rounds).

## n8 validation RESULT (2026-07-10, harvested) — ALL GREEN, chain proven end-to-end
- `fc-01KX6DKHTXQFW5SS85HJ6GNBX5` HARVESTED. Container: Linux-4.19.0-gvisor-x86_64, avx2+avx512f+fma, torch 2.5.1 CPU, constriction 0.4.2.
- In-container greens: round-trip byte-exact (sha `b46897…` reproduced) · GT build (evaluator-aligned AVVideoDataset decode) 5.1 s @n8 · Volume-ledger RESUME exercised for real (round-0 replayed, 2.8 s) · raw custody assert passed (3,662,409,600 B exactly) · full 600-sample `upstream/evaluate.py --device cpu` ran in **176.3 s** (vs the 60–120 min prior estimate — the whole cost model improves ~30×).
- **The exact row (600 samples, exact candidate bytes sha `ad02b012…`, 177,169 B):** seg 0.00055961 · pose 0.00002942 · rate 0.00471878 → recomputed from components **S = 0.1910828** — **beats the pointer 0.19109982 by −1.7e-5 with just 8 clicked pairs** (one ±1 round on n8). Axis: Modal Linux x86_64 CPU container (gVisor) — same axis-class as the pointer row; incumbent components reconstruct to ≈0.1911003 in-container (pointer-consistent to 5e-7).
- Custody: `experiments/results/clickpolish_pr110_20260710/n8_validation/` (candidate_archive.zip + report_cpu.txt + accepted_clicks_ledger.jsonl + harvest_result.json).
- **Pointer handling: NOT updated by this subagent (per coordinator) — main owns pointer refresh + verification.**

## n600 run (IN FLIGHT)
- `fc-01KX6DZWCHNPQ6KN59V2MZ845J` — n600, max_rounds=2, ±1 sweep, wall-clock cap 16,200 s, Volume run-id `n600_r1`, same-container eval at the end. Projected wall-clock ~2–5 h; projected cost $0.3–2.5 (worst-case rate) — under the $5 abort. Harvest: `.venv/bin/python tools/harvest_click_polish_run.py --call-id fc-01KX6DZWCHNPQ6KN59V2MZ845J`.

### n600 health (mid-run)
- Baseline at full n600 in-container: **S=0.19109945, d_seg 0.00055979, d_pose 0.00002942, 177,169 B** — reproduces the canonical pointer row (0.19109982) to **3.7e-7** on the selection surface; the chunked fused render+score survived its first full-scale exercise (no OOM). Round-0 ±1 sweep in progress.
