# Can we train PR130 end-to-end locally? — MEASURED, all four legs run on Metal

Operator question 2026-08-09: *"Are you missing anything else necessary to train end to end
here locally? and then iterate and optimize from there."* Answer: **structurally, no.** Every
leg of `scripts/train.sh` executes on this machine. Audited by RUNNING it, not by reading it.

Authority `[macOS-Metal advisory]`, `score_claim=false`. Intake READ-ONLY; all outputs under
`/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/` (2.0 GB).

## The chain, leg by leg — every row MEASURED today

| leg | status | evidence |
|---|---|---|
| `prepare` (GT caches) | ✅ | both caches restored + SHA-verified: `gt_cache_600.pt` 117,981,133 B / `8248a60d…`, `gt_cache_600_official_ada.pt` 117,981,301 B / `382d7dfe…`. NOTE the pair maps to the AV/DALI axis split. |
| `hpac-init` | ✅ | `extract_integer_hpac_archive.py` on the base archive → 160.8 KB init, 28 state tensors / 38,858 values |
| `semantic` | ✅ TRAINS | 20-step + 100-step runs, `--device mps`. `verdict: PASS`, `quantized_exact_seg` 0.00029703776041666664, **`packed_parameter_bytes: 40252`** = the shipped semantic section exactly. **~0.48 s/step** (100 steps + load + 1 n600 eval = 77 s). |
| `carrier` (pose) | ✅ TRAINS | via pq1's Metal wrapper through `tools/safe_run.py`. 6 steps, **`scope_pairs: 600`**, `full_quantized` mean d_pose **2.4437744286842644e-05**, quantized_basis_coeff mean 2.3431171939591877e-05, `projected_600_payload_bytes: 31632`. peak_rss **3,397 MiB**, 70.72 s incl. master-cache build + 2 evals. |
| `hpac` | ✅ TRAINS | 1 epoch `--device mps`: **59.18 s/epoch**, bpp 0.007995914003821745, top1_error 0.0019705708821614585, `estimated_token_bytes 117905` (shipped 116,980 = 0.8% apart) |
| `encode-tokens` | ⬜ NOT RUN | inference not training; same thin device surface as the others |

The carrier report says `verdict: FAIL` — that is the run's own 4,000-step convergence target
evaluated at step 6. It is NOT a port failure. Do not cite it as one.

## Why the port surface was thin (measured, not assumed)

Per-file CUDA references: `--device default="cuda"` (a default, overridable) plus
`torch.backends.cuda.matmul.allow_tf32 = False` (a flag write, harmless off-CUDA). TRUE
autocast/GradScaler count, word-boundary matched: semantic **2**, carrier **0**, hpac **0**,
codec **0**, pack **0**. My first grep said carrier had 20 — that was `amp` matching
**"amplitude"** (`--amplitude 64.0`). pp2's "ZERO AMP" was right; my grep was wrong.

`--challenge-root` only needs a tree it can `sys.path.insert` and `import modules` from —
our `upstream/` satisfies it.

## The governor caught a raw launch — correctly

First carrier attempt was REFUSED: *"bypassed the P0 memory admission gate."* That is #254
working (the 08-06 OOM lesson). Re-run through `tools/safe_run.py --projected-gib 12`:
`status=ok exit=0 peak_rss=3397MiB`. **Heavy legs go through safe_run, always.**

## Wall-clock for a full `train.sh all` (measured where stated)

- semantic 6,000 steps ≈ **48 min** compute + ~24 periodic n600 evals ≈ **~1 h**
- hpac 60 epochs × 59.18 s = **~59 min** (MEASURED per-epoch)
- carrier 4,000 steps: **NOT separately timed** — the 70.72 s smoke bundles master-cache build
  + 2 n600 evals with 6 steps, so a per-step rate cannot be honestly extracted from it. Owed.
- encode-tokens 600 maps: not timed.

Order-of-magnitude: **a few hours per full chain**, not days. Fits a local overnight.

## What is genuinely still missing

1. **carrier per-step timing** — one clean timed run with evals off. Cheap, owed.
2. **`encode-tokens` on Metal** — untested; the `verify_file` bar is byte-exact against the
   shipped 116,980 B tokens, so it doubles as a strong end-to-end check.
3. **A CUDA/DALI axis** for any *score* claim. Training is local; scoring is not.
4. **Retrain-from-scratch** is a different question than this chain: `train.sh` replays the
   SELECTED TAILS (6k semantic / 4k carrier / 60ep HPAC) from banked init checkpoints. The
   full 49-stage graph is `scripts/e2e.py`. Tails are what their own reproduction contract
   documents, and what we can iterate on today.
