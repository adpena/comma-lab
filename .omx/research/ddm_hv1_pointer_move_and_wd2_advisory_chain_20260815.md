# hv1 pointer move #3 + the wd2 advisory-gate launch chain (MAIN, 2026-08-15)

## THE POINTER MOVED (third micro-edit-campaign move)
**hv1 ep0634: S = 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600].**
Call fc-01M036FY225QC9A75CM0Y7X7NP, Tesla T4, 421.6 s remote, ~$0.16, rc=0, zero
validation errors. Components: seg 0.029611 (decode-identical) + pose 0.0082945765
(decode-identical) + rate 0.1216917 (25·182,759/37,545,489). The realized row equals the
pre-dispatch projection to all 17 digits — the decode-identity proof held through the CUDA
axis. Δ = −4.947332e-4 vs the e480b v2 incumbent (0.1600920261571558).
Archive sha 80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e (receipt:
experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json).
Pointer updated (our_local_frontier_contest_cuda + effective_frontier); refresh tool re-run clean.
**Standing note:** this row is BELOW the public leaderboard #1 (PR #135, 0.162). Submission
remains OPERATOR-GO-gated per the pq1 packet HOLD.

## e960 lineage state (early-stop adjudication)
- The wc2 e960 continuation was ALREADY governed-early-stopped at 15:49Z
  (governed_early_stop_receipt.json: SIGTERM to safe_run 47774, rc=143, ep634 selected as
  argmin joint 130,393 B, Metal lane released, 81 periodic checkpoints retained).
- pid 63183 = detached_resume_r5, a SEPARATE 60-epoch continuation branch, ep52/60, FLAT at
  joint ~134,570 B — 4.1 KB WORSE than the banked ep634 argmin. It self-terminates at ep60
  (~2.3 h), coincident with the wd2 verdict boundary. Verdict: let it finish; do not kill.
  Its endpoint is supplementary only.

## The wd2 advisory n600 gate — LIVE (the sub-0.15 decisive measurement)
Launched detached (pid 12949, launch 42; liveness 12950 + quality 12953 watchers armed;
done receipt wd2_ep60_advisory_n600.done): experiments/contest_auth_eval.py on the wd2 ep60
EMA student archive (e9c4a9ed5e6bef89d228ca877a9f9e37345e3c79dc07ba20087c218ff89fcf87,
165,387 B) — --device cpu
[macOS-CPU advisory], full n600, work dir + inflated payload retained on APDataStore.
Verdict rule: realized Δd_seg ≤ ~1.07e-4 at held pose ⇒ wd2 admitted toward the composed
sub-0.15 T4 row (projection ≈ 0.1480 = hv1 base − 0.0115664 rate); else the distillation
needs more epochs or a successor config.

## Six launch attempts — every failure a real defect, each cured or routed
(the operator's "signal reveals opportunities" applied)
1. **AppleDouble AST-parse crash**: the harness parses every *.py under the submission dir;
   ExFAT `._*` sidecars are binary. CURE: cleaned (46 files); OPPORTUNITY: the harness should
   skip `._*` (routed to #1064-cluster hygiene).
2. **Upstream bytecode refusal**: `upstream/__pycache__` tripped the fail-closed authority
   hasher. CURE: removed (untracked build product).
3. **STRUCTURAL: the local advisory chain was broken by design-vs-reality drift** — the
   authority hasher refuses ALL symlinks/pyc under upstream/, but the PINNED tree itself
   contains two TRACKED symlinks (submissions/{roi_v2,av1_roi_lanczos_unsharp}/lib/
   libSvtAv1Enc.so.2 — #836's "2 lost symlinks") plus upstream/.venv (564 MB, python
   symlinks + thousands of pyc) created by `uv run`. CURE + NEW DURABLE ASSET:
   **/Volumes/APDataStore/pact/upstream_eval_mirror_20260815** (157 MB; rsync of upstream/
   excluding .venv/.git/submissions/__pycache__/._*; 0 symlinks, 0 pyc; snapshot sha
   d5bb36a2b5a9c3b1a32105c129437f6d7311e44e071839d0afdfaba0dd8a2004). Every future LOCAL
   advisory row on ANY candidate passes --upstream-dir at this mirror. Scorer authority files
   byte-identical to the pinned tree by rsync -a construction.
4. **Bare-python receiver hazard (#929 class, live on the shipping inflate.sh)**: the
   cp135-family inflate.sh calls `python` (fine in contest containers, absent on macOS).
   CURE + ASSET: exec-wrapper shim /Volumes/APDataStore/pact/ddm_wd2_width_distillation/
   python_shim_bin/python → repo .venv (Brotli exactly 1.2.0 + torch + numpy, no network
   install). LESSON: a SYMLINK shim fails — venv pyvenv.cfg resolves relative to the symlink
   location, silently dropping site-packages; the wrapper SCRIPT preserves venv resolution.
5. **Launcher SIGTERM propagation**: killing the launcher during --verify-alive-secs kills
   the "detached" child too (reproduced: run.log truncated mid-inflate, no survivor).
   Routed to #1064 cluster; operational rule: outer timeout > verify window.
6. Attempt 6 = LIVE.

## Opportunities routed (operator 08-15 "signal reveals opportunities...")
- **Local advisory chain UNBLOCKED** — a whole class of "advisory n600 when scorer-free"
  fire-orders (mz2's 2 sub-KB candidates: q3/q4 −823 B, FiLM-row −130..−2,051 B) can now
  actually run locally through the canonical chain + mirror. Queue them after the wd2 verdict.
- **Checkpoint-selector as candidate generator**: 81 retained periodic checkpoints; the
  distortion-aware selector found ep634. Sweep the full set for any checkpoint whose coder
  output beats 182,759 B at held distortion — each is another hv1-class row at ~$0.16.
- **wd2 composes ON the hv1 base** if admitted: student pool replaces the frozen 34,763 B
  section on the NEW 182,759 B frontier archive.
- **Candidate-bound runtime law** (attempt-1 lesson from the T4 fire, ~$0.01): this receiver
  family pins ARCHIVE_SHA256+ARCHIVE_BYTES in inflate.py; every candidate needs its own
  staged generation; tree sha changes per candidate by construction.

Vehicle frontier: **S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**
(Δ −4.947e-4 this turn) · Modal ≈ $6.3/$20.
