The exact frontier did not move. CM1 found no trainable coder-matched surrogate, so the three-term objective is **NOT FIREABLE** and its canonical row is blocked.

Measured `[macOS-CPU advisory / scorer-free exact-byte targets]`:

- Cheapest correlation-qualified candidate: F26/HPAC prefix-128, Pearson/Spearman `0.962088/0.942857`, `n=6`, but `167.911 s/eval`.
- Exact edit-support halo-0: `0.999999/1.0`, but requires the full-state `897.675 s` median encode.
- SM2 bank + 2 fresh FS2 rows transferred to held-out dx2 at only `0.207665/0.028571`.
- All other static proxies were negatively correlated.

Artifacts:

- [Verdict memo](/Users/adpena/Projects/pact/.omx/research/ddm_cm1_coder_matched_surrogate_20260826.md)
- [Race harness](/Users/adpena/Projects/pact/experiments/ddm_cm1_coder_matched_surrogate.py)
- [Retained results](/Volumes/APDataStore/pact/ddm_cm1_coder_matched_surrogate/RESULTS.json), sha256 `833e4100e59aa8f7206d21c4be3a0a4c0fa704611cc9f3ebc15919cb468514df`
- Eight full candidate fields, four real prefix payloads, ledgers, hashes, and manifests remain under `/Volumes/APDataStore/pact/ddm_cm1_coder_matched_surrogate/`.

Landed commits: `6e646fa07f` harness, `690e07abab` memo, `1e42871e1b` ledger. The ledger serializer returned rc7 after committing because the declared post-hash used escaped Unicode while the patch normalized it; inspection confirmed the intended six rows landed. Unrelated MAIN/lm1 work remains preserved in the shared worktree.

Verification passed: Ruff, `py_compile`, two genuine review-tracker passes, SM2 soft-histogram parity, and independent SciPy recomputation of all correlations and retained hashes. No scorer, Modal, archive mutation, or `upstream/` write occurred. The `n=6/n=8` bank is a bounded formulation result, not a population-level family kill.

Own-vehicle frontier: **S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**, unchanged.

## NEXT_IF_RESUMED

- **QUEUED** — owner: `MAIN`; consumer store: `.omx/state/canonical_task_status.jsonl::ddm_no1_row1_three_term_objective`; fire trigger: a restartable F26/HPAC state cache or exact-incremental outer loop validates Pearson and Spearman `>=0.9` at trainable cost on stratified-random `n>=32`.
- **QUEUED** — owner: `MAIN`; consumer store: the same canonical row; fire trigger: Metal availability plus MAIN’s single-flight claim for the outstanding wd3 seed-variation prerequisite.

## LIVE-HYPOTHESES

- Restartable coder-state checkpoints could preserve halo-0’s near-perfect ordering cheaply; it already has only `0.806 B` all-row MAE once true coder state is available.
- The sharp improvement from prefix-64 to prefix-128 may expose an adaptive-context horizon near 128 frames.
- A differentiable model of match lengths and adaptive-context state may work where static counts failed, because those are the mechanisms the live coder actually prices.

## DEAD-ENDS

- Marginal entropy and hard entropy plus temporal entropy: strongly anti-correlated on this formulation.
- The existing 152-row SM2 affine model: does not transfer to current F26/HPAC perturbations, even after adding the two FS2 rows.
- Static direction, position-cost, and mismatch features: negative held-out correlation.
- Prefix-32 and prefix-64: fail one or both correlation gates.
- Full-state edit-support replay: accurate but not a trainable surrogate at `897.675 s/eval`.