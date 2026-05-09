# T1 / HNeRV parity adversarial hardening (2026-05-09 Codex)

<!-- generated_at: 2026-05-09T07:20:00Z, author: codex -->
<!-- evidence_grade: source_hardening + subagent_reported_signal; no score claim -->

## Scope

This ledger records the local Codex hardening pass triggered by the HNeRV
leaderboard-retrospective and the T1 scaffold review. It is not a score claim.

## Findings hardened

1. **T1 non-smoke synthetic target leakage**
   - Pre-fix behavior: `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py`
     loaded frozen A1 latents, then always called `make_smoke_target()` for
     targets. A full GPU run would optimize against random pixels.
   - Fix: non-smoke training now requires real targets from
     `upstream/videos/0.mkv` via PyAV or a caller-supplied
     `--target-pixels-path`. `--allow-missing-canonical-a1` is smoke-only.
   - Guard: catalog #114 now treats `make_smoke_*` factories the same as
     `make_synthetic_*` when scanning non-smoke training paths.

2. **T1 scaffold exact-eval false-promotion risk**
   - Pre-fix behavior: the trainer exposed `--auth-eval` even though the
     generated inflate runtime is a pickle/state-dict scaffold with a
     non-contest signature.
   - Fix: `--auth-eval` is refused before training with
     `research_only_no_export`. The remote T1 script is scaffold-smoke only
     and no longer prints a `[contest-CUDA]` completion tag.

3. **T1 phantom dispatch-claim risk**
   - Pre-fix behavior: `tools/dispatch_t1_balle_endtoend.py` claimed the lane
     before creating a real provider job, while provider functions only wrote
     metadata.
   - Fix: non-dry-run dispatch is refused before any claim is written. Dry-run
     remains available for operator planning and uses provider-specific
     claim-preview metadata only.

4. **Score-gradient pose-axis derivative bug**
   - Pre-fix behavior: `tac.score_gradient_param_saliency` documented
     `d sqrt(10*d_pose) / d d_pose ≈ 0.9` at `d_pose ≈ 3e-5`; the correct
     value is about `288.7`.
   - Fix: added `pose_distortion_score_derivative()` and made the default
     pose saliency weight equal the local contest-score derivative at the
     medal-band CPU floor.

5. **Stale FastViT attention narrative**
   - Fix: the HNeRV operator-clarification memo now supersedes the
     FastViT-attention-compounding phrasing with the active loader-byte /
     conv-kernel / Hydra-head hypotheses.

## HNeRV byte-identity signal to preserve

Subagent report from a0be36e / binary-forensics stream:

- T9 self-classified as `DEFERRED` with `48 substrate_tied + 0 compatible`
  empirical matrix.
- A1's decoder / latent / sidecar bytes are reported SHA-256 byte-identical
  to PR101's corresponding sections.
- The reported A1-vs-PR101 score gap (`0.19284` vs `0.193`) is therefore
  attributed to `inflate.py` framing / runtime differences, not weight bytes.

Status: **high-priority forensic signal, with PR100=PR102 byte identity now
stronger than PR101-vs-PR103 causal attribution**. This should redirect score
lowering toward same-archive runtime-constant sweeps, runtime framing controls,
channel-bias correction, and contest-contract-identical inflate reproduction
before more cross-archive composition work.

## Reactivation / next actions

- Run the standalone T1 smoke and full `tools/all_lanes_preflight.py` after
  this patch.
- Add the `rgb_to_yuv6` differentiability fix from the HNeRV dossier as a
  first-class training guard before any new NeRV/HNeRV/Lane12-v2 dispatch.
- Independently verify the A1-vs-PR101 byte-identity claim with section SHA
  manifests and a no-op/runtime-framing control.
