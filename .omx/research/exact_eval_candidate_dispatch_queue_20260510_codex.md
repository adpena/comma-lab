# Exact-eval candidate dispatch queue - 2026-05-10

generated_at_utc: 2026-05-10T15:35:28Z
research_only: true
score_claim: false
dispatch_attempted: false
lane_claim_created: false
remote_gpu_run: false

## Scope

Local ranking only. The operator explicitly requested no remote jobs and no
lane claims. This memo ranks currently available byte-closed exact-eval
candidates under the current provider/claim blockers and records the next
command to run only after the blocker is cleared.

Current live blocker from `tools/claim_lane_dispatch.py summary --format json`:

- Active claim: `t1_balle_128k_endtoend`,
  job `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`, platform `modal`,
  status `active_dispatching`.
- Lightning exact-eval packets remain blocked in this shell by missing
  `LIGHTNING_SSH_TARGET`, `LIGHTNING_REMOTE_PACT`, `LIGHTNING_UPSTREAM_DIR`,
  `LIGHTNING_TEAMSPACE`, `LIGHTNING_STUDIO`, and `LIGHTNING_SDK_USER`.
- Per operator scope, no claim was opened for any row below.

## Ranking

| EV rank | Candidate | Candidate archive | SHA-256 / bytes | Expected score mechanism | Current blocker | Exact next command |
|---:|---|---|---|---|---|---|
| 1 | q10 151-byte Brotli | `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/release_surface/archive.zip` | `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7` / `186088` | Rate-only `-151` bytes versus PR106 source; packet records expected total score delta `-0.00010054470192144788`. Decoded payload is intended unchanged, so EV is high despite small absolute delta. | Missing Lightning env; missing active lane claim; do not open claim until ready to submit. | `.venv/bin/python -c 'import os,sys; missing=[k for k in sys.argv[1:] if not os.environ.get(k)]; raise SystemExit(("FATAL: missing Lightning env: "+", ".join(missing)) if missing else 0)' LIGHTNING_SSH_TARGET LIGHTNING_REMOTE_PACT LIGHTNING_UPSTREAM_DIR LIGHTNING_TEAMSPACE LIGHTNING_STUDIO LIGHTNING_SDK_USER` |
| 2 | PR103-ac hidden gem | `experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/archive.zip` | `8274e88c0ab1d26a06470a0730d17fe004556afa564460cf1c05624ff6060278` / `185574` | Deletes one runtime-consumed u32 from PR103-on-PR106 `merged_ac` and updates packed length; rate delta is only `-4` bytes, but the candidate changes decoded state tensors, so it has higher information value and possible distortion movement. | Score unknown; decoded state changes require exact CUDA before any rank/score claim; missing lane claim; Modal currently occupied by active T1. | `.venv/bin/python scripts/lightning_exact_eval_repro.py --job-name exact_eval_pr103_ac_hidden_gem_20260510 --stage-workspace --submit --archive experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/archive.zip --baseline-json experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json --inflate-sh experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/runtime/inflate.sh --upstream-dir $LIGHTNING_UPSTREAM_DIR --remote $LIGHTNING_SSH_TARGET --remote-pact $LIGHTNING_REMOTE_PACT --studio $LIGHTNING_STUDIO --teamspace $LIGHTNING_TEAMSPACE --sdk-user $LIGHTNING_SDK_USER --machine ${LIGHTNING_MACHINE:-T4} --predicted-band 0.18 0.25 --regression-threshold 0.02 --component-trace --queue-metadata lane=pr103_ac_hidden_gem --extra-artifact experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/manifest.json --extra-artifact experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/runtime_decode_proof.json` |
| 3 | WR01 PR106x | `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/archive.zip` | `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628` / `186222` | Static packet changes `latents_and_sidecar_brotli`, `-9` bytes versus PR106x source. Packet predicts rate-only score delta `-0.0000059927305780995425` with zero expected seg/pose movement, but the readiness audit explicitly defers it behind q10. | Missing Lightning env; missing active lane claim; packet blocker `adversarial_priority_review_prioritizes_rate_only_candidate`. | `.venv/bin/python tools/lightning_dispatch_pr106_stack.py --lane wr01_apply_pr106x_half --archive experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/archive.zip --inflate-sh experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/inflate.sh --predicted-low 0.18 --predicted-high 0.25 --job-name exact_eval_wr01_apply_pr106x_half_20260510_print_only --print-only` |
| 4 | A1 canonical anchor | `experiments/results/A1_canonical/harvested_artifacts/finetuned_archive/archive.zip` | `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` / `178262` | Already has paired evidence: `[contest-CPU GHA] 0.19284757743677347` and `[contest-CUDA T4] 0.2263520234784395`. EV is as baseline/custody comparator, not a new score-lowering dispatch. | No new candidate delta. Re-dispatch only to diagnose infrastructure drift, not to spend score-lowering budget. | `shasum -a 256 experiments/results/A1_canonical/harvested_artifacts/finetuned_archive/archive.zip && jq '{archive_sha256, archive_size_bytes, cpu, cuda, n_samples}' experiments/results/a1_dual_cuda_dispatch_20260509T163400Z/dual_eval_adjudicated.json` |
| 5 | PR106 sidechannels group | `experiments/results/lane_pr106_latent_sidecar_cpu_smoke_20260505/sidecar_archive.zip`; `experiments/results/lane_pr106_yshift_cpu_smoke_20260505T140325Z/pr106_yshift_sidechannel_archive.zip`; `experiments/results/lane_pr106_lrl1_cpu_smoke_20260505T140325Z/pr106_lrl1_sidechannel_archive.zip`; `experiments/results/lane_pr106_stacked_3sister_cpu_smoke_20260505T140325Z/pr106_stacked_archive.zip` | latent `5560af2a6a47db14c0f6ad04eaf832d32f88df103d1eb238a904bb166eb87242` / `186262`; yshift `0930148490ea7897ec80f7658dd0e8c227cd92575cdfd730a3a4312ad2965678` / `186283`; LRL1 `983e34a01c60604eb87b1f971ed989527f4aa5d60c9f8f5ee494c041e6f50e6f` / `186289`; stacked `28a3c751f2280e702c8ef58c8430407bffa79a1e7914ec879a15c6314d710bd6` / `186348` | Possible component-sidechannel movement, but current readiness says local planning only. The shortest PR106-family exact packet is WR01, not the three-sister scaffold. | `stack_readiness.json` records 17 blockers, including missing exact-CUDA artifacts for latent/yshift/LRL1/WR01, stack interaction review, and exact eval. | `.venv/bin/python tools/build_pr106_sidechannel_stack_readiness.py --fail-if-dispatch-ready --json-out /tmp/pr106_stack_readiness_verify.json` |

## Dispatch order

1. Do not dispatch while `t1_balle_128k_endtoend` is active on Modal. First
   recover/close that claim or use a non-conflicting Lightning path only after
   Lightning identity/env is present.
2. Once Lightning env is present and the operator permits a claim, run q10
   first. It is the cleanest byte-closed rate-only exact-eval packet and has
   the largest deterministic rate delta among this set.
3. If q10 is terminal, run PR103-ac hidden gem next only as a distortion-risk
   measurement. It is not rate-only; decoded weights change.
4. Keep WR01 as a follow-up static packet after q10 because it is dominated on
   rate-only EV.
5. Keep PR106 sidechannels in local readiness mode until sister exact-CUDA
   artifacts or a reviewed stack interaction packet exists.

## Solver-stack wire-in disposition

This is a queue memo, not a new candidate implementation or empirical anchor.
`research_only=true`.

- sensitivity-map contribution: N/A, no new score/component anchor measured.
- Pareto constraint: non-binding; rank is dispatch priority only.
- bit-allocator hook: N/A, no allocator policy changed.
- Cathedral autopilot dispatch hook: N/A; no remote job or claim opened.
- continual-learning posterior update: N/A, no empirical anchor.
- probe-disambiguator: N/A, no new ambiguous implementation choice introduced.
