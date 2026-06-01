# Codex Findings: Public HNeRV Lineage And PR95 Runtime Proof

UTC: 2026-06-01T15:52:56Z
Agent: Codex
Axis: [macOS-MLX research-signal], false authority

## Verdict

PR95/HNeRV is not magic and not globally optimal. It is strong because it chose the right contest-rate carrier: a small learned decoder plus small per-pair latents, trained against the actual distortion stack and then packed into a tiny charged archive. Z8/HPRC-style explicit coefficient fields are useful as analysis/residual teachers, but they are too byte-expensive as the primary carrier unless a later codec collapses them below the learned-decoder grammar.

## What PR95 Gets Right

- Archive grammar: one charged `0.bin` inside `archive.zip`; no hidden sidecars.
- Representation: latent row per frame pair, tiny HNeRV decoder, no dense per-pixel or per-coefficient field.
- Runtime: `inflate.sh` deterministically decodes model+latents into contest raw RGB.
- Loss path: official scorer-shaped objective, including SegNet margin-style losses, PoseNet term, eval resize/rounding, QAT, entropy regularization, and Muon fine-tuning in the late stage.
- Codec path: decoder weights quantized to INT8 and Brotli-packed; latents quantized per dimension, delta-coded in time, lo/hi split, then Brotli-packed.

This is why PR95-scale archives can live near the 100k-200k byte band while Z8 stores MB-scale explicit wavelet fields.

## Public PR Lineage To Mine

Live GitHub PR metadata was refreshed into `/Volumes/VertigoDataTier/pact/public_pr_source_mining_codex_20260601/`.

High-value source lineage:

- PR95 `hnerv_muon`: baseline HNeRV + 8-stage curriculum + Muon final stage.
- PR98 `hnerv_muon_finetuned_from_pr95`: direct PR95 continuation.
- PR99/PR100 `hnerv_muon_lc` / `hnerv_lc_v2`: low-codec/LC grammar improvements.
- PR101 `hnerv_ft_microcodec`: microcodec lane, useful for section coding and runtime grammar.
- PR102 `hnerv_lc_v2_scale095_rplus1`: resolution/scale variant around the same byte grammar.
- PR103 `hnerv_lc_ac`: arithmetic-coded LC variant; important for codec comparisons.
- PR104 `qhnerv_ft_best`: broad training/codec source tree with staged losses and QAT.
- PR105/PR106 `kitchen_sink` / `belt_and_suspenders`: full-stack variants worth deconstructing for staged design choices, not blindly copying.
- PR110 `hnerv_fec6_fixed_huffman_k16`: our later sidecar/selector extension; useful only where residual bytes price in.

## Implementation Landed

`tools/run_compact_renderer_mlx_spine_runner.py` now has a PR95 receiver-proof path:

- extracts candidate `archive.zip` member `0.bin`;
- runs the public PR95 `inflate.sh` through `bash` with repo `.venv/bin` prepended to `PATH`;
- hashes the produced raw RGB stream;
- records archive/member/runtime file hashes, argv/cwd/env override, stdout/stderr tails, byte counts, and SHA-256;
- deletes rebuildable raw output by default after certification.

Smoke artifact:

- Report: `/Volumes/VertigoDataTier/pact/compact_pr95_hnerv_receiverproof_4pair_1ep_codex_v4/compact_renderer_mlx_spine_runner_report.json`
- Receiver proof: `/Volumes/VertigoDataTier/pact/compact_pr95_hnerv_receiverproof_4pair_1ep_codex_v4/receiver_proof/pr95_hnerv_receiver_proof.json`
- Archive SHA-256: `66f1b9ff7b64ec6b41913d28f273b95bfde28806272c982e38b397d755d25548`
- Raw output bytes: `24416064`
- Raw output SHA-256: `f16abec628828c3c121347d67004209c6f39ff18517b39520214a0a4eaa68978`
- Receiver proof: valid for 4-pair runtime consumption
- Cleanup: raw output certified rebuildable and deleted

## Remaining Blockers

- 4-pair receiver proof is not score authority.
- Full 600-pair receiver proof has not been run for this runner slice.
- Full-video MLX scorer replay is not attached to the runner output yet.
- Exact CPU/CUDA authority has not been dispatched.
- The PR95 path is currently an executable MLX archive-export control arm, not a source-faithful PR95 reproduction: Stage-8 Muon continuation, source-matched schedules, QAT/C1a/resume semantics, and full scorer-network loss still need faithful porting.

## Next Engineering Move

Port the PR95-PR106 staged curriculum and codec lineage as explicit interchangeable runner stages under the same packet spine: PR95 Stage-8 Muon first, then PR100/PR103 low-codec and arithmetic-code variants, then PR104/PR105/PR106 staged-loss deltas. Every stage must emit trained weights, trained latents, archive bytes, receiver proof, full-video scorer replay, and exact blocker/dispatch packet. If a stage cannot beat the byte-value price under the same archive grammar, demote it and pivot to RNeRV/SR-NeRV/PVQ carriers under the same spine.
