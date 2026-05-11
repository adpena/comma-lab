# Handoff visionary round 2 score-lowering hardening (2026-05-11)

## Purpose

This ledger records the second handoff-polish pass requested by the operator
after `58864c0e` landed on `main`. It mirrors the local Downloads handoff
refresh so durable repository state captures the same decisions.

Score claim: `false`.
Dispatch attempted: `false`.

## Current source of truth

- Branch: `main`
- Remote: `origin/main`
- Current pushed head at refresh time:
  `58864c0e research: harden packet compiler and score-floor handoff`
- Worktree before this ledger: clean and aligned with `origin/main`
- Handoff path:
  `/Users/adpena/Downloads/pact_score_lowering_handoff_2026-05-11.md`
- Handoff SHA-256 after this refresh:
  `f8579f8e454a5d8bacbad2d6e7699f3ef9c25011afff65c3a72959bef9ad380e`
- Handoff line count after this refresh:
  `1253`

## Validation cited by the handoff refresh

- Packet compiler focused suite:
  `93 passed in 1.08s`
- All-lanes preflight:
  `ALL 29 PREFLIGHT CHECKS PASSED`
- Preflight wall time:
  `2.28s`
- Preflight workers:
  `8`
- Estimated speedup:
  `4.78x`
- Timeout contract:
  `30s`
- Kaggle yshift live status at refresh:
  `KernelWorkerStatus.RUNNING`

## R2 promotion-state correction

The earlier handoff text still described PR106 R2 compliance as work to do.
That is now stale. The current state is:

- Release surface:
  `submissions/pr106_latent_sidecar_r2/`
- Strict compliance JSON:
  `submissions/pr106_latent_sidecar_r2/pre_submission_compliance.contest_final.json`
- Compliance result:
  `passed=true`, `strict=true`, `contest_final=true`, `total_checks=54`,
  `failed_errors=0`, `warnings=0`
- Compact custody packet:
  `submissions/pr106_latent_sidecar_r2/pre_submission_compliance.custody_packet.json`
- Remaining promotion blockers:
  - `cpu_leaderboard_reproduction_not_adjudicated`
  - `five_turn_council_greenup_not_executed`
  - `operator_pr_submission_approval_not_recorded`

The next custody action is therefore the paired Linux x86_64 `[contest-CPU]`
eval on the exact R2 archive SHA
`7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`, not
another compliance rerun unless the release surface changes.

## Central operating thesis

The score-lowering system is a deterministic packet compiler wrapped around a
scorer-aware representation learner. Every useful idea must become one of:

1. a lower-score exact packet;
2. a compiler transform with golden vectors;
3. a device-axis mechanism measurement;
4. a training-loop correction that emits contest bytes;
5. a scoped lower-bound certificate;
6. a preserved negative with reactivation criteria.

Everything else is discussion, not progress.

## Updated immediate execution order

1. Promote PR106 R2 from exact score result to promotion-grade custody packet:
   paired `[contest-CPU]`, five-turn council greenup, operator submission
   decision, and public-release hygiene.
2. Harvest yshift; materialize only if bytes are charged and consumed; exact
   T4 only after no-op proof.
3. Use `tac.packet_compiler` for PR101/PR103 grammar work, not new one-off
   scripts. First target: identity parse/re-emit plus sidecar compression
   vectors over PR106 R2.
4. Start the first `adpena/molt` / native PacketIR proof only against committed
   golden vectors. It is a byte-transducer backend, not an eval oracle.
5. Convert the wavelet residual scaffold into a measured coefficient table over
   PR106 decoded outputs, still `score_claim=false` until it emits consumed
   archive bytes.
6. Recover or terminal-classify the active T1 Modal call before any new T1
   dispatch.
7. Regenerate scorecard/roadmap surfaces so stale `0.20739` or PR103-on-PR106
   language cannot route spend ahead of the `0.20664588545741508` PR106 R2
   exact T4 anchor.

## PacketIR build target

The next reusable abstraction should represent:

```text
archive.zip
  -> zip_member("0.bin")
  -> section(decoder weights)
  -> section(latents)
  -> section(sidecar/control stream)
  -> section(optional residual program)
  -> runtime_consumption_proof
  -> exact_eval_axis(cpu|cuda)
```

Required passes:

- `identity`: byte-identical re-emission.
- `analyze`: offsets, entropy, shapes, histograms, sidecar grammar.
- `optimize`: charged-byte-changing transforms only.
- `materialize`: archive/runtime/manifest emission.
- `verify`: no-op proof and exact-eval readiness.

This is the durable antidote to hidden bytes, no-op byte churn, and
research-only representations that never enter the scored packet.

## Native backend rule

`adpena/molt`, Rust, Zig, and C are appropriate for stable byte contracts:
parsers, entropy coders, fast preflight kernels, and tiny deterministic runtime
helpers. They are not appropriate as speculative score or training oracles.

First native proof target:

1. Decode and re-encode `centered_delta_uint8_v1.json`.
2. Decode and re-encode `latent_hi_arithmetic_v1.json`.
3. Emit source SHA, toolchain version/hash, binary SHA, binary bytes, startup
   time, and golden-vector pass/fail.
4. Add malformed-negative vectors before any promotion.

Native speed is useful for search and DX. Native binary size becomes
score-relevant only if it reduces charged runtime bytes or enables a measured
positive packet that exact eval validates.

## Score-lowering economics at PR106 R2

At the R2 anchor:

- 1 charged byte costs `6.6586e-7` score.
- 1000 charged bytes cost `6.6586e-4` score.
- pose is locally about `2.78x` more marginal-sensitive than seg per unit
  distortion.

This ranks near-term byte spend:

1. pose-relevant latent/residual atoms;
2. yshift only if harvested table proves transfer;
3. sidecar compression that preserves R2 correction;
4. wavelet/foveation residual atoms with measured `d_pose/db` and `d_seg/db`;
5. full renderer training only when it exports archive bytes in loop.

Do not spend bytes on prettier RGB unless the scorer moves.

## Non-HNeRV escape discipline

Non-HNeRV should enter as residual/control programs over the current champion
before full substrate replacement:

- PR97 range-coded masks as a mask/residual sidecar.
- HPAC/token maps as categorical residual programs.
- flatpup/lowpass-luma as low-frequency RGB correction atoms.
- wavelets/foveation as sparse multi-resolution residuals.
- Ballé/CompressAI as trained transforms only after archive-in-loop closure.

Replacement substrates compete after they can produce exact packets under the
same custody standard as PR106 R2.

## Proof target

The first serious floor result should be scoped:

```text
Given PR106-style PacketIR grammar G and device axis A:
  prove lower bound L_A(G, atom_budget, runtime_contract)
  construct packet P in G
  exact-eval S_A(P)
  report epsilon_A = S_A(P) - L_A
```

The shortest route to this is identity re-emission, entropy bounds,
branch-and-bound on sidecar atoms, and exact CPU/CUDA paired packets.
