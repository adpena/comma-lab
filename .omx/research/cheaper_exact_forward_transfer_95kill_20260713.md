# Task #456 — cheaper exact frozen-SegNet forward: terminal static-process transfer

**Outcome (FINAL): scoped NO-GO for rankable exact transfer.** A process-static one-thread eager
NCHW frozen-SegNet forward is **MEASURED 2.9562855478032297x faster on Torch 2.12.1** and
**MEASURED 2.9970426994326185x faster on Torch 2.12.0** at full n600. The timing gate passed on
600/600 matched pairs on each build. The argmax-bit-identical gate did not: both builds have the
same 15 cross-arm mismatching pair digests, including pair78. Each arm is internally deterministic
across two measurement children and two independent replay children, so this is an arm-stable
static-thread difference, not an alternating-order transient in these receipts.

`verdict_scope: fresh-child process-static ABBA formulation over first 600 receiver-realized pairs
on the fingerprinted local macOS CPU/Torch build only; n<600 diagnostic; no transfer to another
host/build/model/input set, backward, full training, contest-CPU/CUDA, evaluator, d_seg, d_pose,
archive, score, or promotion`

This negative closes only the tested eager-NCHW, static inter-op=1, six-thread-versus-one-thread
formulation on the two fingerprinted local builds and receiver corpus. It is not a family kill for
other exact-forward implementations, bounded reuse, or surrogate replacement. It does not authorize
the one-thread control in the trainer.

`research_only=true`; `score_claim=false`; `promotion_eligible=false`; canonical pointer **UNMOVED**.
All rows are `[macOS-CPU advisory; process-static torch-fp32 training-forward; no MPS/CUDA]`.
Contest-CPU timing is unavailable. NumPy-fp32 remains the bit-identical reference authority; MPS is
never a score.

## Evidence labels and terminal matrix

- **MEASURED:** timings, ordered argmax-sequence SHA-256 values, per-pair SHA-256 values, child
  identities/PIDs, thread bindings, receipt bytes, runtime/build custody, and pair78 arm digests.
- **DERIVED:** speed ratios from receipt medians; the 15-pair mismatch set from the eight terminal
  per-pair SHA vectors; zero flips only when all eight vectors are equal. Here they are not equal,
  so pixel flip cardinality is unavailable and is deliberately not guessed.
- **INFERRED:** the prior alternating-switch formulation was not the sole cause, because each static
  arm now reproduces its own output across measurements, replays, and both builds. This does not
  identify the lower-level kernel/reduction implementation responsible.
- **ASSUMED:** the three-canary tournament size and 25-pair recovery interval are screening/recovery
  choices only; neither carries verdict authority.

| local build | baseline median | selected median | MEASURED speedup | timing sign gate | exactness | terminal verdict |
|---|---:|---:|---:|---:|---|---|
| Torch 2.12.1 | 893.005052 ms/pair | 302.06995825 ms/pair | 2.9562855478032297x | 600 wins, 0 losses, p=2.409919865102884e-181 | 15/600 pair-SHA mismatches; flip pixels UNKNOWN | NO-GO |
| Torch 2.12.0 | 897.6003335 ms/pair | 299.49534375 ms/pair | 2.9970426994326185x | 600 wins, 0 losses, p=2.409919865102884e-181 | 15/600 pair-SHA mismatches; flip pixels UNKNOWN | NO-GO |

Each build used eight distinct process-static children: ABBA measurements
`baseline_rep0 -> selected_rep0 -> selected_rep1 -> baseline_rep1`, plus one independent full n600
replay child for every measurement. Both receipts report 8 unique child IDs, 8 unique PIDs, one
process segment per measurement pass, and two complete replays per arm.

## Pair78 confound: characterized, not cleared

The old alternating-thread measurement produced a terminal replay drift at pair78, whose prior
top-two margin was **MEASURED 2.384185791015625e-7** at `(y=275, x=356)`, classes 0/1. Earlier
fresh-process spot diagnostics reproduced an expected SHA, so the old verdict was correctly scoped
to the alternating-switch formulation and the 3.3-3.5x timing rows were not rankable.

The terminal static-process experiment does not clear pair78. It makes the failure reproducible:

- six-thread pair78 SHA on both builds, both measurements, and both replays:
  `eca965a83f375bafd10701b0f00416051cba54d4ba76188c6ab9799bd3c86b15`;
- one-thread pair78 SHA on both builds, both measurements, and both replays:
  `d85d66aba63e3b4094bda0b85594a2842991aeefb19627e98131af48048551ee`.

Therefore `pair78.stable=false` and `pair78.resolved=false` under the producer's eight-way admission
definition. The exact cross-arm mismatching pair indices are **DERIVED** identically on both builds:
`[35, 78, 88, 120, 131, 132, 140, 196, 214, 242, 327, 395, 468, 495, 578]`.

The full ordered sequence hashes also split cleanly by arm on both builds:

- six-thread baseline: `b5b76b74724b9428e575e27a2540f82523ded8eb8b2332d294cb61b9226f839c`;
- one-thread selected: `9f315cea1fccfddabee84d0d492f3618fc2a922648b7de95959200aa2b60fb9e`.

## Historical diagnostic rows preserved without promotion

The predecessor's in-process alternating-thread runs remain useful timing diagnostics but are not
terminal authority:

- Torch 2.12.1: **MEASURED diagnostic** 954.545 ms vs 291.594 ms = 3.27354x;
- Torch 2.12.0: **MEASURED diagnostic** 1078.183 ms vs 307.345 ms = 3.50805x.

Their matched measurement stage had zero observed argmax flips across 600/600, but the independent
terminal replay raised pair78 drift. The present experiment supersedes only the rankability
interpretation: its static-process timing is 2.95629-2.99704x, but exactness still fails. Historical
receipt bytes are not erased or relabeled.

## Terminal custody

Common custody:

- receiver raw SHA-256: `3819479cf6afc44b0366b01a1f1babfd25cd8fcc180825a24097e10b10d98975`
  (`3662409600` bytes, read in place);
- frozen SegNet weights SHA-256:
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`;
- static producer SHA-256:
  `694c21a78050cd1a3ebe23790a4c816f55fb20e33ce8711cdfb1f1f23c085b20`;
- dependency fingerprint SHA-256:
  `2597500491e48059b1c4350973c23ebe4c934b13b86569aa05b124b98c97da2d`;
- seed `0`; `PYTHONHASHSEED=0`; eager NCHW autograd; input-gradient graph preserved;
  intra-op baseline/selected `6/1`; inter-op `1`; no MPS/CUDA/contest-CPU timing.

Build-specific durable evidence:

- Torch 2.12.1 receipt:
  `experiments/results/segnet_exact_forward_static_transfer_torch2121_n600_20260713/receipt.json`,
  SHA-256 `134ad9bb23fcfc91ce6a8903002dc2c96dc392a057c73f15c71f30b4024341ad`;
  failure marker
  `experiments/results/segnet_exact_forward_static_transfer_torch2121_n600_20260713/receipt_static_checkpoints/failure_terminal_n600_no_go_000035_1783924238102993000_42007eb7a233.json`,
  SHA-256 `64e3849aa8e17bdcc3334c04b5e8b4ff07cf4865f3188c82b086f389e1cd7821`.
- Torch 2.12.0 receipt:
  `experiments/results/segnet_exact_forward_static_transfer_torch2120_n600_20260713/receipt.json`,
  SHA-256 `7c6a6c3c57511de599d2d2279e9ded11472c69cf3b87742507fdc07e4560d0eb`;
  failure marker
  `experiments/results/segnet_exact_forward_static_transfer_torch2120_n600_20260713/receipt_static_checkpoints/failure_terminal_n600_no_go_000035_1783927331385659000_4e97dca54bff.json`,
  SHA-256 `888e70b9b2473e418d14ed5790e3759023b8586719168386156d0443a4d0ac01`.

The probe writes atomic small JSON checkpoints every 25 pairs, preserves every completed
measurement/replay stage, and resumes only under an identical run fingerprint. It persists no
argmax tensor bulk and creates no rebuildable scratch, so no cleanup deletion or SSD move is
authorized or needed.

## Triality and integration honesty

- DSL: `src/tac/witness_dsl/segnet_exact_forward_transfer_policy.py` derives the finite thread
  tournament, static child lifecycle, canary provenance, exact replay gate, and false-authority
  fallback.
- Equation: `src/tac/canonical_equations/segnet_exact_forward_cpu_thread_law_20260713.py` preserves
  historical v1 and adds `segnet_exact_forward_cpu_thread_static_process_v2`. V2 independently
  validates terminal artifact bytes, timings/sign test, eight unique children/PIDs, one segment per
  pass, per-stage measurement/replay equality, pair78, and GO/NO-GO admission. It preserves
  `argmax_flip_count=None` for these two negative anchors rather than coercing zero.
- DAG: the shared `sub015_DAG` surface is live/hot under a sibling, so this agent did not edit it.
  Main must append the ready FEED below after ownership clears; deferral avoids a shared-hot-file
  collision and is the only incomplete triality landing.
- Runtime integration: deliberately deferred to main. The exactness gate failed, so no trainer flag,
  policy activation, or sibling-owned trainer/scorer module was changed.

### FEED-task456-static-thread-terminal (2026-07-13) — scoped NO-GO

**SUPERSEDES only the rankability interpretation of FEED-task456-exact-forward-thread-control;
historical receipt bytes remain evidence.** Two independent local Torch builds measured a
process-static one-thread eager-NCHW frozen-SegNet forward at 2.9562855478032297x and
2.9970426994326185x versus the six-thread baseline at n600. Both exactness gates fail on the same
15 receiver-pair SHA indices, including pair78; each arm is internally exact across two measurements
and two independent full replays. Pixel flip cardinality is UNKNOWN because raw argmax tensors were
not persisted. `verdict_scope=fresh-child process-static interop1 eager-NCHW 6-vs-1 on the two
fingerprinted local macOS CPU/Torch builds and this n600 receiver corpus only`.
`research_only=true`; `score_claim=false`; `promotion_eligible=false`; `[macOS-CPU advisory]`;
contest-CPU unmeasured; pointer **UNMOVED**.

## STORES CONSULTED

Full `CLAUDE.md`; full `AGENTS.md`; full `docs/operating_manual_craft_handoff.md`; `PROGRAM.md`;
top Claude memory entries; relevant Codex memory summary; v7.5 §8 and v8 specifications; canonical
frontier/task/lane/subagent surfaces; Task #456 predecessor checkpoints, probe, tests, policy,
equation, memo, historical receipts, and terminal stage artifacts; real n600 receiver bytes and
frozen SegNet weights; local Torch 2.12.1/2.12.0 runtime custody; scorer source/call graph; active
sibling ownership map; shared DAG ownership state. Paid/cloud/provider state, the live trainer,
protected run dirs outside these new receipt trees, MPS, CUDA, Metal, and `upstream/evaluate.py`
were not actuated.

