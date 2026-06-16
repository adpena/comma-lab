# Local artifact pointers — gitignored bytes that live ONLY in the main tree

**Why this file exists (the bug class it permanently fixes).** `experiments/results/`
is broadly **gitignored** (`.gitignore` ~lines 89–129: `experiments/results/`,
`experiments/results/*`, `experiments/results/**/*.pt`, with only narrow `!`
exceptions). So the converged basin checkpoint and the verified frontier archive
are **absent from any `git worktree`'s working tree** — a worktree checks out only
*tracked* files. A worktree-isolated agent that does `ls experiments/results/…`
in its CWD sees an empty dir and may (a) fail, or (b) **fabricate** a result
against an artifact it cannot see (a NO-FAKE supreme-rule violation).

**The fix (this manifest = certify-don't-block, durable + tracked).** The bytes
still exist on disk in the **main tree**; absolute paths resolve from inside a
worktree (worktrees are not filesystem sandboxes). Any agent — worktree or not —
MUST read these artifacts via their **absolute main-tree paths** below, and MUST
verify the recorded sha256/bytes before using them. NEVER report a measured
result against an artifact whose sha does not match here.

Authority: all rows `[contest-CPU advisory] NON-PROMOTABLE` except the frontier
row (`[contest-CPU]` authoritative, pointer-of-record). Pointer SoT remains
`.omx/state/canonical_frontier_pointer.json`; this file is a path/locator index,
not a score record.

## Converged 600-pair base_ch20 basin (G0 warm-start, long-train warm-start, G2 real-archive test input)
Absolute dir: `/Users/adpena/Projects/pact/experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best`

| file | sha256 (first16) | bytes |
|---|---|---|
| `best_ema_decoder.pt`  | `1d291e206b702f4a` | 342081 |
| `best_ema_latents.pt`  | `09a98f275c85d200` | 68840  |
| `best_archive.bin`     | `9dee820bd5b051e1` | 89136  |
| `best_meta.json`       | `726ee74a311ffcf6` | 317    |

`best_meta.json`: base_ch20, ep2120, d_seg **0.002600919948890805**, d_pose
**0.00034168662969022987**, rate 0.00237408, advisory score 0.377898,
stage1_v328_ce. This is the symposium's "600-pair plain-CE basin 0.002601".

## Verified frontier archive (G4 additive-adapter base; the 0.19110 bytes)
The frontier pointer stores the archive **by sha256 + bytes**, not by path. Locate
the bytes by sha before adapting; do NOT assume a path.

- sha256: `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e`
- bytes: **177169**
- lane / architecture_class: `lane_pr110_payload_entropy_recode_20260610`
- axis: `contest_cpu` `[contest-CPU]`, score **0.19109982419209975**, hardware linux_x86_64_cpu
- measured_at_utc: 2026-06-10T08:43:20Z
- locate command (run in MAIN tree):
  `find experiments .omx -type f -name '*.bin' -o -name '*.zip' 2>/dev/null | while read f; do [ "$(shasum -a256 "$f"|cut -d' ' -f1)" = b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e ] && echo "$f"; done`

## Worktree-agent contract (NO-FAKE)
1. Build machinery (export, parity gate, adapter, byte-close) on **tracked code**
   (`src/tac/torch_vehicle/**`, `experiments/launch_*.py`) — this needs NO gitignored artifact.
2. For any **real-artifact** step (load the basin, adapt the frontier bytes): use the
   absolute main-tree path above, verify sha256, and if the artifact is unreachable
   from the worktree, STOP and mark the step `pending-main-tree-integration` —
   do NOT fabricate a number.
3. Code lands back on `main` (CLAUDE.md "Main branch source of truth"); it runs in
   the main tree where these artifacts exist.
