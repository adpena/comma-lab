# PR101 Nonlocal Sweep Agent - 2026-05-14

Scope: push PR101/HNeRV-family nonlocal sweeps using existing PR101/PR106
artifacts and local CPU/proxy tools only. No GPU spend and no provider dispatch
from this agent turn. I did not edit shared source code.

## Preflight

- Read `CLAUDE.md`, `AGENTS.md`, and `PROGRAM.md`.
- Checked `.omx/state/lane_registry.json` and
  `.omx/state/active_lane_dispatch_claims.md`.
- Read current 24-hour directive files:
  `.omx/research/lane_dp1_comma2k19_autoload_log_incremental_20260514_directive_layered_chunking_efficiency_oss_20260514.md`
  and
  `.omx/research/lane_grand_council_maximize_value_20260514_directive_zen_floor_field_medal_grade_20260514.md`.
- Read HNeRV public-frontier memory; it reinforced exact replay custody,
  scorecard/anatomy-first routing, and no cosmetic-byte overclaims.
- Attempted checkpoint write per CLAUDE.md crash-resume protocol, but this
  checkout's `tools/subagent_checkpoint.py` currently exposes only the `read`
  subcommand. No state write was made through that tool.

## Commands

```bash
/usr/bin/time -l .venv/bin/python tools/build_pr101_nonlocal_sweep_packets.py \
  --out-dir experiments/results/pr101_nonlocal_sweep_20260514_agent_codex \
  --top-k 5 \
  --exclude-candidate-id bias_refine_cmaes_0050 \
  --exclude-candidate-id bias_refine_cmaes_0053 \
  --exclude-candidate-id bias_refine_cmaes_0044
```

```bash
env PATH=/Users/adpena/Projects/pact/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  /usr/bin/time -l .venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/packets/bias_refine_cmaes_0047/archive.zip \
  --inflate-sh experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/packets/bias_refine_cmaes_0047/inflate.sh \
  --upstream-dir upstream \
  --video-names-file upstream/public_test_video_names.txt \
  --device cpu \
  --json-out experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/local_cpu_advisory/bias_refine_cmaes_0047/contest_auth_eval.macos_cpu_advisory.json \
  --allow-temp-work-dir \
  --inflate-timeout 1800 \
  --evaluate-timeout 1800 \
  --expected-runtime-tree-sha256 8ddcecfb9008d7b6c23944579099ca4a8a5915431ef61de85d17372b19f1e5ce
```

## New Artifacts

- `experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/summary.json`
- `experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/agent_findings.json`
- `experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/exact_ready_pr101_nonlocal_bias_queue.json`
- `experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/packets/*`
- `experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/candidate_manifests/*`
- `experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/exact_ready/*`
- `experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/local_cpu_advisory/bias_refine_cmaes_0047/contest_auth_eval.macos_cpu_advisory.json`
- `experiments/results/pr101_nonlocal_sweep_20260514_agent_codex/logs/*`

## Packet Sweep

The packet tool selected the next five PR101 runtime-bias rows after excluding
exact-CUDA negatives `0050` and `0053` plus the already-screened local CPU
negative `0044`. All packets keep the same charged PR101 archive bytes and
SHA (`178258`, `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`);
only runtime bias parameters differ. `score_claim=false` for every row.

| candidate | proxy objective | runtime tree sha256 |
|---|---:|---|
| `bias_refine_cmaes_0047` | 0.19286038938290923 | `8ddcecfb9008d7b6c23944579099ca4a8a5915431ef61de85d17372b19f1e5ce` |
| `bias_refine_cmaes_0046` | 0.19286058338451523 | `adc97a066038adc334dd574270ccea82c567c194e8c4cc7c9663f6b595dc1811` |
| `bias_refine_cmaes_0062` | 0.19286131350449545 | `317d9bb101c0dca0854dcb334fc05172315b82a982eb09d93830bc005229738d` |
| `bias_refine_cmaes_0042` | 0.19286234038533343 | `6215aba0f2827ca9d86de1babc0096222f2a4f79dd96f1649d89f02b7aa5a18c` |
| `bias_refine_cmaes_0061` | 0.1928627239171165 | `7f5f424a57bcdf59a9f341296c93ca6f868844cb1ce08158349be060d7aea519` |

Build runtime: 2.02 s real. Max RSS: 222,445,568 bytes.

## Scores And Axis Separation

| artifact | axis | score | seg | pose | bytes | classification |
|---|---|---:|---:|---:|---:|---|
| PR101 local baseline | `[macOS-CPU advisory]` | 0.1928610127024255 | 0.00056039 | 0.00003286 | 178258 | baseline only |
| PR101 `0044` | `[macOS-CPU advisory]` | 0.1929870127024255 | 0.00056165 | 0.00003286 | 178258 | local advisory regression |
| PR101 `0047` | `[macOS-CPU advisory]` | 0.19313928561413787 | 0.00056309 | 0.00003289 | 178258 | local advisory regression |
| PR101 `0050` | `[contest-CUDA]` | 0.2265033667659732 | 0.00066465 | 0.00017093 | 178258 | exact CUDA regression |
| PR101 `0053` | `[contest-CUDA]` | 0.22657615737364284 | 0.00066539 | 0.00017092 | 178258 | exact CUDA regression |
| PR106-R2-HDM8-HLM2-XMEMBER | `[contest-CUDA]` | 0.20636166502462222 | 0.00064260 | 0.00003236 | 186395 | current HDM8 anchor |
| FES1 frame-exploit selector | `[contest-CUDA]` | 0.2261263460995081 | 0.00064260 | 0.00013847 | 187209 | CUDA regression |
| FES1 frame-exploit selector | `[contest-CPU/modal-linux-x86_64]` | 0.2088908495021802 | 0.00063198 | 0.00004426 | 187209 | CPU-axis improvement, still above 0.192 |

The `0047` local advisory eval completed full 600 samples: inflate 39.04 s,
evaluate 475.90 s, total 516.23 s, max RSS 10,856,005,632 bytes, inflated raw
aggregate SHA `a4f38ce2c9e323f8cfc23d26035d67fe465688be5cf6a486167e5fe62c3a71b8`.

## Byte-Closed Variant Review

1. PR101 nonlocal runtime-bias variants are not plausible sub-0.192 candidates
   on current evidence. The two exact-CUDA probes (`0050`, `0053`) both
   blow up PoseNet relative to PR101/HDM8, and the two local advisory probes
   (`0044`, `0047`) both regress against the matching local PR101 baseline.
2. PR101 ZIP repacking is blocked. The source archive is already a single
   stored `x` member with the theoretical 100-byte ZIP overhead; deflated
   rewrites add 55 bytes.
3. FES1 is contest-compliant in the narrow charged-side-info sense: the selector
   payload is inside `archive.zip` (`385` wire bytes, `+387` archive bytes), no
   scorer at inflate. But the matching-axis evidence does not support dispatch:
   CUDA regressed to `0.226126`, and CPU improved only to `0.208891`, still above
   the operator's `0.192` HNeRV-family cutoff.
4. HDM8/HLM2 rate-only work has an exact-CUDA anchor at `0.20636166502462222`.
   Its current rate-only win is 10 bytes versus HDM7, useful custody but not
   enough to justify more PR101/HNeRV local-minimum variants.

## Dispatch Decision

No GPU/provider dispatch is warranted from this sweep.

Dispatchable condition: a new non-bias or CUDA-derived selector mechanism must
first produce a byte-closed packet with old/new section SHA custody, charged
bytes, runtime-consumption proof, exact-readiness, and component/rate math
plausibly below `0.192` on the matching axis. Then the lane must be claimed via
`tools/claim_lane_dispatch.py` before any remote/GPU run.

Highest-EV next step without new spend: harvest the already-active CUDA-in-loop
HDM8 postfilter/palette sweeps, then rebuild a selector only from CUDA-positive
modes. If those modes do not beat HDM8 on CUDA, leave this HNeRV/postfilter basin
and move to a nonlocal mechanism with a different failure mode.

## Wire-In Status

`research_only=true`; this memo is the operator-visible control ledger. No
solver code, bit allocator, or autopilot hook was changed because this turn
produced a negative/blocked routing result rather than a promotable atom.
