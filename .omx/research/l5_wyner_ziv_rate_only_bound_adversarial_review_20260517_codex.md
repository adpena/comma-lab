# L5 Wyner-Ziv Rate-Only Bound Adversarial Review - 2026-05-17

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
- partner WIP reviewed but not modified:
  - `.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md`
  - `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md`
  - `docs/pr_writeups/cpu_frontier_fec6_20260517.md`

## Verdict

L5 Wyner-Ziv pose-delta coding remains a high-value P0 because the current FEC6
CPU anchor only needs about `78` charged bytes to cross below `0.192`. But the
new grand-reunion WIP's `-0.008` to `-0.015` claim cannot be a rate-only
consequence of compressing `poses.bin` from about `4800` bytes to `1500-2000`
bytes. The score math bounds that claim.

If L5 is purely rate-axis and decoded pose values are unchanged, its predicted
gain should be treated as about `-0.0019` for a `4800 -> 2000` byte shrink, and
at most `-0.0032` even if the entire `4800` byte pose stream disappeared. Any
claim in the `-0.008` to `-0.015` band, or the campaign-plan `0.174-0.182`
post-L5 row, must prove component movement (`d_pose` or `d_seg`), a larger
charged-byte section, or both.

This is not a lane kill. It is a correction to prevent a real L5 score-lowering
opportunity from being dispatched under impossible arithmetic.

## Closed-Form Audit

Contest rate coefficient:

```text
dS/d(byte) = 25 / 37,545,489 = 6.657151674078856e-7
```

FEC6 current anchor:

```text
score_cpu = 0.1920513168811056
archive_sha256 = 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
avg_pose = 0.00002943271901344679
pose_term = sqrt(10 * avg_pose) = 0.017155966604492673
```

L5 WIP rate-only premise:

```text
poses.bin original bytes ~= 4800
candidate bytes ~= 2000
saved bytes ~= 2800
decoded pose values unchanged
```

Exact rate-only result:

```text
score_saving = 25 * 2800 / 37,545,489 = 0.0018644050687420797
projected score = 0.1920513168811056 - 0.0018644050687420797
                = 0.19018691181236352
```

Absolute rate-only ceiling if the full `4800` byte pose stream disappeared:

```text
score_saving_max = 25 * 4800 / 37,545,489 = 0.0031961229749864224
projected score_max = 0.18885519390611918
```

So L5 rate-only is enough to beat the operator's `<0.192` threshold if it emits
a byte-closed consumed packet, but it is not enough to justify the
`0.17-0.18` route in the WIP.

## Overclaim Test

New reusable guardrail:

- `src/tac/score_geometry.py::audit_rate_only_delta_claim`
- `src/tac/score_geometry.py::required_byte_savings_for_score_delta`
- `src/tac/score_geometry.py::score_saving_from_byte_savings`
- `src/tac/tests/test_score_geometry.py::test_rate_only_delta_audit_bounds_l5_pose_stream_claim`

Audit output for the WIP's lower-band claim:

```text
claimed_score_saving = 0.008
candidate_saved_bytes = 2800
candidate_rate_only_score_saving = 0.0018644050687420797
required_saved_bytes_for_claim = 12015
max_possible_score_saving_if_section_removed = 0.0031961229749864224
feasible_from_candidate_savings = false
feasible_even_if_section_removed = false
blocker = claim_exceeds_rate_only_section_capacity
```

Audit output for the WIP's upper-band claim:

```text
claimed_score_saving = 0.015
candidate_saved_bytes = 2800
candidate_rate_only_score_saving = 0.0018644050687420797
required_saved_bytes_for_claim = 22528
max_possible_score_saving_if_section_removed = 0.0031961229749864224
feasible_from_candidate_savings = false
feasible_even_if_section_removed = false
blocker = claim_exceeds_rate_only_section_capacity
```

## What Would Make The Larger Claim True

For the `-0.008` claim, after the `2800` byte rate saving, the missing score
delta is:

```text
0.008 - 0.0018644050687420797 = 0.006135594931257921
```

At the FEC6 CPU pose term, that missing delta would require reducing
`avg_pose` from `2.943271901344679e-5` to about
`1.2144859181623492e-5`, a pose reduction of
`1.7287859831823297e-5`.

For the `-0.015` claim, the missing delta after rate-only savings is
`0.01313559493125792`, requiring `avg_pose` about
`1.6163388390948404e-6`, a reduction of `2.781638017435195e-5`.

Therefore an L5 claim in the original WIP band must stop calling itself
rate-only. It must ship a component-moving proof:

1. paired CPU/CUDA exact-eval component deltas,
2. raw-output aggregate SHA for each cell,
3. decoded-pose parity or decoded-pose-delta manifest,
4. byte-consumption proof for the Wyner-Ziv stream,
5. exact attribution of rate-only versus PoseNet/SegNet movement.

## Dispatch Decision Change

The corrected L5 decision is:

- Continue L5 as P0 if the next artifact is a byte-closed FEC6/A1 packet that
  consumes a changed pose-delta stream and saves at least `78` charged bytes.
- Do not dispatch an L5 run expecting `-0.008` to `-0.015` from rate shrink
  alone.
- Treat `-0.0019` to `-0.0032` as the honest rate-only band for the current
  `poses.bin` section.
- Escalate back toward `-0.008` to `-0.015` only after a score-response probe
  shows decoded-frame or decoded-pose component movement, or after the encoded
  side information covers a larger charged-byte section than `poses.bin`.

This keeps the L5 staircase alive while removing the arithmetic false
authority that would otherwise send the next dispatch wave into a bad
expectation.
