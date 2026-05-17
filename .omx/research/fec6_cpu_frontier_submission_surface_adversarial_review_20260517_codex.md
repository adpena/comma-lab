# FEC6 CPU Frontier Submission-Surface Adversarial Review - 2026-05-17

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_submission=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
- reviewed partner WIP left unmodified:
  - `.omx/research/full_problem_space_reverse_engineering_cpu_gpu_both_20260517.md`
  - `.omx/research/alien_tech_reverse_engineering_pr106_format0_family_20260517.md`
  - `docs/pr_writeups/cpu_frontier_fec6_20260517.md`

## Verdict

FEC6 is a real, byte-closed `[contest-CPU]` anchor on Linux x86_64 with score
`0.1920513168811056`, archive SHA-256
`6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`, and exact
archive size `178517` bytes. It is not submission-ready from the current WIP
surface. The current WIP writeup is useful signal, but it over-compresses
evidence into release language before the packet surface, public-source
scanner, runtime-tree claim row, and CPU-score-claim schema are clean.

The practical decision is:

1. Treat FEC6 as the current canonical best `[contest-CPU]` anchor and a
   public-leaderboard-relevant reproduction target.
2. Do not submit the current WIP packet.
3. Do not spend more P0 time shaving same-runtime FEC6 bytes unless a profiler
   finds at least `78` charged bytes of real, consumed archive savings.
4. Use FEC6/A1 as the Rule #6 substrate for component-moving or
   scorer-aware byte-closed bolt-ons, not as a local-basin polishing trap.

Public context check: the live comma leaderboard page still exposes the video
compression challenge with a top visible row at `0.193` for PR #101 as of this
review (`https://comma.ai/leaderboard`). That makes the CPU-axis anchor
important, but it does not remove the submission-surface blockers below.

## Positive Evidence Preserved

- CPU exact-eval artifact:
  `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json`
  - `score = 0.1920513168811056`
  - `score_axis = contest_cpu`
  - `lane_tag = [contest-CPU]`
  - `cpu_leaderboard_reproduction_eligible = true`
  - `samples = 600`
  - `archive_bytes = 178517`
  - `avg_segnet_dist = 0.0005602915120599784`
  - `avg_posenet_dist = 0.00002943271901344679`
- CUDA paired artifact:
  `experiments/results/modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json`
  - `score = 0.22621002169349796`
  - `score_axis = contest_cuda`
  - `lane_tag = [contest-CUDA]`
  - same archive SHA-256 and size
- Xray paired-axis review:
  `experiments/results/xray_paired_cpu_cuda_axis_delta_pr101_fec6_20260515_codex/paired_axis_delta.md`
  - verdict: `cpu_positive_cuda_miss_due_to_component_drift`
  - CPU strict gap to `<0.192`: `78` bytes
  - CUDA byte-equivalent gap to CPU target: `51378` bytes
  - raw output aggregate differs between CPU and CUDA, so CPU/CUDA are not
    interchangeable for promotion.
- Byte-escape review:
  `.omx/research/pr101_fec6_byte_escape_profile_20260515_codex.md`
  - same-frame byte-only realistic upper bound was about `16` bytes, below the
    `78` byte strict CPU gap.
  - selector bytes are charged and consumed; the mechanism is not a no-op.

## Strict Submission-Surface Check

Command:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  --contest-cpu-auth-eval-json experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json \
  --auth-eval-json experiments/results/modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json \
  --archive-manifest-json experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive_manifest.json \
  --contest-final \
  --submission-score-axis contest_cpu \
  --max-submission-score 0.1921 \
  --expect-single-member x \
  --expected-archive-sha256 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf \
  --expected-archive-size-bytes 178517 \
  --expected-runtime-tree-sha256 f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166 \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515 \
  --expected-job-id modal-cpu-pr101-fec6-k16-wrapper-20260515T015430Z \
  --competitive-or-innovative-statement-file docs/pr_writeups/cpu_frontier_fec6_20260517.md \
  --require-competitive-or-innovative-statement \
  --public-scan-path docs/pr_writeups/cpu_frontier_fec6_20260517.md \
  --json-out experiments/results/fec6_cpu_submission_surface_review_20260517_codex/pre_submission_compliance_cpu.json \
  --strict
```

Result: `passed=false`.

Evidence file:
`experiments/results/fec6_cpu_submission_surface_review_20260517_codex/pre_submission_compliance_cpu.json`.

Important positive after hardening: the frontier-regression section now derives
the candidate score from the selected-axis auth-eval record when
`--submission-score` is omitted:

```json
{
  "candidate": {
    "axis": "contest_cpu",
    "score": 0.1920513168811056,
    "score_source": "strict_formula"
  },
  "canonical_best": {
    "contest_cpu": {
      "score": 0.1920513168811056,
      "archive_sha256": "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
    }
  }
}
```

This fixes the previous fail-open `AttributeError("'Namespace' object has no
attribute 'submission_score'")` warning in the frontier checker path.

## Blocking Findings

P0 release-surface blockers:

1. The WIP writeup references
   `submissions/pr101_fec6_fixed_huffman_k16/archive.zip`, but the actual
   reviewed archive lives at
   `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`.
2. The reviewed `submission_dir` lacks `archive.zip`.
3. The reviewed `submission_dir` lacks `report.txt`.
4. The archive manifest matches top-level SHA and size, but lacks the member
   list. The actual archive has one member, `x`, with uncompressed/compressed
   size `178417`, CRC `3299285626`, and SHA-256
   `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`.
5. The CPU auth-eval artifact is stamped as CPU-reproduction-eligible but has
   `score_claim=false`, `score_claim_valid=false`, and
   `promotion_eligible=false`. That is honest for current custody, but the
   release checker currently treats it as not a final CPU submission claim.
   Do not paper over this by editing evidence JSON; regenerate or define the
   CPU-final schema explicitly.
6. The terminal lane-claim row binds the archive SHA but not the CPU runtime
   tree SHA
   `f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166`.
7. The public-source and axis-label checks scan `submission_dir/README.md`,
   not the untracked WIP writeup. The release packet therefore lacks a public
   source repo link, pinned revision, reproduce command or SHA binding, and
   explicit `[contest-CPU]` / `[contest-CUDA]` labels in the scanned public
   packet surface.
8. The post-deadline policy scanner did not accept the current WIP statement
   as explicitly naming competitive and/or innovative status.
9. The selected-axis runtime-tree expectation is under-specified for a paired
   CPU/CUDA packet: the CPU artifact records runtime tree
   `f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166`; the
   CUDA artifact records
   `12d4315dcbf0943f07fcd357eaf06b126a999c252f8edeb2681179831248df04`; both
   share portable runtime tree
   `6811f28c2116757851b4a6e68a5bdefd7866b4da1867eb13b3c62405de8834df`.
   A final checker must make the selected-axis and paired-axis runtime
   semantics explicit instead of accepting a vague single expected hash.

P1 wording/science blockers in the WIP writeup:

1. The phrase "CPU axis is public-leaderboard axis" should be softened unless
   the official rules or runner config are cited directly. Stronger wording:
   "FEC6 is exact on our Linux x86_64 `[contest-CPU]` reproduction axis; the
   live public leaderboard exposes PR #101 at `0.193`, but submission mode and
   runner semantics must be confirmed in the final packet."
2. The WIP says it avoids approximate placeholders but still contains
   approximate values such as `~107`, `~0.195`, `approximately`, and
   order-of-magnitude phrasing. Convert each to either exact evidence or
   explicitly tagged estimate.
3. Mechanism-level claims about FastViT, selector behavior, and per-layer
   epsilon should remain tagged as hypotheses unless tied to an xray artifact
   that proves the claimed causal route.

## Score-Lowering Consequence

FEC6 is too close to the CPU target for unfocused byte polish. The byte-only
gap to `<0.192` is `78` bytes, while the measured same-runtime byte-escape
upper bound was about `16` bytes. The CUDA miss is component-dominated and
equivalent to roughly `51378` charged bytes, so rate-only work is the wrong
knob on CUDA.

The next score-lowering work should be:

1. Materialize a clean contest-final CPU packet surface for FEC6 only if the
   operator wants a CPU-mode submission: `archive.zip`, `report.txt`, public
   README, canonical member manifest, pinned source/repro command, terminal
   claim row with runtime-tree SHA, and explicit CPU-final score-claim schema.
2. In parallel, keep frontier development on component-moving Rule #6
   bolt-ons: Balle hyperprior, PR101-style entropy stack with changed consumed
   runtime grammar, and VQ-codebook. Do not route P0 back into PR106 or FEC6
   local-basin polish.
3. If FEC6 is reopened for score lowering, require a profiler finding of at
   least `78` real charged bytes or a component-moving selector/curriculum
   change before dispatch.

## Parent-Level OMX Markdown Check

The `.omx` parent directory contains two tracked Markdown files:

- `.omx/notepad.md`: stale April AV1/Track-B notebook. It preserves useful
  historical signal about film-grain, colorspace/range, and local scorer
  hardening, but it is not current L5, TT5L, FEC6, or Rule #6 authority.
- `.omx/release_manifest_v0.2.0-rc1.md`: historical release-candidate manifest.
  It is useful release-hygiene context, but not current score authority.

This review therefore treats `.omx/state/current_focus.md`,
`.omx/state/next_experiments.md`, and dated `.omx/research/*_20260517*.md` as
the current control plane, with parent-level Markdown retained as preserved
history rather than active routing authority.
