# Contest Harness Bug Classes - 2026-05-04 Codex

This ledger records durable bug classes found during the PR85/PR90/PR91
frontier push. It is not a score ledger. Score and lane outcomes remain in the
dated lane ledgers and exact artifact directories.

## Bug Class: Exact Runtime Contract Drift

Observed incident:

- PR85 + STBM1BR lossless mask-recode candidate decoded bit-identical masks in
  the builder and reduced archive size, but the first exact T4 submission used
  the plain PR85 replay inflater.
- That inflater parsed the single-member PR85 bundle but treated non-QMA mask
  bytes as Brotli AV1 and failed before score with `brotli: decoder failed`.
- Evidence grade: `invalid` pre-score harness/runtime failure. No score claim.

Permanent protection target:

- Candidate builders must distinguish local/runtime support from exact-eval
  runtime support.
- Exact-eval readiness must require the actual submitted `inflate.sh` runtime
  to be aware of the candidate wire format, not merely the repo's
  `submissions/robust_current` runtime.
- Manifests should record the exact runtime path, runtime tree hash when
  available, and format probes proving the submitted runtime can parse the
  charged payload.

Current mitigation:

- A dedicated PR85 replay runtime with an explicit STBM1BR mask branch was
  created under the STBM result directory and used for the corrected T4 queue.
- A subagent owns the permanent builder/preflight guard so the same class cannot
  claim `ready_for_exact_eval_after_lane_claim` from the wrong runtime again.

## Bug Class: CLI Surface Drift And Misspelled Remote Flags

Observed incident pattern:

- Repeated remote orchestration friction came from stale assumptions about
  argparse surfaces, SSH aliases, and remote/preflight flags.
- A live example during this turn: searching for patterns beginning with
  `--...` without `rg -e` made `rg` treat the pattern as a flag. This is a
  small local command issue, but it is the same class as stale `--remote` /
  `--ssh-target` invocations: flags must be validated against the real parser
  before dispatch.

Permanent protection target:

- Remote submit wrappers should expose a dry-run/doctor path that enumerates
  the real argparse surface and validates SSH target configuration before
  submission.
- Unknown or likely-misspelled critical flags should fail with actionable
  diagnostics before staging, queue submission, or provider spend.
- Dispatch plans should preserve command arrays as the source of truth, and
  tests should assert key flags by parsed command list rather than by human log
  fragments.

Current mitigation:

- A subagent owns the Lightning CLI/SSH argument hardening pass across
  `scripts/lightning_exact_eval_repro.py`,
  `scripts/launch_lightning_batch_job.py`, and focused tests.

## Bug Class: Self-Reported Frontier Without Decode Parity

Observed incident:

- PR91 self-reports a score below the verified PR85 frontier, but local replay
  fails inside HPM1 entropy decode.
- Existing parity probes narrow the current raw/source contract failure to
  `frame=0, group=10, symbol_in_group=191` after `5951` decoded symbols.
- Byte and word order variants fail earlier, so dispatching PR91 without
  recovering the actual probability/entropy contract would burn queue time on
  a known invalid replay.

Permanent protection target:

- Public-submission replays must pass local deterministic decode/roundtrip
  contract tests before exact CUDA dispatch.
- A self-reported score from a PR title, comment, or report is external signal,
  not score evidence, until the exact archive bytes pass canonical CUDA auth
  eval.

Current mitigation:

- A subagent owns independent PR91/HPM1 parity recovery from first principles.

## Bug Class: Negative Stack Candidate Reuse

Observed incident:

- QRGB singleton and randmulti candidates transferred from PR90 to PR85 were
  exact T4 negatives. A combined STBM1BR + QRGB builder exists as useful
  infrastructure, but dispatching that stack without a new exact-positive QRGB
  premise would be low signal.

Permanent protection target:

- Stack builders may exist before dispatch, but their manifests must carry
  evidence gates for each component.
- Exact-negative atoms should be available as negative training signal and
  interaction probes, not silently recycled as promotion candidates.

Current mitigation:

- The STBM1BR + QRGB stack builder is held local-only pending corrected STBM
  exact evidence and a new reason to believe QRGB interaction changes sign.
