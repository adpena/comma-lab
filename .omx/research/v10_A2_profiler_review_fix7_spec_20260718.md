# v10 A2 profiler review fix 7 — governed profiler and receipt custody

Date: 2026-07-18  
Lane: `v10_A2_profiler_20260718`  
Authority: `codex_delegate:v10_A2_profiler_20260718:20260718T160947Z`  
Status: **PRE-LAUNCH / NO-GO UNTIL CLEAN RE-REVIEW**

## Fresh-seal reset

Three independent post-integration reviews agreed that the mathematical
lattice bounds, source seeding, receiver parse-back, semantic replay, and cache
chain are sound.  They nevertheless reproduced launch- and receipt-level
false-authority gaps on working-tree bundle
`b09b0731ad993105bb2c880718c116ebd44e526d5923ed5792001d91c0cc20bf`.

1. The profiler's generic admission check was advisory when repository
   enforcement was unarmed, and an imported `run_profile()` bypassed it
   entirely.
2. The rich profile identity was reduced to a hash and never persisted.  The
   final receipt omitted that identity/hash, exact argv, and terminal stage
   root, so its compressed-byte rows were not independently reproducible.
3. `MIN_DESCRIPTION_EXACT` conflated an exhaustive per-block signed-residual
   selector optimum with a global zlib/Brotli stream optimum.
4. Stage timing/RSS values had no exact schema or stated non-replayable custody,
   and the profiler initialization path could strand PID-named scratch.

## Required correction

### Governed execution

- Require the real repository governed-child marker inside `run_profile()`
  before production output, cache reads, scorer loading, or other heavy work.
  Force strict/no-bypass helper semantics as the extractor does.  Test-only
  local output remains the only carveout.
- Add required positive `--rss-cap-mb` and `--timeout-seconds` production
  arguments.  Bind them as requested outer-governor limits and state that the
  profiler does not self-enforce them.  A launch still goes through the
  governed `safe_run` process-group/system-memory path.
- Regress raw/unarmed and bypass-only refusal, governed-marker acceptance, and
  imported-call refusal before any output is created.

### Durable identity and output certification

- Atomically persist canonical `identity.json` before stages.  Resume must
  compare the whole persisted object with a fresh identity and re-derive its
  SHA-256, not trust a stored digest alone.
- Bind every executed helper that affects inputs or validation, including the
  feature-cache module and stored-NPZ helper module.  Record Python, Torch,
  NumPy, zlib build/runtime, Brotli version, and exact codec settings.
- Create a machine-readable output certification with output path, identity,
  exact rebuild argv, storage waterfall/preflight, rebuildability/retention,
  false-authority flags, and cleanup disposition.  Use a stable certified
  creation-staging directory; recover or remove only certified pre-stage
  scratch and refuse unidentified bytes.  No PID-orphaned initialization file.
- The final receipt must bind the full identity hash, `identity.json` hash,
  exact argv, ordered stage count, terminal stage-chain SHA-256, progress
  pointer SHA-256, and stream hashes/bytes already reported by the RD row.
  Resume and final validation must reject config, source, stage-root, or
  identity substitutions.

### Exact claim scope

- Rename the exhaustive selector label to
  `PER_BLOCK_RECEIVER_PUBLIC_SELECTOR_MINIMUM_EXACT`.
- Keep exact feasible-cardinality and exhaustive per-block selector facts
  distinct from global compression.  Persist explicit booleans for
  `per_block_selector_minimum_proved` and
  `global_compressed_stream_minimum_claim`; the latter stays false unless a
  global framed-stream optimization is actually proved.
- `min_description_claim` must not imply zlib/Brotli/global MDL optimality.
  Actual raw/zlib/Brotli byte counts remain measurements of the selected
  receiver-public stream.

### Timing and terminal custody

- Validate an exact timing schema: finite positive wall time, finite positive
  blocks/second consistent with block count and wall time, and nonnegative
  integer peak RSS.
- Label timing as a measured process observation protected by stage-chain
  custody but not semantically replayable.  Produce a final per-frame timing
  summary bound by the terminal stage root.  Do not claim a fresh timing replay
  on resume.
- Add coordinated rehash/substitution regressions and a positive clean-room
  final-receipt validation test.

### Intermediate-audit refinements

The first fix7 implementation audit also refused the following shortcut
classes; they are part of this same gate.

- Local-output test policy must never bypass admission.  `run_profile()` always
  requires the strict governed marker; tests set that marker explicitly.
- Caller-provided `exact_argv` must parse back to the effective typed
  configuration (with only the intentional resume-mode difference), so an
  imported caller cannot attest a fictitious rebuild command.
- Bind the executed admission-guard and tool-bootstrap modules, plus all zlib
  and Brotli mode/window/memory/strategy settings that determine bytes.
- Certified recovery may remove only known atomic temp patterns inside an
  already identity-matched staging/output root.  Reject symlinked identity,
  progress, certification, receipt, stage directory, and stage files.
- Clean-room final validation must require immutable-input semantic replay of
  every stage.  A self-consistent replacement chain is not scientific custody.
- A stale final receipt from a crash after a resumed stage/progress commit is a
  recoverable prior-prefix artifact only when its embedded progress snapshot
  validates exactly; otherwise refuse.

## Stop rule

No source freeze or real n600 work until fix7 passes focused tests and three
fresh adversarial clean passes.  This fix adds no score, contest-axis,
promotion, factor-10/Pose, sacred-run, or pointer authority.  MAIN landing
review remains mandatory.
