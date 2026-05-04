# Archive Bit Budget Profiler - 2026-05-02 Codex

## Scope

Implemented `experiments/archive_bit_budget_profiler.py` as a read-only,
deterministic profiler for contest archives. It is intentionally empirical
byte evidence only: it does not inflate frames, load scorer modules, import the
contest runtime, dispatch jobs, or make promotion/score claims.

## Contest Compliance Language

Every report carries:

- `score_claim=false`
- `promotion_eligible=false`
- evidence grade `empirical_byte_profile`
- canonical score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA`
- a compliance note that self-compression probes are planning signals only
  until all decoder/runtime bytes are charged inside the archive and exact CUDA
  auth eval validates identical bytes.

## Output Contract

The profiler accepts one or more ZIP archives and emits deterministic JSON,
CSV, and Markdown-friendly output. Reported fields include:

- archive bytes, SHA-256, ZIP member count, ZIP payload/header/global overhead,
  and archive rate-term contribution under
  `25 * archive_bytes / 37,545,489`;
- per-member compressed/uncompressed bytes, CRC, ZIP method, header estimate,
  uncompressed SHA-256, entropy, generic zlib/lzma/Brotli probes, type guesses,
  and rate estimates;
- payload segment anatomy for recognized single-member `p` payloads, including
  RPK1, RP2 fixed3, PR64 length-table, PR64 outer-Brotli length-table, and
  public PR63/PR67 fixed-slice length buckets when recognizable;
- ranked self-compression opportunities, explicitly marked
  `directly_deployable=false`.

## Guardrails

- ZIP member names are zip-slip checked; absolute paths and parent traversal
  fail closed.
- Duplicate ZIP member names fail closed.
- Directory members fail closed because contest archives must contain charged
  payload files, not extraction-side filesystem structure.
- RPK1 payload member names are checked against the contest payload allowlist
  before segment anatomy is accepted.
- The profiler does not import `torch`, `tac`, `upstream`, or
  `submissions/robust_current` helpers.

## Verification

Focused tests cover:

- argparse surface for archive inputs and JSON/CSV/Markdown outputs;
- RPK1 segment anatomy and no-score-claim metadata;
- outer-Brotli PR64 length-table `p` anatomy;
- zip-slip and duplicate-member rejection;
- CLI output file creation and no runtime/scorer imports during execution.

Verification commands were run after implementation and are recorded in the
final handoff.
