# Codex Findings: submission bundle byte shaving compliance

research_only: true
written_at_utc: 2026-06-05T00:44:02Z
worker_scope: compliance memo only
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Scope

This memo checks the local upstream submission/evaluate conventions and the
local strict compliance gates relevant to moving bytes into `inflate.py` /
runtime code, removing human-readable symbols, and preserving byte-frontier
SNeRV/HiNeRV work without crossing into uncharged payload relocation.

No web lookup was used. The local repo already contains `upstream/README.md`,
`upstream/evaluate.sh`, and `upstream/evaluate.py`.

## Local upstream contract

- `upstream/README.md:98-103` defines a submission as a PR with a download link
  to `archive.zip`, an `inflate.sh` that turns extracted `archive/` into raw
  frames, and optional compression/supporting assets.
- `upstream/README.md:114-119` gives the 30 minute official inflate limit,
  T4-GPU-vs-CPU evaluation substrate, and the key large-artifact rule: external
  libraries/tools are free unless they use large artifacts; neural networks,
  meshes, point clouds, PoseNet, and SegNet must be included in the archive and
  counted when used.
- `upstream/evaluate.sh:41-47` removes/recreates `archive/`, unzips
  `archive.zip`, and invokes `inflate.sh` as
  `inflate.sh "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"`.
- `upstream/evaluate.py:62-65` charges only
  `submission_dir/archive.zip`. Runtime source size is not directly part of the
  upstream rate meter.
- `upstream/evaluate.py:74-80` evaluates tensors loaded from
  `submission_dir/inflated`, asserts the contest raw shape, and computes
  PoseNet/SegNet distortions on original vs inflated batches.
- `upstream/evaluate.py:89-104` writes `report.txt` with component means,
  `archive.zip` byte size, rate, and final score formula.

## Local strict policy overlays

The local policy intentionally closes upstream-meter loopholes before any score
or promotion claim:

- `docs/contest_compliance_authority.md:34-42` says `archive.zip` is the
  charged payload, `inflate.sh`/`inflate.py` are runtime code, and runtime code
  is not a hiding place for score-bearing payloads. Compression-time scorer use
  is allowed; inflate-time use of large artifacts, including PoseNet/SegNet,
  requires those artifacts to be archive-charged.
- `docs/contest_compliance_authority.md:64-80` separates three procedural modes:
  `archive_seeded`, `weight_derived`, and `runtime_constant`. The safe default
  is to put score-bearing seeds, tables, transducers, and generated payloads in
  `archive.zip`, or derive them from already charged archive bytes.
- `docs/contest_compliance_authority.md:103-128` requires an authority packet
  with archive path/bytes/SHA, runtime tree SHA, exact inflate command, payload
  carrier inventory, mutation proof, scorer-free inflate proof, and separated
  `[contest-CPU]` / `[contest-CUDA]` evidence before promotion.
- `AGENTS.md:1231-1285` adds fail-closed rules: no upstream scorer edits, no
  score-affecting sidecars outside the archive, no hidden/resource/cache/debug
  ZIP members, deterministic archive construction, no ZIP parser divergence,
  exact eval through `archive.zip -> inflate.sh -> upstream/evaluate.py`, JSON
  artifacts as authority, runtime tree hash custody, no dummy/proxy promotion,
  and unknown renderer wire formats must fail closed instead of falling through
  to `torch.load()`.
- `experiments/contest_auth_eval.py:3-18` is the canonical local exact-eval
  harness for arbitrary submissions: exact archive, submission `inflate.sh`,
  upstream evaluator, and score.
- `src/tac/submission_packet/paired_auth_eval.py:56-62` keeps paired auth eval
  non-promotable by default and allows promotion only after paired CUDA plus
  Linux x86_64 CPU pass on the exact same archive bytes.

## Concrete gate behavior

- `scripts/pre_submission_compliance_check.py:402-488` inspects ZIP member
  safety, local-vs-central header parity, payload readability, duplicate names,
  expected single member, and packed payload multiplicity.
- `scripts/pre_submission_compliance_check.py:764-820` computes the runtime
  dependency manifest and runtime tree SHA, excluding custody files while
  including runtime files, declared dependency roots, repo-local `tac` imports,
  and upstream `evaluate.py` identity.
- `scripts/pre_submission_compliance_check.py:1116-1225` scans runtime `.py` and
  `.sh` files for scorer/eval imports, forbidden side effects, and disallowed
  non-stdlib imports.
- `scripts/pre_submission_compliance_check.py:2010-2038` requires
  `archive.zip`, executable `inflate.sh`, `report.txt`, the 3-arg
  `archive_dir/output_dir/file_list` contract, a file-list read loop, and no
  scorer/eval tokens in `inflate.sh`.
- `src/tac/submission_packet/builder.py:99-123` sets default bundle budgets:
  `inflate.py` <= 200 physical LOC, <= 2 external dependencies, canonical
  3-arg `inflate.sh`, and `set -euo pipefail`.
- `src/tac/submission_packet/linter.py:798-930` enforces UTF-8 `inflate.py`,
  the LOC budget, device-routing reviewability, and Pythonpath
  self-containment or vendoring.
- `src/tac/submission_archive.py:51-89` rejects unsafe archive member names and
  emits deterministic ZIP metadata. `src/tac/submission_archive.py:129-208`
  compares stored vs deflated single-member archives and records bytes/SHA while
  keeping all authority flags false.

## Current SNeRV bundle facts

Inspected bundle:
`/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/submission`

- `archive.zip`: 51,694 bytes,
  SHA-256 `2f57653c2e21834b731cb102a77d1fa603198f7c9ab13c0b82d94f3ad1f42ee2`.
- ZIP member set: exactly `0.bin`; member raw size 53,596 bytes, compressed
  size 51,586 bytes.
- `archive/0.bin`: starts with `SNAR2`,
  SHA-256 `533fa1a90b239da7b9da45e3e7fcc0ab45cdd3b9eee14a9be594371781445866`.
- `inflate.sh`: 453 bytes, canonical 3-arg wrapper with `set -euo pipefail`.
- `inflate.py`: 1,288 bytes, reads `archive_dir/0.bin`, iterates file list,
  rejects unsafe file-list entries, and calls the vendored SNeRV inflate path on
  CPU. No scorer imports were visible in the wrapper.
- `snerv_upstream_submission_bundle.json` keeps `score_claim=false`,
  `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`; its
  blockers include full-video scorer replay, paired CPU/CUDA auth eval, and
  pre-submission compliance gate.
- `snerv_upstream_submission_runtime_audit.json` adds runtime-closure blockers:
  minimal runtime closure not materialized, pruned dependency closure not
  proven, source minification not materialized, static import closure missing
  members, and runtime source receiver proof missing/failed.

SNAR2 direction is compliance-safe: `src/tac/substrates/snerv_inverse_steg_carrier/archive.py:833-890`
packs a fixed binary header with fixed section order, compact integer metadata,
section lengths, and short section hash prefixes. Tests
`src/tac/substrates/snerv_inverse_steg_carrier/tests/test_archive.py:148-190`
and `src/tac/tests/test_snerv_snar_header_minimizer.py:406-430` prove the SNAR2
header omits human-readable section names and metadata strings.

There is still inner-packet byte headroom. A raw byte scan of `0.bin` found
`SNSA2` in the step-map section as intended, but also found strings such as
`schema` and `decoder_payload` in the packet body. Those body strings are
charged bytes, so removing them is a valid optimization only by changing the
corresponding receiver-visible inner payload grammar and proving decode/replay
parity. They must not be hidden in `inflate.py`.

## Safe byte-shaving moves

1. Keep SNAR2/SNSA2 as the default packet path.
   The outer SNAR2 header is a fixed binary receiver contract and already
   removes the largest human-readable outer grammar. Continue using reports for
   provenance, not packet labels.

2. Replace inner decoder JSON/microheaders with compact binary grammars.
   This is safe when the grammar is in charged `archive.zip` bytes, the runtime
   parser is updated, section SHA/byte accounting is preserved in side reports,
   and receiver/full-video replay proves identical intended decode behavior.

3. Shorten ZIP member names only with runtime proof.
   Upstream does not require `0.bin`; the runtime does. Moving from `0.bin` to a
   shorter member such as `x` is a tiny safe win only if `inflate.py`, receiver
   proof, archive manifest, and local/central ZIP header checks all agree.

4. Continue deterministic single-member archive selection.
   Compare stored vs deflated ZIP forms for each already-compressed packet,
   using deterministic timestamps/permissions and fixed member order. Do not
   assume deflate wins; the current SNeRV packet did win under deflate by about
   2 KB, but other entropy-coded payloads may expand.

5. Move only generic decoder logic into `inflate.py`.
   Generic parsers, fixed enum tables, bit unpackers, ANS/range/Huffman decode
   routines, shape formulas, and tiny implementation constants can live in
   runtime code when they are not per-video score-bearing information, do not
   load scorers, have no side effects, stay reviewable, and are runtime-tree
   hashed.

6. Prefer `weight_derived` over `runtime_constant` for clever generation.
   If a seed, table, or codebook can be derived from existing charged bytes,
   record the source member SHA and mutation/no-new-bytes proof. This preserves
   byte wins without laundering payload through runtime source.

7. Keep human-readable provenance outside charged packet bytes.
   Reports, manifests, and compliance ledgers may remain readable. They are not
   part of `archive.zip` unless intentionally bundled. The charged packet should
   use fixed enums/offsets/lengths/hashes instead.

## Unsafe byte-shaving moves

1. Do not move per-video labels, seeds, lookup tables, trained weights,
   generated code blobs, distilled transducers, or scorer-derived payload into
   `inflate.py` merely because upstream charges only `archive.zip`. Local
   authority classifies that as payload relocation unless an explicit ruling and
   non-payload proof exists.

2. Do not load PoseNet, SegNet, `evaluate.py`, `DistortionNet`, scorer
   checkpoints, or scorer loss helpers during inflate unless the large artifacts
   are included in `archive.zip` and counted. The local stricter path is
   scorer-free inflate.

3. Do not read original video, host-local files, external sidecars, hidden
   files, network downloads, package installs, cloned repos, or undeclared
   runtime dependency roots at inflate time.

4. Do not rely on malformed ZIP behavior, duplicate member names,
   local/central header mismatch, resource forks, hidden members, parent
   traversal, absolute paths, or parser-divergent extraction order.

5. Do not make byte or score claims from packet parser parity alone. Symbol,
   tensor, or latent parity is parser-consumption evidence only until
   `inflate.sh archive_dir output_dir file_list` output parity or exact same
   runtime eval exists.

6. Do not treat minified or generated runtime source as automatically safer.
   It is only safer when it removes review noise or generic code bytes. Opaque
   literal arrays or candidate-specific encoded strings inside source are
   score-bearing until proven otherwise.

## Recommended next automation

1. Add a byte-string census step to SNeRV/HiNeRV packet reports:
   locate ASCII runs inside charged archive members, classify them by section,
   and emit candidate work orders for inner grammar compaction.

2. Build the next SNeRV inner grammar as `archive_seeded` or `weight_derived`:
   fixed enum table, compact section table, binary decoder microheaders, and
   provenance-only readable names outside the packet.

3. Wire runtime minification into the existing runtime audit:
   minify only generic code; keep a runtime tree hash and a source-to-minified
   manifest; refuse if the minified runtime introduces scorer imports, side
   effects, or candidate-specific payload literals.

4. Promote only through the canonical sequence:
   bundle/lint -> compliance gate -> receiver/full-video replay -> local CPU
   replay gate -> paired exact CPU/CUDA auth eval on the exact same archive.

## Solver hook accounting

This is a research-only compliance memo, not a code landing.

- Sensitivity-map contribution: N/A. No byte mutation or scorer-response anchor
  was produced.
- Pareto constraint: N/A. The output is a compliance constraint list for future
  candidate builders.
- Bit-allocator hook: N/A for this memo. Recommended follow-up is the
  byte-string census work order.
- Cathedral autopilot dispatch hook: N/A. No dispatchable candidate was emitted.
- Continual-learning posterior update: N/A in this scoped worker landing.
- Probe-disambiguator: N/A. The safe/unsafe split uses existing local authority:
  `archive_seeded` and `weight_derived` are promotion paths;
  `runtime_constant` is research/probe unless explicitly ruled non-payload.
