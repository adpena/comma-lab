# ddm_mrs6 — the READER'S PASS on the minimal packet: one implementation, no residue, a format spec, a README for a stranger (charter, MAIN 2026-09-16; Opus; scorer-free)

## Why
A fresh reader with no context (`.omx/research/ddm_mrs5_20260916/FRESH_READER_VERDICT_mrs5.md` — read it first, every
line is an instruction) refused to trigger eval on `submissions/mrs5/` as-is. The packet is byte-identical to move 53
on all 600 pairs (mrs5 memo) and its T4 timing is being measured now (call fc-01M2NVF2NAPPTJG8CXSAAS2QT5; MAIN's). What
remains is what a maintainer sees. Every item below comes from the reader's list; nothing else is in scope.

## Deliverable — `submissions/mrs6/` (start from a COPY of `submissions/mrs5/`; mrs1–mrs5 stay as landed)
1. **One implementation.** Require the C compiler (inflate.sh exits with one clear line if `cc` is missing or a build
   fails — no `|| true`); DELETE the Python fallbacks for the range decoder, corrector and geometry, and the
   13-constant config comparison that stood in for an equivalence test. The three C files are the implementation;
   inflate.py calls them. (The contest environment has a compiler; PR103's compiled coder is the precedent.)
2. **No residue.** Remove every half-removed feature the reader named (the compensation/overlay path, `compensation_blob`,
   dead guards, discarded `padding_mask`), the monkeypatching (2164–2167 → plain methods), the in-loop
   `np.testing.assert_array_equal`, and the flattening artifacts (`render_render_video`, `miss_miss_cells`,
   `basis_basis_contexts`, `inference_inference_round`, module-name prefixes) — one name per thing, chosen for a reader.
   If a removed path is reachable on the C path for THIS archive, identity will catch it: re-prove.
3. **The format spec.** `FORMAT.md` ≤ 80 lines: the three sections of `p` and their header (every flag bit the code
   tests: the "mixer rider" bit, PRIOR_ADAPTIVE, RENDERER_PLANES, `config[4] & 15`), the prior/renderer/carrier
   objects and their quantization, the carrier basis + CABAC/AR(1) coefficients + selector, the 23 context families
   and the 25-state boundary table, the arithmetic coder's parameters, the pair layout of `0.raw`. Then turn the four
   unreadable literals (lines ~1625/1789/1794/1903) into named, commented structures that FORMAT.md refers to.
4. **README for a stranger** (≤ 40 lines): what the archive holds, how decoding works in three sentences, the exact
   commands, dependencies (numpy, torch with CUDA, brotli, a C compiler), the file-list expectation (the public list is
   exactly `0.mkv`; say so plainly and why the decoder checks it), determinism as MEASURED (the T4 decode of these
   bytes reproduced move 53's raws; say what is and is not deterministic and on which device — take the facts from
   MAIN's T4 harvest when available, else state "measured by MAIN before submission"), the archive sha/size, and ONE
   sentence on why three small C files ship. No jargon: no "sealed", "receiver", "rider", "lane", "reviewer budget",
   "native set", "counted".
5. **Proofs**: full cold n600 public parse-back → raws byte-identical to move 53 (600/600; compare the full-file sha to
   pd6's `parseback2/PARSEBACK_RESULT.json`); bare-venv smoke with the compiler present (the only supported path) in an
   APFS scratch venv; `cc -std=c11 -Wall -Wextra` clean on all three C files; `ruff` clean on inflate.py; a
   fresh-reader prompt for MAIN (MAIN spawns the second reader). PR body draft ≤ 20 lines (carry mrs1's).
6. Budget: ≤ 8 files (inflate.sh, inflate.py, corrector.c, geometry.c, range_decoder.c, README.md, FORMAT.md,
   archive.zip unchanged 179,286 B sha aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957); report
   inflate.py's LOC before/after (the fallbacks and residue should take it well below 2,891).

## Boundaries
No Modal, fire, packet, PR, push, authorize_*; upstream/, the closed PR tree, sealed trees, mrs1–mrs5, contract code
read-only; work under `/Volumes/APDataStore/pact/ddm_mrs6/`; venvs on APFS (`.omx/tmp/`); retain ≤ 2 GiB with sha256;
heavy steps via `tools/launch_detached_process.py --done-receipt <bare-name>`; do not touch ddm_pd7/ or ddm_mrs5_fire/.
You can write git objects: serializer commits with post-edit shas, two review passes per .py, `[no-triality]
[p0-ledger-ok]`; `archive.zip` is gitignored (commit the other seven). Never a co-author trailer or AI attribution.
Checkpoint `ddm_mrs6`; lane `ddm_mrs6_readers_pass_20260916`. Label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
Reference forms: mrs5's packet (behaviour on the C path) and the fresh reader's list (the acceptance test). Declared
deltas: removal of the Python paths and residue (behaviour on the C path unchanged, proven by identity), the format
spec and README (documentation). Provenance pins: mrs5 packet shas (inflate.py f79f1e3f0ef897df, corrector.c
71f632bc59893cb3, geometry.c ac99ab49efbc9fce, range_decoder.c d70a494987ff9ea7); reader verdict file sha ebcc5838d9795eed;
move 53 packet a91a7dde3.

## Prior negatives accounted (operator 2026-08-15)
PR #140 (unreviewable); mrs1–mrs5 (each pass measured one thing; this pass is the reader's, measured by a reader);
pd6's rider-drop law (the archive is copied, never re-staged); the ExFAT venv.

Final message: the file list with LOC before/after, each of the reader's 11 items with what you did, 600/600 raws,
the smoke, the fresh-reader prompt path, the PR draft path, every boundary, the serializer commit shas, ending with
`composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)`.

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
