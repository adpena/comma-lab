# ddm_mrs6 — the reader's pass: one implementation, no residue, a format spec, and a README a stranger can use (2026-09-16)

<!-- # FORMALIZATION_PENDING: arm memo. The objects here are a packet and an identity proof, not a
law. No canonical equation is registered: nothing new was derived, and the one measurement that
crosses a device boundary (the decoded token plane) is already registered upstream. -->

**Axis of everything measured locally in this memo: `[macOS-CPU advisory]`, `score_claim=false`,
`promotable=false`, `authority=false`. The exact pointer did not move and this arm never fires.**

---

## 1. What this arm was for

A fresh reader with no context refused to trigger an evaluation of `submissions/mrs5/`
(`.omx/research/ddm_mrs5_20260916/FRESH_READER_VERDICT_mrs5.md`, sha `ebcc5838d9795eed…`). The
packet was already byte-identical to move 53 on all 600 pairs; what failed was everything a
maintainer sees. This arm answers the reader's eleven items and re-proves the identity afterwards.
`submissions/mrs6/` is a copy of `submissions/mrs5/`; mrs1–mrs5 are untouched.

## 2. The packet — eight files (MEASURED)

| file | LOC before | LOC after | sha256 |
|---|---:|---:|---|
| `inflate.sh` | 8 | 10 | `06b1c9821d753efcdc68679c1f26eebb6e023b9118cdea7780ee76027b37265e` |
| `inflate.py` | 2,891 | 2,080 | `3f09d597f0c3ee15e0a771808a6ebb4e08bb64c19822299d7dccb74e7a954382` |
| `corrector.c` | 693 | 693 | `71f632bc59893cb33673e44f8baa9bac588586463ab057f917fc0fa16b5bed77` |
| `geometry.c` | 145 | 145 | `ac99ab49efbc9fce9bab177b70a62b8e32b576caea16a03d78326b61bf9a0e63` |
| `range_decoder.c` | 100 | 100 | `d70a494987ff9ea720b9150a5baf3343299139c6ce8c090c388974d03c14baec` |
| `README.md` | 28 | 40 | `6644cb6bcd3bfef479ad3c43bb1a238625de3379459d8e5b04af3bef806e0701` |
| `FORMAT.md` | — | 80 | `df92248e9bd593c7a227f4f651430ef12d92274557094b529c1f90bc8854f933` |
| `archive.zip` | — | — | `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957` (179,286 B) |

`inflate.py` lost **811 lines, 28.1%**. The three C files are **byte-identical to mrs5's**, so
mrs5's payload-cleanliness audit (`INDEPENDENT_PAYLOAD_AUDIT.json`, `LITERAL_CENSUS.json`,
`GEOMETRY_SOURCE_AUDIT.md`) transfers without re-derivation, and so does the T4 timing lineage.

## 3. What was removed, and what proves the removal was safe

Removed, in one pass: the Python range decoder's decode loop (47 lines), `CausalGeometry` and
`preceding_window` (91 lines), the whole Python corrector — `counts_*`, `odds_*`, `context_*`,
`miss_*` (619 lines) — `make_corrector`'s fallback branch, and the 13-constant plus
`miss_SHIPPED_CONFIG` comparison that stood in for an equivalence test. All three library loaders
now raise `SystemExit` with one line instead of printing and returning `None`; `inflate.sh` checks
for `cc`, drops every `|| true`, and stops on a failed build.

**Three more second implementations were found only because I looked past the reader's own list**,
and they matter: `log2_fixed` and `mix_probabilities` each carried a full NumPy implementation
behind `if _CORRECTOR_LIBRARY is not None`, unreachable once the loader stopped returning `None`
but still there to be read as an alternative; and `CabacRangeDecoder` was **byte-for-byte the same
class as `RowRangeDecoder`** under a second name. The two decoders are now one `ByteRangeDecoder`,
`BitRangeDecoder` inherits its register file instead of copying it, and both NumPy branches are
gone. The reader would have found these; the honest reading of item 4 is that it named two
implementations and there were five.

Removing a second implementation is safe only if the first one is the one that runs. Three
measurements say it is: the shell stage `no_compiler` (rc 1, no library survives the refused
build), the shell stage `help` (all three `.so` present, no fallback text on any stream), and the
600/600 identity in §5. **The four unreadable literals the reader named (lines ~1625, 1789, 1794,
1903) were the deleted mixer registries** — they are gone rather than reformatted.

## 4. The rename, and how I know it changed nothing

193 names carrying eighteen module-name prefixes (`archive_`, `render_`, `shared_`,
`geometry_mixer_`, `prior_io_`, …) became names chosen for a reader, with duplicates merged onto
one name each: three copies of `12` onto `CARRIER_DIM`, four of `600` onto `PAIRS`, four of `5`
onto `CLASSES`, two identical `frozenset`s onto `ROW_PRUNE_NAMES`, two identical `struct.Struct`s
onto `SELECTOR_HEADER`, and a `parse_geometry_config = parse_geometry_config` alias deleted.
The map refuses to run if any new name is already in use or if any function shadows a merge target.

**MEASURED, and this is the check that makes the rename reviewable:** re-applying the map to the
pre-rename file and diffing against the shipped file yields **42 hunks, every one an edit made
deliberately afterwards** — the constant block, the pixel-mode table, `decode_report`, four
docstrings, the CUDA comment, `self._sparse_cache = None`, `import json`, the merged duplicates,
the deleted NumPy branches, and the residue below. **The rename itself introduced zero unintended
differences**, which is the only reason a 193-name rename is reviewable at all.

Also removed as residue: four methods monkeypatched onto a live object with `MethodType`, and with
them a latent bug — `prior_requantize` had **no module-level definition at all** and existed only
because `inference_optimize_sparse_evaluator` created it through `global`; `np.testing.assert_array_equal`
inside the decode loop; `compensation_blob` and the `overlay_split_selector_compensation` that
returned `None` for a non-empty overlay while both callers unpacked two values; the discarded
`padding_mask`; the `brotli is not None` and `model_sections is not None` guards that could not be
false; four write-only dataclass fields; and eight dead local assignments.

Three more pieces of residue that a linter finds and a reader would not forgive: `take_bytes` took
a `label` it never used — it now names the section in the error when a stream is short;
`walk_prior_bits` took a `learning_shift` it never read, and `restore_prior_stream` read a `rate`
byte to pass to it, so both are gone (the offset arithmetic is unchanged); `mix_probabilities` took
an `original_rows` neither branch used. Three loop variables that nothing read were removed with
their `enumerate`/`reversed` wrappers, which iterate the same number of times.

`prior_ste_round` was a straight-through estimator, `value + (value.round() - value).detach()`,
that `inference_optimize_sparse_evaluator` replaced at runtime with `value.round()`. It is now
defined once as `value.round()`. **MEASURED: the two forms agree on 0 of 28,000,000 float32
samples — no mismatches** across seven magnitude scales and a dense sweep, which is what Sterbenz's
lemma predicts. The 600/600 identity is the binding proof.

## 4a. Checking the spec against the code, the way the fresh reader will

`FORMAT.md` is only worth writing if its claims hold, so I checked them: the 14-byte header and its
values (`RX1M`, version 1, codec 2, flags 250), the five flag bits the code tests and the three it
does not, the four section lengths (11,629 / 29,862 / 18,483 / 119,198) and their inflated sizes,
the 96-byte boundary table's arithmetic (25 x 5 x 6 bits + one fp16 scale), the 64-byte
geometry-mixer split (1 variant nibble + 40 int8 weights + 19 configuration bytes), the nine
geometry bins, the 23 families and 4,000 weight sets in `corrector.c`, the frequency construction
in `range_decoder.c` (truncate, floor at 1, first maximum absorbs the slack), and the raw's
1,200 x 874 x 1,164 x 3 = 3,662,409,600 bytes.

**One claim was false and the check caught it.** I had written that the group order alone makes a
group's neighbours already decoded. **MEASURED: it does not** — at a 64-pixel patch edge the left
neighbour's group index is 63 against the current pixel's 0, so it is *later*, not earlier. What
actually holds causality is a flag: `corrector.c` keeps a `known` bit per pixel and reads a
neighbour only when it is set. `FORMAT.md` now says that. Two smaller corrections came out of the
same pass: the boundary state is a pair `(boundary, predicted)`, not a product, and the 29 stored
selector bytes are a body whose 5-byte header the decoder supplies as a constant.

Also MEASURED, since the spec asserts the partition: the decoder's own `group_masks` equals
`column % 64 + 2 * (row % 64)` on **190 of 190 groups** and covers all 196,608 pixels exactly once.

## 5. The proofs (MEASURED)

**Cold n600 public parse-back, byte-identical to move 53.** One run of
`experiments/ddm_mrs6_proof.py` drove the eight shipped files through their own `inflate.sh` in a
bare venv, decoded all 600 pairs cold, and compared the output at two granularities:

- **Full file: `8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b`**, 3,662,409,600 bytes —
  equal to move 53's raw in pd6's `parseback2/PARSEBACK_RESULT.json`
  (`rendered_raw.sha256` and `inflate_report.raw_sha256`).
- **Per pair: 600/600 equal**, differing pairs `[]`,
  against the 600-element `pair_sha256` list in mrs1's `public_smoke/IDENTITY.json`.
- **Decoded token plane: `4cb0b147bdae8ce0618938453bb6ccc5b7b5a4927d6ad4e1b0bd9dd9ff46119c`** — the digest that crosses the device
  boundary, equal to move 53's. Coded bit position at the end: 952,362.

Cold decode wall clock 774.6 s on this loaded host. **That number is not timing
evidence** — pd7 and ffi6 shared the machine throughout; it is run metadata.

**The shell is the only path, and it fails loudly.** In the same run, on the same copied tree:

| stage | rc | what it proves |
|---|---:|---|
| `no_compiler` | 1 | `PATH` without `cc`: one line, `inflate.sh: no C compiler on PATH`, and no `.so` survives the refused build |
| `help` | 0 | with `cc`: all three libraries built in 2.2 s, and no fallback text on either stream |
| `default_guard` | 1 | no `--device`: `CUDA is required by default`, nothing written |
| `public_decode` | 0 | the cold n600 decode above, no fallback text |

**The frozen contract accepts the new report line.** `tac.decode_wall_clock._cold_public_report`
parsed it out of the real decode's stdout inside the run, and
`tac.candidate_seal._pf_cold_work_facts` reads all five work-bearing fields from it here:
archive `aab908d3b32c6558…` at 179,286 B, raw
`8a14f55a6a8b1415…`, token plane `4cb0b147bdae8ce0…`, bit position
952,362.

**Compiler and linter.** `cc -O2 -std=c11 -Wall -Wextra -ffp-contract=off -fno-fast-math` on all
three C files: **0 warnings, rc 0** (Apple clang 21.0.0). `ruff check --select F,ARG` on
`inflate.py` and the instrument: clean. Receipts in
`.omx/research/ddm_mrs6_20260916/ENGINEERING_PROOFS.json`.

**Environment and storage.** Bare APFS venv at `.omx/tmp/ddm_mrs6_bare_venv`, built offline from
mrs5's retained wheels, system site packages off: brotli 1.2.0, numpy 1.26.4, torch 2.12.1. The receipt tier
(`/Volumes/APDataStore/pact/ddm_mrs6`) held 12.4 GiB
free at preflight, above the 8 GiB floor; the 3.42 GiB raw went to local APFS scratch, which had
230.7 GiB, because the shared SSD tier could not hold it
and stay above the floor. The raw was deleted after both comparisons passed, with its facts recorded
first in `RAW_CLEANUP.json`.

Certificate: `/Volumes/APDataStore/pact/ddm_mrs6/proof/IDENTITY.json`, sha `9dd39f24719c98aadf53b8089fd0843665937b29fa2c86afbdb88668d48463d3`.

## 6. The two additions MAIN made mid-flight

1. **The T4 numbers are MAIN's, and the README states them as measured.** On a Tesla T4, n600:
   S 0.1361014714463198 — identical to move 53 to the last digit — inflate 1,045.824 s, evaluate
   44.288 s (`.omx/research/ddm_mrs5_20260916/t4_remeasure/`). **verdict_scope: that run measured
   mrs5's seven files. The three C files are byte-identical here, and this arm's 600/600 identity
   shows the Python side produces the same output, so the number transfers to the same decode path
   — but it is not a measurement of mrs6's own bytes, and the README does not claim it is a fresh
   one.**
2. **The decoder now prints one JSON summary line**, because the contract's decode-leg builder
   refused to mint a `t4_direct` leg for mrs5: "cold receiver report absent from the T4 stdout log".
   `decode_report` emits exactly the fields `tac.decode_wall_clock._cold_public_report` and
   `tac.candidate_seal._pf_cold_work_facts` read: `pair_count`, `checkpoint_resume`,
   `token_cache.status`, `archive_sha256`, `archive_bytes`, `raw_sha256`, and a `token_decoder`
   section with `checkpoint_resumed_from_frame`, `decoded_token_sha256` and `decoder_bit_position`.
   Every field is honest generic output; nothing is fabricated to satisfy the reader. **MEASURED:
   the frozen `_cold_public_report` accepts the line** — first in a standalone smoke against a log
   with surrounding noise, then inside the cold n600 run itself.

## 7. What I did not do, and why

Three exception classes (`ModelCodecError`, `WeightFormatError`, `SelectorError`) have identical
empty bodies. They are three distinct error types, not one thing under three names, so they stay.
`BitRangeDecoder` still declares its own `decode_bit`; only the shared register file was folded in.

The reader's item 2 also named `torch.backends.cudnn.deterministic = False` and
`torch.use_deterministic_algorithms(False)`. Both calls restate PyTorch's own defaults, so deleting
them would be behaviour-neutral in a fresh interpreter — but they sit on the CUDA path, which a
local CPU identity run cannot cover, and MAIN's T4 measurement was taken with them present. I
explained the block in a comment and in the README instead of changing it. **verdict_scope: this is
a judgement about what a CPU proof can and cannot certify, not a claim that the two lines matter.**

## 8. Prior negatives accounted

PR #140 (unreviewable — this pass is the cure, measured by a reader rather than asserted); mrs1–mrs5
(each measured one thing; this one measured what a maintainer sees); pd6's rider-drop law (the
archive was copied, never re-staged, and its sha is checked at three points); the ExFAT venv (the
bare venv is on APFS under `.omx/tmp/`); the charter-time optimal-form law (no toy: the identity is
the full cold n600 public decode, not a prefix).

---

**composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53) — unchanged by this arm.**
