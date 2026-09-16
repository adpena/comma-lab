# ddm_mrs7 — the final polish: the second reader's items closed, and the identity re-proved (2026-09-16)

<!-- # FORMALIZATION_PENDING: arm memo. The objects here are a packet and an identity proof, not a
law. No canonical equation is registered: nothing new was derived, and the one measurement that
crosses a device boundary (the decoded token plane) is already registered upstream. -->

**Axis of everything measured locally in this memo: `[macOS-CPU advisory]`, `score_claim=false`,
`promotable=false`, `authority=false`. The exact pointer did not move and this arm never fires.**

---

## 1. What this arm was for

A second fresh reader with no context said **YES to `submissions/mrs6/` as-is**
(`.omx/research/ddm_mrs6_20260916/FRESH_READER_VERDICT_mrs6.md`) and then listed what still made
them hesitate. This arm closes that list. `submissions/mrs7/` is a copy of `submissions/mrs6/`
with the reader's items fixed and nothing else; `archive.zip` is untouched, and mrs1–mrs6 are
untouched. **Every code change is behaviour-preserving on this archive, and the proof is the same
one mrs6 used: a cold n600 public decode whose output is byte-identical to move 53.**

## 2. The packet — eight files (MEASURED)

| file | LOC mrs6 | LOC mrs7 | sha256 (mrs7) |
|---|---:|---:|---|
| `inflate.sh` | 10 | 10 | `a4089a1a18a6e40e1ea241f8a3522a3576e44f0a469f9658fc9eb0d675a8bf6b` |
| `inflate.py` | 2,080 | 2,180 | `db48036fc33465627f9b1288eff05ee4b50f9b831a158cd4c7e9379d84b6b7ab` |
| `corrector.c` | 693 | 715 | `ce98ed5d731d0cc83e1c1ffb9cb42d66ac5a0e4f882ae5a3cd1da7a457f7bc34` |
| `geometry.c` | 145 | 165 | `63dc7d7df810712389bfd5f287eabf3b0b1915db3d7f1033125f6f9e13c0ec22` |
| `range_decoder.c` | 100 | 100 | `d70a494987ff9ea720b9150a5baf3343299139c6ce8c090c388974d03c14baec` |
| `README.md` | 40 | 57 | `3d596c938bd1d24ff609065d575da03f5fa53e63463262c24b476781b28d8cbe` |
| `FORMAT.md` | 80 | 101 | `66a005316e34b66394af6cec108458d8319217ab37cebb5d65df824c1407c4e1` |
| `archive.zip` | — | — | `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957` (179,286 B, unchanged) |

`range_decoder.c` is **byte-identical to mrs6's**. mrs6 cut 811 lines out of `inflate.py`; mrs7
puts 100 back on net, and every one of them is a comment, a named constant or a header check.
No decoding logic was added, and the arithmetic in all three C files is untouched.

## 3. The reader's items, closed

**Documentation.**
- `FORMAT.md` §1 claimed the prior stores "int8 weight codes with one fp16 scale per row". It does
  not. MEASURED at the archive: the prior's coded weights carry **one bit depth per output channel
  packed two to a byte** (`unpack_nibbles`, 0–15 bits), then that many two's-complement bits per
  weight; every other parameter is a plain `<i2` (biases) or `i1` array, which is where the
  per-channel power-of-two exponents live. **No fp16 is in that stream at all** — the sentence had
  drifted over from the renderer stream. Rewritten, and the renderer sentence now says so.
- `FORMAT.md` claimed 4-neighbour and 8-neighbour spatial contexts. MEASURED in `corrector.c`:
  `N_CAUSAL` is 4, `spatial` reads **two** neighbours (left, above) and `spatial4` reads **four**
  (`CAUSAL_DX`/`CAUSAL_DY`: left, above, above-right, above-left). No eight-neighbour context
  exists. Rewritten, with the neighbour lists named.
- "11,629 bytes stored, 16,249 after Brotli" read backwards. MEASURED by decompressing the member:
  11,629 is the **stored** length and 16,249 the **decompressed** one. All three sections now read
  "N bytes as stored, M bytes once Brotli-decompressed", with M measured.
- The 16 unread carrier bytes now have their own `FORMAT.md` section. **MEASURED: offsets 123–138
  are 16 zero bytes**, and the new table lists every offset `decode_carrier` does read beside the
  one range it does not. Header flag bit 32 is now stated as "present and unread, so its value
  cannot change what the decoder produces", in both `FORMAT.md` and `HeaderFlags`.
- `README.md` gained the paragraph the reader asked for: what frame 0 of each pair **is** (a smooth
  field centred on mid-grey — twelve coefficients weighting twelve stored three-channel 24 x 32
  basis images, scaled `127.5 + 64 x carrier`, enlarged 24x32 → 384x512 → 874x1164), **why** (the
  scorer reads each pair as a pair; one of its two measurements is the camera motion between the
  two frames, so the first frame's whole job is to carry that signal), and what the seven
  whole-frame pixel edits are for (a three-bit label, MEASURED on **24 of the 600 pairs** in this
  archive, positioning that field for the motion measurement). Also added: the JSON summary line
  and what is in it; NumPy and PyTorch assumed present in the evaluator with a missing one
  surfacing as Python's own `ImportError`; **the output needs about 3.5 GiB free**.

**Code (each change re-proved by the identity run).**
- `split_payload_sections` named byte 7 `reserved` while using it as the flag byte, and named byte 6
  `table_mode` when byte 6 is the reserved one. Swapped to `reserved` / `flags`, matching
  `FORMAT.md`, with a comment saying so.
- `PriorPredictors.__init__` now refuses any family but 1 with a named `ValueError` instead of
  reaching an `UnboundLocalError` inside `predict`, and the unreachable `family == 2` branches in
  `predict` and `update` are gone.
- The dead xz branch and `import lzma` are gone. The `codec` byte did not become unread in the
  process: the decoder now **refuses** any codec but Brotli (2) rather than silently treating it as
  Brotli, so the field is live and checked.
- `validate_rice_parameters` now validates: exactly `CARRIER_DIM` parameters, each in [0, 16],
  because the coded residuals are zigzagged 12-bit values and `quotient << k` writes into an int32.
  MEASURED in this archive: 12 parameters, values 4 and 5.
- `packed_length` no longer calls `signed_bounds` for its return value; `signed_bounds` had no other
  caller and is deleted. `IntegerPrior` no longer takes `norm_mode` or `use_norm_gates`, and neither
  does its one call site. `unpack_renderer_weights` raises on a bad magic instead of returning
  `None`. `ByteRangeDecoder._byte` raises a named error past the end of the stream instead of
  returning `None` — which, being `None`, crashed one line later with a `TypeError` anyway.
- Four more never-validated tags are now checked, which is what turns the unused unpacked fields
  into something a reader can follow: the payload header's magic and version, the renderer mixer's
  magic, version and weight count (`MIXER_WEIGHTS`, now one constant for both mixers, replacing two
  hardcoded 24s), the prior mixer's magic, version, weight count, row count and total length, and
  the prior weight blob's own `IHS1`. The one byte the prior stream skips is now named in a comment
  as unread rather than silently stepped over.
- Both `bytes.fromhex('')` are `b''`, and the five hex-string magics are byte literals: `b'SM3R'`,
  `b'IHS1'`, `b'F0E1'`, `b'RCF1'`, `b'F0E1\x01'`. MEASURED: mrs6 had 12 bare strings standing in for
  docstrings after a constant; mrs7 has none, all 12 are comments.
- Comments added to every literal the reader named — `inflate.py` at the geometry domain check, the
  quartile arithmetic in `scale_buckets`, the `kind`/`block` chains, the 23-key context list, both
  adaptation-rate tables, and the `IntegerPrior` construction; `corrector.c`'s four parallel family
  tables (what each is, and why five families repeat an earlier rule: a count cap makes them
  forget); `geometry.c`'s two-limb arithmetic (what each of the nine operations does, and why a
  128-bit integer is written out rather than borrowed from a compiler extension).
- Jargon: "the Lane mixer" is gone, replaced by a comment saying what `mixer_log2` computes.
  `SLOT_WIDTH` is `PATCH_SIZE`. `RULE_SHIPPED` is `RULE_JOINT` and `FAMILY_IS_SHIPPED_JOINT` is
  `FAMILY_MIXER_STARTS_AT_ONE`, which is what it selects. "semantic" is gone: the renderer stream,
  blob, bytes and model are all called `renderer`, the name the documentation uses.
- The seeds and the two determinism calls are **kept exactly as they were**, because they were
  present in the measured T4 run. Each now carries one comment saying why it is there.
- `inflate.sh` links `range_decoder.c` with `-lm`; it includes `math.h` for `isfinite`.
- **No Python fallback was added, and no C arithmetic was changed.**

## 4. One item I did not do, and why

The reader called `checkpoint_resume`, `token_cache` and `checkpoint_resumed_from_frame` leftovers
in `decode_report`. **They are not droppable: all three are read.** MEASURED by reading the frozen
contract — `tac.decode_wall_clock._cold_public_report` finds the report line by the literal
`"checkpoint_resume"`, then requires `pair_count == 600`, `checkpoint_resume is False`,
`token_decoder.checkpoint_resumed_from_frame == 0` and `token_cache.status == "DISABLED"`; and
`tac.candidate_seal._pf_cold_work_facts` reads `archive_sha256`, `archive_bytes`, `raw_sha256`,
`token_decoder.decoded_token_sha256` and `token_decoder.decoder_bit_position`. Dropping any of them
would break the evaluation harness's proof that this was one cold run over all 600 pairs. They are
constants here because this decoder has neither a checkpoint nor a cache — which is exactly what
they report. The docstring now says this, so the next reader does not have to rediscover it.

## 5. The proofs (MEASURED)

**Cold n600 public parse-back, byte-identical to move 53.** One run of
`experiments/ddm_mrs7_proof.py` (mrs6's instrument, copied and repointed; mrs6's files untouched)
drove the eight shipped files through their own `inflate.sh` in a bare venv, decoded all 600 pairs
cold, and compared at two granularities against pd6's
`parseback2/PARSEBACK_RESULT.json` and mrs1's `public_smoke/IDENTITY.json`:

- **Full file: `8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b`**, 3,662,409,600 bytes.
- **Per pair: 600/600 equal**, differing pairs ``[]``.
- **Decoded token plane: `4cb0b147bdae8ce0618938453bb6ccc5b7b5a4927d6ad4e1b0bd9dd9ff46119c`**; coded bit position at the end `952,362`.

Cold decode wall clock 842.2 s on this loaded host. **That number is not timing evidence**;
it is run metadata.

**A confound caught before it produced a number.** I restarted this proof three times, because each
time I found a documentation or comment error while it ran and the instrument binds the file shas at
run start. Stopping it the first two times, I killed the instrument — and the `inflate.py` the
instrument had started through `inflate.sh` was **not** in that kill: `/bin/sh` had already exited,
so the decoder was reparented to init and kept running. MEASURED: at the moment I looked, **four
decoders were alive at once**, three of them orphans from killed runs, **all four writing the same
`0.raw` through a `mode='w+'` memmap**, and the oldest two running the pre-final `inflate.py`. No
certificate had been written yet, so no false result was ever recorded — but the run in flight was
contaminated and would have compared a raw that four processes had interleaved. Cure applied: kill
the decoder by its own command line, not just the instrument; delete the raw and the run directory;
confirm by `ps` that zero decoders survive; relaunch. The run reported below is that clean
relaunch, and its watcher stops the moment a second decoder appears. **The lesson is general: `pkill` on a
launcher does not reach a grandchild that has been reparented, and a `w+` memmap gives concurrent
writers no error at all — just a wrong file.**

**And the guard I wrote against it could not fire.** The relaunch watcher counted decoders with
`pgrep -fc`. **MEASURED: macOS `pgrep` has no `-c`** — it prints usage and exits 2, so the
`|| echo 0` I had added for safety pinned every check at zero. A second trap sat in the same line:
`pgrep -f` matches the watcher's own shell, because the pattern appears in its command line. Both
were found by running the expression by hand instead of trusting its silence. The working form is
`ps -axo args | grep -c "^python3 .*<run_dir>/public/inflate.py"`, anchored on the interpreter so it
cannot match itself. Both findings are durable in
`memory/pkill_on_the_launcher_leaves_orphan_decoders_writing_the_same_memmap_20260916.md`.

**The shell is the only path, and it fails loudly.** Same run, same copied tree:

| stage | rc | what it proves |
|---|---:|---|
| `no_compiler` | 1 | `PATH` without `cc`: one line, `inflate.sh: no C compiler on PATH`, and no `.so` survives the refused build |
| `help` | 0 | with `cc`: all three libraries built, and no fallback text on either stream |
| `default_guard` | 1 | no `--device`: `CUDA is required by default`, nothing written |
| `public_decode` | 0 | the cold n600 decode above, no fallback text |

**A second, narrower measurement that isolates the edits.** Before the full run, the whole
model-loading path — every validator, every renamed field, the Rice check, the refused codec, the
raising `_byte`, `PriorPredictors`, `load_prior_weights` — was run under mrs6's code and mrs7's code
on the same archive and the digests compared: renderer state dict, prior state dict, carrier basis,
carrier coefficients, selector choices, token section, boundary table and geometry-mixer parameters
were **identical on all eight**. That is the narrow A/B; the 600/600 identity is the binding proof.

**The frozen contract accepts the report line.** `tac.decode_wall_clock._cold_public_report` parsed
it out of the real decode's stdout inside the run, and again standalone afterwards through
`experiments/ddm_mrs7_report_parse_check.py`, which surrounds the line with noise so the search is
the one the harness performs. Both readers were satisfied: `pair_count` 600, `checkpoint_resume`
false, `token_cache.status` DISABLED, `checkpoint_resumed_from_frame` 0, and
`tac.candidate_seal._pf_cold_work_facts` read archive `aab908d3b32c6558…` at 179,286 B, raw
`8a14f55a6a8b1415…`, token plane `4cb0b147bdae8ce0…`, bit position 952,362 — every one equal to
move 53's. Receipt: `/Volumes/APDataStore/pact/ddm_mrs7/proof/REPORT_PARSE_CHECK.json`.

**Compiler and linter.** `cc -O2 -std=c11 -Wall -Wextra -ffp-contract=off -fno-fast-math` on all
three C files: **0 warnings, rc 0** (Apple clang 21.0.0). `ruff check --select F,ARG` on
`inflate.py` and both instruments: clean. Receipts, with every file's size, line count and sha256:
`.omx/research/ddm_mrs7_20260916/ENGINEERING_PROOFS.json`.

**Environment and storage.** Bare APFS venv at `.omx/tmp/ddm_mrs7_bare_venv`, built offline from
mrs5's retained wheels with system site packages off. The receipt tier
(`/Volumes/APDataStore/pact/ddm_mrs7`) held 12.2 GiB at preflight, above the 8 GiB floor; the
3.42 GiB raw went to local APFS scratch and was deleted after both comparisons passed, with its
facts recorded first in `RAW_CLEANUP.json`.

Certificate: `/Volumes/APDataStore/pact/ddm_mrs7/proof/IDENTITY.json`, sha `f6eaed20191448836adc18b5933bcf336e0ec3f4a5c936400acab8081491aac7`.

## 6. What I did not do

I did not touch the C arithmetic, add a Python fallback, change `archive.zip`, or take the 16 unread
carrier bytes out. That last one is a real −16 B rate lever
(`.omx/research/ddm_mrs6_20260916/RATE_LEVER_unread_carrier_bytes.md`), but removing them makes a
**new archive**, which is a seal-and-fire path, not a polish. It stays documented and owned by the
frontier line.

The T4 numbers in the README are MAIN's, measured on mrs5's files.
**verdict_scope: that run measured a different file set. `range_decoder.c` is byte-identical here,
the other two C files differ only in comments and identifier names, and the 600/600 identity shows
the Python side produces the same output — so the number describes the same decode path, and the
README now says exactly that. It is not a measurement of mrs7's own bytes, and MAIN's final T4 run
is what the PR body will carry.** The PR draft at
`.omx/research/ddm_mrs7_20260916/PR_BODY_DRAFT.md` carries mrs6's text with the T4 figures as
explicit `PENDING_T4_*` placeholders.

## 7. Prior negatives accounted

PR #140 (unreviewable — mrs6 was the cure and a reader confirmed it; this arm removes what was left);
mrs1–mrs6 (each measured one thing); pd6's rider-drop law (the archive was copied, never re-staged,
and its sha is checked at three points — in `main`, in the instrument, and in the report line); the
ExFAT venv (the bare venv is on APFS under `.omx/tmp/`); the charter-time optimal-form law (no toy:
the identity is the full cold n600 public decode, not a prefix).

---

**composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53) — unchanged by this arm.**
