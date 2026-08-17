# ddm_kp2 — payload-retention census, detector blind spot #4, and the STRICT-flip verdict

Date: 2026-08-16 · Arm: ddm_kp2 (successor to ddm_kp1, commit `acd8c87c5e`) · Task #1001
Axis: static-analysis census + one executed retrofit. No score claim. `score_claim=false`.
Rule: CLAUDE.md **ALWAYS KEEP THE PAYLOAD** (P0, DEF CON 1000, operator 2026-08-09).

## Verdict first

1. **kp1's 581 / 275 reproduces exactly** — but only at the scope preflight actually uses.
2. **kp1's pricing-helper hypothesis is FALSE.** The split is 309 PRICING / 272 RECORD, not
   123 / 458, and the pricing class does **not** collapse to a handful of shared oracles.
3. **A fourth detector bug exists** (kp1 fixed three): project-local byte-persistence
   wrappers were invisible. 20 findings were retained payloads reported as discards.
4. **kp1's declared `.write()` relaxation hides nothing.** Measured, not assumed: it
   suppresses 44 findings across 29 files, and all 44 are the correct length-prefix idiom.
5. **No STRICT flip is safe for any scope.** Every root is far from zero.

## STAGE 0 — the population, with its denominator

Re-derived independently (`audit_measure_and_discard_payload`, live tree 2026-08-16):

| scope | discovered | examined | AST candidates parsed | unreadable | findings | files |
|---|---:|---:|---:|---:|---:|---:|
| default roots (no exclusions) | 51,303 | 51,303 | 4,371 | 0 | **873** | 432 |
| **preflight scope** (`excluded_parts=results,tests,fixtures`) | 6,576 | 6,576 | 1,401 | 0 | **581** | **275** |

The 581 reproduces kp1 exactly. The gap to 873 is **not** noise and should not be quoted as
the population: 130 findings live under `experiments/results/`, a gitignored rebuildable
artifact tree that contains **vendored copies of our own `src/tac`** (e.g. a second
`archive_codec.py` inside a kaggle workspace), and 162 live in tests/fixtures. Excluding
them is right; quoting 581 without naming the exclusion is not.

**The tree is live.** Sister arms edited files mid-census (denominator drifted
6,576 → 6,579 within the hour; 2 findings appeared in an untracked sister file). Every
count here is a timestamped sample, not a constant.

## STAGE 1 — the pricing-helper hypothesis is FALSE

Classifier: for each finding, locate the enclosing `FunctionDef` and ask whether the
`len(...)` is the function's OUTPUT (a cost oracle) or lands in a row/record.

| class | findings | shape |
|---|---:|---|
| PRICING — the scalar is returned | **309** | a cost oracle priced per call |
| RECORD — the scalar lands in a row | **272** | the genuine run-level P0 queue |

kp1 predicted ~123 pricing / ~458 record. Both halves are wrong, and the *direction*
matters: the genuine queue is **272**, not 458 — 40% smaller than the handoff claimed.

**The decisive half.** kp1's live hypothesis was that the pricing class collapses to a few
SHARED oracles, so waiving those few would clear the class cheaply. It does not. The 309
findings spread across **209 distinct (file, function) pairs in 170 files**. The repeated
names are **copy-pasted local definitions, not one shared import — measured by grep,
zero import sites for any of them**:

| oracle name | findings | files | `import` sites | `def` sites |
|---|---:|---:|---:|---|
| `run_codec` | 19 | 8 | **0** | 8 separate local defs |
| `lzma1_raw` | 7 | 7 | **0** | 11+ separate local defs |
| `_brotli_len` | 6 | 6 | **0** | 6 separate local defs |
| `encode_decode_measure` | 7 | 3 | **0** | 3 separate local defs |

So the cheap waiver path does not exist: clearing the pricing class by waiver costs ~209
edits. The real cure is a **canonical pricing oracle in `tac`** that prices AND retains,
with the duplicated locals migrated to import it. That is a named next-arm item, below.

## STAGE 2 — detector blind spot #4, and one file taken to zero

### The bug: project-local byte-persistence wrappers

kp1's named top target, `tools/measure_realization_g2_lattice.py` (13 findings), turned out
to be **already compliant**. Four lines above the flagged `len()` it writes every payload:

```python
_atomic_bytes(output_root / "payloads" / "frozen_scorer_palette.g2pal", palette_payload)
_atomic_bytes(output_root / "payloads" / "static_charts_n64.zlib9", static_zlib)
_atomic_bytes(output_root / "payloads" / "lane_chart.brotli11", lane_brotli)
_atomic_bytes(output_root / "payloads" / "exact_iframe.brotli11", exact_iframe)
```

The gate matched persistence against a fixed NAME list, so it could not see the repo's
dominant retention idiom. **38 modules** define this copy-pasted atomic-write helper under
**15 different names** (`_atomic_bytes`, `_atomic_write`, `atomic_bytes`,
`_publish_immutable`, `save_decoder_npz`, `write_deterministic_zip`, ...).

`_local_byte_persisters` now discovers them structurally. Two guards keep the gate exact at
the anchor boundary, and **both are load-bearing**:

* the sink must be **binary** — a text `open(p, 'w')` or `json.dump` does not qualify;
* the parameter must reach that sink **unreduced**.

The second guard was found by adversarial review of my own first draft, which would have
read `def rec(p, payload): json.dump({"n": len(payload)}, open(p,"w"))` as persistence —
**line 47 of the anchor itself, laundered one call deeper.** Pinned by
`test_scalar_only_wrapper_is_not_a_persister`.

**Effect: 20 findings cleared, across 7 files. Every one hand-verified at source as a
genuine retention** (`_write_archive`, `write_deterministic_zip`,
`_atomic_write(all_packets_path, bytes(packet_stream))`, `_publish_bytes`, ...). No real
discard was blinded. The 23.8% upper bound (138 findings in modules that *define* a
wrapper) is loose because defining one does not mean this payload flows into it; the
realized effect is 3.4%.

### The retrofit: `experiments/ddm_lr2_realization_ladder.py`, 8 → 0

Genuine discards: a per-candidate sweep (M × arms × pairs) solving real carriers with a
15 s/pair Adam descent, pricing each with LZMA1, and keeping only the lengths — the anchor
shape exactly, including its per-candidate half ("the discarded loser measured −2,120 B
better than the shipped winner").

All four call sites now route through `retain_arm_payload` → `tac.payload_retention`.
This also fixes a latent routing failure: the module hardcoded `OUT_DIR` to
**VertigoDataTier, measured 893 MiB free (100% capacity)** while APDataStore has 240 GiB;
`retention_root` picks the tier that fits, so payloads landed on APDataStore.

**Executed, not merely edited** — three subcommands run, 16 payloads verified:

| subcommand | arms retained | verification |
|---|---:|---|
| `solve0 --pairs 0 --m-max 8 --steps 2` | 6 | sha256 + byte count |
| `solve --pairs 0 --steps 2` | 7 | sha256 + byte count + **LZMA1 re-price** |
| `keys --pairs 0 --steps 2 --m-keys 8` | 3 | sha256 + byte count + **LZMA1 re-price** |

The re-price is stronger than a hash echo: the retained bytes were re-compressed under the
script's own filter chain and the recorded `payload_lzma1` scalar had to fall out of them.
It did, for all 10 tested. **0 failures.** The pre- and post-retrofit result rows are
**byte-identical** modulo the added custody block, so the retrofit is observationally inert
on the science.

The file's one remaining site is its `lzma1_raw` oracle, now waived with a substantive
rationale — legitimate precisely *because* every caller retains before pricing.

## The known limitation — measured, not inherited

kp1 declared honestly that bare `.write(...)` counts as escape without proving the buffer
is itself persisted. Counterfactual census (disable the bare-write escape, diff):

* **44 findings suppressed across 29 files**, denominator 1,402 AST candidates.
* 38 sit in modules that `return buf.getvalue()` — duty moves to the caller, correct.
* The remaining 6 were read at source. **All 6 are correct**: length-prefix-plus-payload
  into a real binary handle (`with archive_bin_path.open("wb") as f: f.write(struct.pack("<I", len(compressed))); f.write(compressed)`),
  or `handle.write(payload0); handle.write(payload1)` after a length/sha header.

**The relaxation hides no real discard in the measured population.** It stays.

## STAGE 3 — the STRICT flip is NOT safe, for any scope

| scope | live findings | flip? |
|---|---:|---|
| `experiments/` | 165 | NO |
| `tools/` | 217 | NO |
| `src/` | 173 | NO |

No subset is at zero, so kp1's recommendation stands and the wire-in remains **warn-only**.

**A hazard the next arm must not step on.** The scope is not stated once. Preflight calls
`audit_measure_and_discard_payload(..., excluded_parts=("results","tests","fixtures"))` and
reports **555**; the module's own `check_no_measure_and_discard_payload()` defaults to *no*
exclusions and returns **825**. Flipping that function to `strict=True` would therefore
enforce a population 49% larger than the one preflight has been reporting as the queue —
including vendored copies of our own source under `experiments/results/`. Any future flip
must pin the scope explicitly at the call site.

## NEXT_IF_RESUMED

| # | row | owner | fire-condition |
|---|---|---|---|
| 1 | Canonical pricing oracle in `tac` (prices AND retains); migrate the ~209 copy-pasted locals | next payload-retention arm | now — it is the only cheap path through the 309-finding PRICING class |
| 2 | Retrofit the RECORD class, 272 findings; next targets `ddm_p3v2_optimal_form_pose_resolve` (7), `road_undriv_bulk_field` (8), `analytic_lane_render_band` (7) | next arm | after #1, or in parallel |
| 3 | Pin the scope of `check_no_measure_and_discard_payload` to preflight's, or make the default explicit | same arm as #1 | before ANY strict flip |
| 4 | Cross-module persister detection (a helper imported from a sibling repo module is still invisible) | deferred | only if a census shows it is material; same-module was the dominant case |
| 5 | STRICT flip | deferred | blocked on a scope reaching live count 0 |

## Residual over-permissiveness, stated

`_local_byte_persisters` registers helper names **module-wide** (a nested helper is visible
to sibling scopes), and a helper that persists *one* parameter clears *any* unreduced
argument at its call sites. Both match the coarseness the gate already had for canonical
persisters, so neither is a new class — but both are real, and both are why the 20 cleared
findings were hand-verified at source rather than trusted.
