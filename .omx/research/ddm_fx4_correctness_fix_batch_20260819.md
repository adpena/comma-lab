# ddm_fx4 — correctness fix batch (3 rows)

**Arm:** `ddm_fx4` · **Date:** 2026-08-19
**Authority:** `[local advisory]` — code-correctness work. No scorer run, no archive, no exact row.
**Pointer delta: NONE.** This is apparatus repair, a means. It is not frontier progress.

Operator routing 2026-08-19 ("Fix the most necessary using opus subagents"). Three filed rows,
each verified at source before any edit per the charter-recall clause.

| Row | Filed as | Premise | Outcome |
|---|---|---|---|
| 1 | fp16 cast one line below `clamp_min` destroys its own zero-scale guard | **LIVE — and 35× wider than filed** | fixed at 35 sites / 22 files + regression gate |
| 2 | Launcher supersede corrects paths by KEY NAME, not by VALUE | **LIVE, exactly as filed** | fixed by value-sweep + 5 tests |
| 3 | de1 merge blocked by a file STAGED by nobody in this session | **STALE** — index clean everywhere | closed with evidence; surviving orphan reported |

---

## ROW 1 — the fp16 cast destroys its own zero-scale guard

### Premise: VERIFIED, and materially understated

**Ladder placement: BUG CLASS → silent numerical corruption.** Not a single defect. The filed row
named one line; the same construction is repeated across the repository. It is a *class* because the
mechanism is a property of fp16, not of any one call site, and it fails **silently** — no exception,
no log, just a zero where a positive scale should be.

### The mechanism, measured

fp16's smallest positive (subnormal) value is `2**-24 = 5.960464e-08`. Under round-to-nearest-even,
**every fp32 value at or below `2.980232e-08` casts to exactly `0.0`.** Measured boundary:

```
any fp32 <= 2.980232e-08          -> 0.0 in fp16
guard 1e-8                        -> 0.0        BREACHED
guard 1e-8 / 127.0 = 7.874e-11    -> 0.0        BREACHED
guard 1e-12                       -> 0.0        BREACHED
smallest SAFE fp16 floor          -> 5.960464e-08
```

So a floor written to keep a scale positive stores the very zero it exists to prevent.

### Repro — the exact breaching values, in the real data path

Reconstructed verbatim from the pre-fix source lines:

```
-- variant B latent packer, constant latents (maxs == mins) --
   OLD stored scales       : [0.0, 0.0, 0.0]     all-zero: True
   OLD (latents-mins)/scale: [nan, nan, nan]     <- 0/0
-- pose filler, perfectly constant poses --
   OLD delta_scale         : [0.0]*6             all-zero: True
   OLD deltas/delta_scale contains nan: True
-- variant A, int8 weight scale, all-zero tensor --
   OLD stored scale        : 0.0    decoder 1/scale -> inf
```

**The guard is not even the widest hole.** For `scale = max(max_abs, 1e-8) / 127.0`, the stored fp16
scale is zero whenever `max_abs <= 3.785e-06` — the guard never fires, and the scale is still zero.
Every dequantized weight in such a tensor collapses to 0.

Damage is real where these scales are *written to the archive*: `pack_semantic_pose.py` and
`ddm_sd1_semantic_rd_curve.py` both serialize the scale bytes.

### CLASS-POPULATION

```
CLASS-POPULATION (guard-then-narrow to fp16, floor not fp16-representable)
  live sites found ...... 35
  live sites breaching .. 35   (100% — no site in the class was safe)
  live sites fixed ...... 35
  live files ............ 22
  out of scope .......... 2 sites in experiments/results/kaggle_pr106_.../ (frozen custody snapshot)
```

Found in two passes. The first pass (same-line and adjacent-line) found 25 sites. Widening the
detector to cross-statement floors — per the fractal-audit discipline, a single pass is partial —
found **10 more**, including `src/tac/codec/pose_filler_stc_codec.py`, of which the already-found
`tools/pr101_pose_filler_stc_anchor.py` site is a mirror.

Fixed files (22): 12 `*_as_renderer.py` + `vqvae_as_full_renderer.py` (2 variants each) ·
`codec/pose_filler_stc_codec.py` · `experiments/benchmark_int4.py` · `pr130_lift/editability_levers.py` ·
`pr130_lift/lifted/{evaluate_semantic_quantization,train_semantic_quantized}.py` ·
`pr130_lift/pose/lifted/pack_semantic_pose.py` · `experiments/ddm_{sd1,sm3}_*.py` ·
`tools/pr101_lossy_int4_qat.py` (2) · `tools/pr101_pose_filler_stc_anchor.py`.

### Fix

Re-apply the floor **after** the cast, on the value actually stored and read back, at
`_FP16_MIN_POSITIVE = 5.960464477539063e-08`. The renderer modules import nothing from `tac` by
design (their standalone export property), so the constant is module-local rather than shared.

**Byte-neutral where the encoder already worked** — the clamp only engages below `2**-24`. Measured
on every one of the 12 renderer modules: stored scale bytes and int8 codes are **identical** to
pre-fix for normal weights. It changes bytes only in the degenerate cases, which were broken.

### Self-protection (second landing)

`src/tac/tests/test_fp16_scale_floor_guard.py` — 29 tests:

1. the exact breaching values, with the boundary pinned one ulp either side;
2. the pre-fix expression reproduced, asserting the NaN it produced;
3. per-module: no zero scale for degenerate/near-degenerate tensors (12 modules);
4. per-module: byte-identity with the pre-fix encoder on normal weights;
5. a **CLASS SWEEP** over `src/tac`, `tools`, `experiments`, `scripts` that fails on re-introduction
   anywhere, covering both the same-statement and the cross-statement form.

The sweep **zeroes on the cure**: it reads the last floor in the statement and requires it to sit
after the `float16` token and be fp16-representable. It asserts `scanned > 0` so it cannot pass
vacuously. **Negative control executed:** reintroducing both variants in `ffnerv_as_renderer.py`
made the suite fail (2 failed / 27 passed); restoring made it pass (29 passed).

**Owed, not done:** the STRICT preflight wire-in. `src/tac/preflight.py` is owned by `ddm_sp2` this
session, so the gate lives in the test suite only. Handing the gate function to `ddm_sp2` for a
preflight callsite is the remaining half.

### Separate finding, NOT bundled

In the variant-A helpers the encoder quantizes with the **fp32** scale while the decoder dequantizes
with the **fp16** stored scale — a genuine encoder/decoder mismatch. Fixing it (quantize with
`float(scale_fp16)`) would change archive bytes for *every* tensor, so it is reported, not bundled
into a correctness fix that is otherwise byte-neutral. Needs its own row and its own baseline.

---

## ROW 2 — supersede corrects by key name, not by value

### Premise: VERIFIED at source

`tools/launch_detached_process.py::_derive_watcher_config`. Corrections are keyed on the literal
names `pid_file` (`config.get("pid_file")`) and `log_path` (`if "log_path" in config`). Anything
holding the same wrong value under any other key is never inspected. The live case is
`success_receipts[].path`: when present, the branch records `config_declared` and hands the path
straight back without looking at it.

**Ladder placement: BUG → CONFOUND.** It corrupts the instrument. The launcher prints
`watcher_config_superseded` on stderr and writes an "effective" config, so it *announces* that the
config was made correct while an identical wrong path survives in the same file. That is the
`silence-is-the-defect` class the function's own docstring was written to end — the cure was applied
to one holder of the value and reported as if applied to all.

### Repro

Captured stderr from the negative control, on the 2026-08-16 drift shape:

```
{"watcher_config_superseded": {"kind": "quality", "key": "log_path",
  "declared": ".../run/stdout.log", "derived": ".../run/run.log"}, ...}
```

…emitted while `success_receipts[0].path` still held `.../run/stdout.log`.

### Fix

`_sweep_superseded_values` — once a value is *proven* wrong (the launcher derived the right one),
sweep it by value across the whole document, recursing into lists and dicts. Each replacement is
recorded with its JSON path as a `value_swept` supersession, as loud as a key-name one. Only exact
matches are replaced. Record snapshots taken before the sweep are refreshed so the manifest reports
what the watcher will actually read. The docstring's fail-closed policy list gained the new case.

### Test

`src/tac/tests/test_launch_detached_watcher_value_sweep.py` — 5 tests, including the exact
two-config scenario the row asks for: the same wrong path under `log_path`, `success_receipts[].path`
and `probes[].source`; assert **all** end correct. Plus an executable control showing key-name-only
correction leaves the twin behind; a record-completeness test; an inert-when-nothing-wrong test; and
a test that a merely similar value (`stdout.log.1`) is never rewritten.

**Negative control executed:** disabling the sweep failed 2 of 5; restoring passed 5 of 5.
The 34 pre-existing tests that exercise this tool still pass.

---

## ROW 3 — file STAGED in the de1 index

### Premise: STALE. No index hazard exists.

Evidence, 2026-08-19:

```
main                     : 0 staged
de1 worktree             : 0 porcelain, 0 staged, 0 unmerged, 39,808 tracked
  path .omx/tmp/codex_worktrees/ddm_de1_20260803T112347Z
  HEAD 7a0d6f0abc  branch codexwt/ddm_de1_20260803T112347Z
ALL worktrees (57) swept : 0 staged, 0 unmerged
```

Nothing was unstaged, deleted, or disposed of — there was nothing there. No index was mutated, so
the serializer lock discipline was not needed.

### Surviving finding — the merge never happened, and the work is still orphaned

The blocker is gone but the branch was never merged. `codexwt/ddm_de1_20260803T112347Z` carries
**2 unmerged commits** adding one file:

```
7a0d6f0abc research: complete description-efficiency derivation
181c3e6555 research: derive description efficiency from frozen scorer
A  .omx/research/ddm_de1_description_efficiency_derivation_20260803.md   (783 lines, 41,556 B)
```

I read it in full. It is a complete, self-labeled non-promotable derivation ("Pointer delta: none")
covering what sets `W = 4*N0/Ns`, whether separatrix-dimensional coding exists, the description-space
`R(D)` object, and a ranked table of five falsifiable description classes each with a pre-registered
falsifier and a named consumer.

**I did not land it.** Its own `NEXT-IF-RESUMED` #1 requires MAIN review before landing, and it
requests review of four specific corrections. Landing another arm's memo over that instruction would
be the wrong call. MAIN can recover it with:

```
git checkout codexwt/ddm_de1_20260803T112347Z -- \
  .omx/research/ddm_de1_description_efficiency_derivation_20260803.md
```

**Two items inside it that MAIN should see, because they are stranded on an unmerged branch:**

1. **A live NO-FAKE finding.** §3.3 and §6.7: *"`src/tac/boundary_math/contour_codec.py` is not the
   explicit boundary-edge codec its module prose claims. The implementation serializes every uint8
   label in raster order and LZMA-compresses the dense array."* The memo explicitly routes this to
   MAIN and declines to modify the code. Note that `contour_codec.py` is currently modified in the
   working tree, so someone may already be on it — or may not know.
2. **A correction to an always-loaded instruction.** §6.5: power/Laguerre/tropical equivalence is
   exact at the terminal-feature head, but Morse–Smale equivalence is not automatic. This agrees with
   the already-landed `ddm_mf1` retirement of the Morse–Smale reading, so it is corroboration rather
   than news — but it was derived independently.

**Ladder placement: not a bug — a signal-loss instance.** The `deferral-scatter / unmerged-branch
recall surface` genus: a completed unit's output sits reachable-but-unreferenced, and only an
unrelated audit found it.

---

## What I did not do

- No preflight gate wired (`preflight.py` owned by `ddm_sp2` this session).
- No encoder/decoder scale-mismatch fix (byte-moving; needs its own row).
- No de1 merge (MAIN-review-gated by the memo itself).
- Did not touch `ddm_fx3` or `ddm_jg3` surfaces.
- Two pre-existing failures I proved are **not** mine: `test_train_mnerv_as_renderer` and
  `test_train_vqvae_as_renderer` fail with `SystemExit: 7` from the admission guard. Reverting
  `mnerv_as_renderer.py` to HEAD reproduces the identical 4 failures.
- Pre-existing unused `math` import in `train_semantic_quantized.py` left alone (present at HEAD).

## Verification run

- `test_fp16_scale_floor_guard.py` 29 passed · `test_launch_detached_watcher_value_sweep.py` 5 passed
- 320 renderer-suite tests pass (8 pre-existing admission-guard failures, proven pre-existing)
- 47 codec/int4 tests · 59 editability/semantic tests · 34 launcher tests — all pass
- ruff (repo config) on the 23 tracked files: **60 errors at HEAD → 59 now.** Zero introduced; the
  fix removed one pre-existing `RUF034` dead ternary in `benchmark_int4.py`.
- Both negative controls executed and reported above.
