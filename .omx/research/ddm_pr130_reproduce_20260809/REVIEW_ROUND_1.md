# Review round 1 — the fx1–fx5 cure wave (task #823, clean-pass counter)

**Protocol:** re-derive from primary artifacts, do not confirm the arms' summaries. Fixes are
unreviewed new code; a cure wave is exactly where the next defect hides. Round result below.

**Round 1 verdict: 1 FINDING — in MAIN's own landing. Clean-pass counter stays 0/3.**

---

## F1 (CONFIRMED, CURED) — I enriched a declaration that no code reads

**Where:** `src/tac/pr130_runtime/fx1_runtime_tree/runtime-dependencies.json`, landed by me at
`0889b41fec` two commits before this review.

**What I did:** after the Linux bare-image run exposed that FX1 declared 1 of 3 third-party imports,
I enriched the manifest to enumerate all three with `provisioning` classes, `pin_source`,
`imported_by`, and a `closure_provenance` block.

**The defect:** a consumer sweep found the file is read by **zero executable code**. The only
in-repo match outside docs is inside my own error-message *string*. `inflate.sh` independently
hardcodes `EXPECTED_CONSTRICTION_VERSION=0.5.0` (line 6) and the asserted tuple `("numpy", "torch")`
(line 25). Two sources of truth, no binding.

I made the declaration **more complete without making it more binding** — which is the config-orphan
genus, not a cure. It is also the third instance of the same shape in one session: an artifact that
looks like a contract but cannot fail.

**Cure (landed with this review):** `tests/test_fx5_runtime_closure_binding.py`, six tests. The
strongest derives the closure from the receiver source with `ast` and compares it to the manifest, so
a future receiver module that adds an import **cannot** silently escape — the exact defect the Linux
bare image found by accident is now structurally impossible.

Deliberately **not** done: making `inflate.sh` parse JSON at decode time. That would add a failure
mode to the shipping path for no benefit. The contract is enforced at test time, where drift fails
loudly and costs the decode path nothing.

**Mutation controls (a test that only passes proves nothing):**

| mutation | expected | observed |
|---|---|---|
| `constriction.version` 0.5.0 → 0.5.1 | version-binding test fails | **FAILED** `test_entrypoint_version_matches_declared_version` (1f/5p) |
| drop `numpy` from `dependencies` | closure + assert-set tests fail | **FAILED** ×3 (3f/3p) |
| `torch.imported_by` → `["inflate.py"]` only | AST-derived test fails | **FAILED** `test_imported_by_matches_actual_module_imports` (1f/5p) |

Manifest restored byte-identical after each; 6/6 pass at rest.

A sixth test is a **direction control**: it refuses a manifest that moves `constriction` into the
asserted-absent set, because that would make the entrypoint exit 68 on a clean host instead of
bootstrapping — undoing FX1's entire cure. Guarding only one direction is how you cure a defect into
its mirror image.

---

## F2–F4: the other three arms re-derived, no findings

**fx2 (adapter selection) — CONFIRMED at source, not from the summary.** `pose/mps_port.py:199`
now reads `use_sparse = mode == REFERENCE_SPARSE_MODE`. The defective
`use_sparse = device.type != "mps"` is gone: **device no longer selects mechanism.** `RowLocalDenseAdam`
survives as a named opt-in (`DENSE_ADAPTER_MODE = "dense-adapter"`), and the MPS reference path is
pinned to `PINNED_MPS_TORCH_VERSION = "2.10.0"`. The cure is real.

**fx3 (EMA argmax parity) — correct sampling shape, and I added the bound it lacked.** Pair selection
is *seeded random without replacement* (`seed 20260809`, IDs `[458, 460, 529, 585]`), **not a prefix** —
so the m88/m96 prefix hazard does not apply (that law is prefix-specific; a random subsample is
bias-safe in shape, only lower-powered).

The arm reported "0/6,291,456 argmax pixels differed" without stating what that bounds. Derived here:

```
n = 6,291,456 comparisons, 0 events
rule of three -> 95% upper bound on flip rate = 3/n = 4.768372e-07
PR130 base d_seg = 0.028609/100      = 2.860900e-04
bound is 600.0x BELOW the base seg term ; worst-case S contribution 4.768e-05
```

So the honest adopt statement is not "EMA changes nothing" but **"the EMA argmax-parity risk on the
deployed QAT path is bounded at ≤1/600 of the base seg term with 95% confidence."** That is adequate
for the adopt decision and is now stated rather than implied. Zero observed ≠ zero rate.

**fx4 (GT lineage) — strongest shape in the wave.** n600 full-population, explicitly "no prefix and no
sampling," 20,750/117,964,800 seg sites. Nothing to re-scope.

---

## What this round did NOT check (stated, not implied)

- **fx1/fx2/fx3 test suites were not re-run end to end** — only the cure mechanisms were re-derived at
  source. A full-suite run belongs to round 2.
- **`assert_provided_deps` uses `find_spec`, not `import`.** `find_spec` can succeed where import fails
  (a broken C extension resolves as a spec). The assert is therefore fail-closed in the direction that
  matters — absent → refuse — but a *broken* numpy/torch install still fails later at the real import
  with a raw error. Named, not fixed; fixing it would import torch (~seconds) on every decode.
- No score claim anywhere in this round. `score_claim=false`.

## Round 2 entry criteria

Counter is **0/3**. Round 2 must run the fx1/fx2/fx3 focused suites, and should attack the
declaration-vs-code binding genus more broadly: this round found one instance because I went looking
in my own landing. The population of manifests-nobody-reads across the repo is **unmeasured**, and
that absence is the honest state, not a clean bill.
