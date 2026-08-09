# Review round 2 — the fx-wave cure guards, and the population they sit in

**Entry criteria set by round 1:** run the fx1/fx2/fx3 focused suites end to end, and widen the
declaration-vs-code binding hunt beyond my own landing.

**Round 2 verdict: 1 FINDING, more consequential than round 1's. Clean-pass counter stays 0/3.**

---

## F5 (CONFIRMED, narrowly CURED) — the fx2 and fx3 cure guards were never collected

**Chain of how this surfaced, including my own near-miss:**

1. I ran `pytest tests/ -k "fx1 or fx2 or fx3 or fx5"` → 6 passed. Only my fx5 suite exists under
   `tests/`. The obvious read is "the other arms shipped no tests."
2. That read would have been **wrong**, and wrong in the session's recurring shape: my selector was
   scoped to `tests/`, so its silence was about the *scope*, not the *repo*. A repo-wide `find`
   located them immediately at `src/tac/pr130_lift/tests/` — three files the fx2 and fx3 arms did
   land, and `git log --diff-filter=A` confirms both commits (`9049e1caa5`, `f43180a761`) added them.

   That is the **third scope-shaped near-miss today**: a `--include=*.py` glob eaten by zsh, a `tests/`-
   scoped selector, and (round 1) an unfinished background job read as empty. Each returns silence that
   is indistinguishable from a real negative.

3. **The actual defect, MEASURED:** `pyproject.toml` had `testpaths = ["tests", "src/tac/tests"]`.
   A default collection gathers **47,159 tests and exactly 0 of them from `pr130_lift`.**

So the fx2 adapter-selection cure and the fx3 EMA/resumability cure — both landed today, both real —
shipped with guards that **no default or CI invocation ever runs.** Same orphan genus as round 1's
manifest-nobody-reads, one layer out: this time the unwired thing is the guard on a cure.

**Population, with its denominator (measured, not estimated):**

```
test files under src/ + tests/ : 3,961
  COLLECTED by testpaths       : 2,860
  ORPHANED (never collected)   : 1,101   = 27.8%
largest orphaned dirs:
  137  src/tac/canonical_equations/tests
  108  src/tac/witness_dsl/tests
   89  src/tac/optimization/tests
   68  src/tac/witness_control/tests
   67  src/tac/boundary_math/tests
```

**Cure — deliberately NARROW, and here is why.** I added exactly one root:
`src/tac/pr130_lift/tests`. Measured cost: **42 tests, 2.39 s, all green**; collection goes
47,159 → 47,201. The fx-wave selector now returns 17 instead of 6.

I did **not** blanket-widen to `src/tac`. Adding 1,101 uncollected files at once would (a) change
CI runtime by an unmeasured amount and (b) potentially turn RED files into blocking failures on an
innocent commit — the exact hazard #851/#854 already recorded. **Whether each of the other 1,099 is
deliberately excluded (slow, MLX-gated, device-pinned) or accidentally orphaned is UNMEASURED**, and
that absence is the honest state, not a clean bill. The narrow add is justified by evidence specific
to it: it is the sole guard on two cures landed today, and its cost is measured.

The `testpaths` line now carries that reasoning inline, so the next reader sees why one package-local
root is listed and the rest are not.

---

## F2–F4 re-verified under the fix

With `pr130_lift` collected, the fx2/fx3 guards run: **42 passed in 2.39 s.** Round 1's source-level
re-derivation of the fx2 cure (`use_sparse = mode == REFERENCE_SPARSE_MODE`) is now backed by guards
that actually execute. My fx5 binding suite (6 tests, 3 mutation controls) remains green.

---

## What round 2 did NOT check

- **The 1,099 other orphaned test files.** Not run, not classified. A sweep would need to run each
  orphaned root, record green/red/slow, and classify deliberate-vs-accidental per directory — that is
  a real piece of work, not a line in `pyproject.toml`, and it is now a named debt with a denominator.
- **Whether CI uses the same `testpaths`.** I verified the local default collection only. If CI
  invokes pytest with an explicit path, the orphan set differs and this measurement is scoped to the
  default invocation.
- No score claim. `score_claim=false`.

## Round 3 entry criteria

Counter is **0/3** — two rounds, two findings, both in the apparatus rather than the arithmetic.
Round 3 should (a) classify the 1,101 orphaned test files by directory with a green/red/slow read,
and (b) test the round-1 genus at repo scale — manifests and declarations that no code reads — since
both findings so far are the same shape: *an artifact that looks like a contract but cannot fail.*
