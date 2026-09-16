# ddm_mrs5 — fresh-eyes review of the three predecessor instruments, and what I verified myself

<!-- # FORMALIZATION_PENDING: review findings on evidence instruments; no law registered. -->

Axis `[macOS-CPU advisory]`, `score_claim=false`. A fresh-context reviewer read
`experiments/ddm_mrs5_{proof,smoke,bootstrap}.py` in full on 2026-09-16 with one question: which of
their checks could report PASS while measuring nothing, and which of their claims are asserted rather
than measured. Six findings came back. I then verified each against the primary artifacts myself,
because a reviewer's finding is no more load-bearing than a producer's claim until someone checks it.

## F1 — one of the two geometry comparisons cannot fail. CONFIRMED by me; NOT load-bearing.

`ddm_mrs5_proof.py:67` runs `compare_bits(self.native.plane, self.python.plane, 'geometry plane')`.
Both planes are written by the **same pure-Python statement** — `self.plane.ravel()[positions] = symbols`
appears verbatim in `CausalGeometry.observe` (`submissions/mrs5/inflate.py:224`) and in
`NativeGeometry.observe` (`inflate.py:288`) — from the same `positions` and `symbols`. The C library
contributes nothing to either side, so this comparison is **structurally incapable of failing**. It is
the vacuity genus, inside an otherwise sound proof.

**Why it does not touch the parity claim:** the number this arm cites — 9,437,184 geometry bin values
per path over 48 pairs — comes from the *other* comparison, `ddm_mrs5_proof.py:62`
(`compare_bits(a, b, 'geometry bins')` inside `ShadowGeometry.contexts`). I verified that
`NativeGeometry.contexts` (`inflate.py:273-281`) calls `_GEOMETRY_LIBRARY.geometry_contexts(self.handle, …)`
— it reads the **C library's own accumulated moment state through the handle** — while the Python side
derives the same values from its own numpy moments. So a divergence in the C `observe` arithmetic
surfaces at the very next `contexts` call. The geometry C path is genuinely covered; it is covered by
`bins`, not by `plane`. The `plane` row should be dropped or replaced with a read-back of the native
record the way `ddm_mrs4_proof.py:140-144` does through `native_arrays`.

## F2 — the smoke's parity prerequisite is not bound to the receiver it smokes. CONFIRMED gap; CLOSED here by measurement.

`ddm_mrs5_smoke.py:109-111` accepts the 48-pair parity result on `matched_pairs != 48` alone. Nothing
cross-checks that the parity run used the receiver being smoked: the parity ran against
`/Volumes/APDataStore/pact/ddm_mrs5/development_public/`, the smoke against `submissions/mrs5/`, and the
smoke's own `binding` does not carry the parity receipt. `ddm_mrs1_public_smoke.py:305-307` binds
exactly this (`predecessor_receiver_sha256` vs `public_receiver_sha256`); mrs5 dropped it. A stale or
different-receiver parity run would satisfy this gate silently.

**Measured here:** the parity result's own `binding.sources` does record `{path, bytes, sha256}` for the
four development_public files, and I compared all five shipped files sha-for-sha:
`inflate.py f79f1e3f…`, `geometry.c ac99ab49…`, `corrector.c 71f632bc…`, `range_decoder.c d70a4949…`,
`inflate.sh f7f1b608…` — **all five identical** between `development_public/` and `submissions/mrs5/`.
So the 48-pair parity was measured on exactly the shipped bytes. The gate is weak; this instance is
sound, and it is sound because I checked, not because the gate checked.

## F3 / F4 — two resume-path claims are earned only on a first execution. CONFIRMED hazard; DID NOT FIRE here.

`stale_library_removed=True` (`ddm_mrs5_smoke.py:205`) and `cold_public_subprocess=True` (`:256`) are
written unconditionally, while the stages that earn them (`:167-172`, `:233`) are skipped when a cached
`*.DONE.json` receipt exists. `ddm_mrs1_public_smoke.py:304` carries `resumed_stage_only=True` for this
case; mrs5's certificate has no equivalent field, so a resumed run would assert both claims without
re-earning either.

**Measured here:** `public_native/attempts/` holds exactly one attempt per stage (environment, help,
default_guard, public_decode) and `public_fallback/attempts/` exactly one per stage (environment, help,
fallback_pairs). **Every stage executed once; nothing was served from cache.** Both claims were earned
in the runs this arm cites.

## F5 — `ddm_mrs5_proof.py:44` omits `promotable=False`. CONFIRMED; cosmetic.

The binding carries `axis='[macOS-CPU advisory]'` and `score_claim=False`, and every sibling also sets
`promotable=False` (`ddm_mrs4_proof.py:181`, `ddm_mrs5_smoke.py:54`/`:146`, `ddm_mrs5_bootstrap.py:45`).
`RESULT.json` inherits the gap. No claim in this arm rests on it, and the two flags that matter are
present.

## F6 — "bare" venv is partly asserted. CONFIRMED; disclosed scope limit.

`ddm_mrs5_bootstrap.py` reuses the venv when `pyvenv.cfg` exists and probes only `numpy, torch, brotli`;
nothing enumerates `site-packages`, yet `READY.json` writes `bare=True, system_site_packages=False`
unconditionally. Measured evidence is: the `pyvenv.cfg` string, `sys.prefix`, `ENABLE_USER_SITE`, and
three module paths. The compiler-absent smoke's substance does not depend on it — that claim rests on
the three `.so` being absent and the receiver's own fallback stderr strings, both checked at
`ddm_mrs5_smoke.py:194`/`:207`. The venv itself was certified-removed afterwards
(`bootstrap/VENV_CLEANUP_DONE.json`, `environment_absent: true`, certificate sha `3dd4b256…`).

## What the reviewer found CLEAN, and I agree after checking

- **No silent same-path risk.** `NativeGeometry` subclasses `CausalGeometry` but overrides
  `__init__`/`contexts`/`observe` with ctypes bodies, so the two shadow arms are genuinely distinct
  implementations. `proof.py:50` refuses unless all three `.so` loaded; `smoke.py:50` refuses unless all
  three are `None`. The `|| true` on `inflate.sh`'s compiles is intended receiver behaviour, and those
  two checks are exactly what stops it becoming a silent-fallback comparison.
- **No swallowed exceptions.** No bare `except`, no `or True`, `try/finally` with no handler at
  `proof.py:90-94`, `check=True` on the subprocess at `bootstrap.py:66`.
- **No shared state between arms.** Two independent instances; `vectors.clear()` per pair; in-place
  restore with a class-name check that raises on mismatch; the shadow is step-locked so the native
  output is fed to both `observe`s.
- **The counts are enforced, not trusted.** 48 pairs checked three separate times against the seeded
  24-stratum × 2 selection; the n600 run does 600 literal iterations with exact `RAW_BYTES` equality,
  trailing-byte rejection, 600 per-pair hashes compared element-wise to the mrs1 reference, and an
  mtime-stability check.
- **The scope of the n600 claim, stated exactly:** `smoke.py:136` forces the mrs5 archive to equal
  mrs1's `ARCHIVE_SHA`, so what is proven is **receiver-refactor identity at a fixed archive**, not an
  independent decode. That is precisely the claim this arm makes and no more.

## Disposition

F1's vacuous row and F2/F3/F4's missing bindings are **instrument defects worth fixing before the next
arm reuses these files**, and none of them invalidates a number this arm reports — I checked each
against the primary artifacts rather than accepting either the producer's claim or the reviewer's.
The files are committed **unmodified**: their bytes are pinned in `REVIEW_PASSES.json` and they produced
the committed receipts, and editing an evidence producer after it has produced the evidence would
break that custody for a cosmetic gain. The fixes belong in the successor instrument.
