# QUEUED EDIT (do NOT apply mid-run) — native per-class realization-vs-gradient in the VERDICT row

**Status: QUEUED for the post-c2 boundary.** The target file
`experiments/train_levelset_witness_realized_through_R_mlx.py` is imported by the LIVE c2 run
(`levelset_n600_witness_20260717T113932Z`, pid ~13783). Per the operating contract, edits to
in-trainer files do NOT land mid-run. The external modules (organ upgrades A/B/C) are already
main-safe and give this signal NOW on the sacred run via
`tac.witness_control.realization_regime` + `tools/costate_live_ingest.py` +
`tools/costate_digest.py` reading `.omx/state/witness_realization_regime.jsonl`.

This file is the exact, ready-to-apply native edit so future runs carry the per-class
realization-vs-gradient split in EVERY verdict row alongside `d_seg_by_class` /
`flip_share_by_class`. **It is score-neutral OBSERVABILITY → defaults ON** ("'off' is a tracked
queue / observability defaults ON when score-neutral"), gated only on a compute-cadence
(the VJP needs a grad-enabled forward, unlike the inference-mode verdict).

## Why native (in addition to the external consumer)

The external module reads the ROLLING EMA npz on disk and re-decodes/re-scores; it is exact but
costs a decode+SegNet pass. The trainer already has, in the annulus branch, the realized argmax
maps and the frozen `seg_cpu` + camera frames `f1s` + GT `lstars` in scope — so the native path
can compute the per-class sub-LSB split on the SAME frames the verdict already scored, at a
small stratified pixel sample, for a few seconds every N verdicts.

## Exact insertion (anchored to current line numbers, 2026-07-17)

Site: the verdict-row assembler, right after the existing per-class block at
`experiments/train_levelset_witness_realized_through_R_mlx.py:9340-9341`:

```python
            if isinstance(_per_class, dict) and "error" not in _per_class:
                row["d_seg_by_class"] = _per_class.get("d_seg_by_class")
                row["flip_share_by_class"] = _per_class.get("flip_share_by_class")
```

Add immediately below (inside the same `if`, so it only fires when the annulus branch collected
realized maps — the same gate the per-class d_seg already respects):

```python
                # (queued 2026-07-17) NATIVE per-class realization-vs-gradient split — the
                # decision-critical "is class c's remaining error sub-LSB (irreducible ->
                # terminal SOLVE) or amplitude-open (recoverable -> keep training)". Score-
                # neutral OBSERVABILITY => default ON; gated only on a compute cadence (the VJP
                # needs a grad-enabled SegNet forward, unlike the inference-mode verdict).
                # Reuses the external module math (no re-derive); fail-open (never breaks a row).
                if _realization_regime_due(ep):  # e.g. every 4th verdict; ledger-recorded cadence
                    try:
                        from tac.witness_control.realization_regime import (
                            per_class_regime_from_realized,
                        )
                        row["realization_regime_by_class"] = per_class_regime_from_realized(
                            seg_cpu, f1s, lstars,
                            realized=_per_class.get("realized_argmax"),  # the collected maps
                            n_pixels=160, seed=int(ep),
                        )
                    except Exception:
                        pass  # observability must never break the verdict row
```

Notes for the applier:
- `_realization_regime_due(ep)` is the cadence gate to add (default: `ep % 100 == 0`, i.e. every
  4th verdict at the 25-epoch verdict cadence). Record its reason in run provenance per the
  default-derivation discipline; do NOT hardcode a silent switch. A `--realization-regime-cadence`
  DSL lever (int, default 100; 0 = off) is the clean form so the cadence is config-visible.
- `per_class_regime_from_realized(...)` is a THIN helper to add to
  `tac.witness_control.realization_regime` that: builds a `MarginSnapshot`-equivalent from the
  ALREADY-REALIZED argmax maps + the frozen logits (a grad-enabled re-forward on ONLY the
  sampled flip frames — not all n600), then calls the existing `vjp_sub_lsb_over_snapshot` and
  returns `result.per_class`. It must reuse the exact `min_norm_crossing_max_coord` +
  `SUB_LSB_MAX_COORD` convention (NO second implementation). If `realized` is None it falls back
  to decoding from the EMA npz (the external path).
- The field is additive + row-only (never history/result.json), exactly like `d_seg_by_class`
  (the surrounding block's own contract) — so byte-identity of the archive/score is preserved by
  construction.
- The dashboard/costate-digest already know how to read `per_class` (the external
  `witness_realization_regime.v1` schema); the native field name `realization_regime_by_class`
  should carry the same per-class dict shape so both surfaces agree.

## Verification the applier must run at the boundary
1. `ruff check` + the four organ tests green.
2. A 2-verdict smoke on a throwaway run dir confirming the field appears only on the cadence
   epochs and is byte-identical-absent otherwise.
3. Confirm the archive bytes / d_seg are unchanged with the field on vs off (observability
   guarantee).
