# Retroactive sweep for Wave 5 NSCS06 v8 cargo-cult re-audit + fix — 2026-05-29T21:30:00Z

Per Catalog #348 `check_new_gate_landing_includes_retroactive_sweep_evidence` discipline + the canonical 4-field contract.

## Field 1 — Bug-class symptom signature

The Wave 5 landing introduces TWO canonical helpers + 1 NEW canonical equation + 1 NEW canonical anti-pattern. The bug-class symptom signatures are:

1. **Cargo-cult #3 wire-in bypass**: trainer source contains inline `hashlib.sha256(chroma_lut.tobytes()).digest()[:32]` rather than canonical helper invocation. **Detection AST predicate**: `import hashlib` immediately followed by direct `.digest()[:N]` slice on a SHA-256 output, in any file that ALSO imports from `tac.substrates.nscs06_v8_chroma_lut`. Wave 5 fixed the 1 instance at `experiments/train_substrate_nscs06_v8_chroma_lut.py:781-784`.

2. **Cargo-cult #6 cls_lowres strided NEAREST without empirical-vs-MODE**: trainer source contains `cls_full[:, ::ds, ::ds]` strided indexing directly (rather than routing through canonical helper). **Detection regex**: `cls_full\[:,\s*::\s*\w+\s*::\s*\w+\]` OR `cls_\w+\[:,\s*::\s*\w+,\s*::\s*\w+\]` in any file that ALSO writes a `cls_bytes` variable.

## Field 2 — Pre-fix window

The pre-fix window for both bug classes spans the **2026-05-21 → 2026-05-29 inclusive** range. Specifically:

- **Cargo-cult #3**: 2026-05-26 commit `a6e2a06e3` landed the canonical helper but the trainer at `experiments/train_substrate_nscs06_v8_chroma_lut.py:783-784` was authored BEFORE the helper landed (2026-05-21 OVERNIGHT-V) and was never migrated. Re-audit window: 2026-05-26 → 2026-05-29 (3 days).
- **Cargo-cult #6**: 2026-05-21 OVERNIGHT-V landed cls_lowres strided NEAREST as inline trainer code. Never audited until Wave 5 (8 days exposure).

## Field 3 — Historical KILL/DEFER/FALSIFY search results

`grep -ri "nscs06_v8" -l ~/.claude/projects/-Users-adpena-Projects-pact/memory/ .omx/research/ 2>/dev/null | xargs grep -l "FALSIFIED\|KILLED\|RETIRED" 2>/dev/null | head -20`:

NO historical KILL/FALSIFIED/RETIRED verdicts on NSCS06 v8 chroma_lut found. All prior verdicts are PROCEED-with-revisions or DEFER. The 6 prior rc=22/rc=1 Modal dispatches per 2026-05-26 audit were `DEFERRED-pending-research`, NOT killed. Per Catalog #307 paradigm-vs-implementation classification: those dispatches were IMPLEMENTATION-LEVEL failures (rc=22/rc=1 dispatch crashes) NOT paradigm-level falsifications.

`grep -rn "nscs06_v8.*KILL\|cls_stream.*FALSIFIED\|chroma_lut.*KILLED" .omx/research/ ~/.claude/projects/-Users-adpena-Projects-pact/memory/ 2>/dev/null | head -5`:

No matches found. NSCS06 v8 substrate-class paradigm INTACT per `[[forbidden-premature-kill-without-research-exhaustion-the-kill-too-fast-trap]]`.

## Field 4 — Per-finding RE-EVAL-priority assignment

| Finding | Affected prior artifacts | RE-EVAL-priority | Action |
|---|---|---|---|
| Cargo-cult #3 wire-in bypass FIXED | 6 prior Modal dispatches (fc-01KRP*, fc-01KRQ*, fc-01KRS*) that consumed stale inline-sha256 archive bytes | **LOW** | Prior dispatches produced identical seed bytes (`hashlib.sha256(...)` and canonical helper byte-identical); no archive bytes changed; no re-eval needed. |
| Cargo-cult #6 NEW unwind helper LANDED | NEAREST policy preserves prior archive-byte parity | **LOW** | NEAREST stays default; no archive bytes changed; no re-eval needed. MODE arm requires future paired-CUDA RATIFICATION (~$0.06) which IS the empirical question. |
| Cargo-cult #4 (median aggregation) UNFIXED at helper surface | All NSCS06 v8 chroma LUT derivations to date | **MEDIUM-DEFERRED** | RECOMMENDED-WAVE-6 op-routable extends helper pattern to `build_chroma_lut_from_ground_truth`. No empirical re-eval pending; sister wave handles. |
| Pre-existing per-substrate symposium PROCEED_WITH_REVISIONS verdict | `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` | **NONE** | Verdict remains valid; Wave 5 lands within the 14-day symposium window (2026-05-21 → 2026-06-04 per Catalog #325). |
| 11 substrates in `.omx/state/probe_outcomes.jsonl` with NSCS06 v8 surface mentions | scan + classify | **NONE** | No surface affected by Wave 5 helper landings. Wave 5 ADDS NEW probe outcome; does not invalidate prior. |

## Conclusion

The Wave 5 landing introduces 2 canonical helpers + 1 canonical equation + 1 canonical anti-pattern WITHOUT invalidating any prior empirical anchor. The NEAREST byte-default policy preserves all prior dispatch artifact byte hashes. The cargo-cult #3 fix is byte-for-byte identical to the prior inline behavior (test `test_canonical_helper_matches_inline_sha256_byte_for_byte` is the regression guard).

NO historical verdicts require re-classification. Wave 6 op-routable (cargo-cult #4 extension) is the next-priority follow-up.

## Sister cross-references

- Companion canonical landing memo: `.omx/research/wave_5_nscs06_v8_cargo_cult_re_audit_plus_fix_landed_20260529.md`
- Predecessor 2026-05-26 cargo-cult audit: `.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Predecessor 2026-05-26 substrate design decision: `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md`
- Canonical equation #26: `src/tac/canonical_equations/procedural_codebook_savings.py`
- Wave 5 canonical equation: `cls_lowres_downsample_policy_boundary_preservation_v1`
- Wave 5 canonical anti-pattern: `cls_lowres_nearest_strided_without_empirical_vs_mode_v1`
- Wave 5 council deliberation anchor: `wave_5_nscs06_v8_chroma_lut_cargo_cult_re_audit_plus_fix_20260529`
- Wave 5 probe outcome: `wave_5_nscs06_v8_cargo_cult_re_audit_plus_fix_20260529` (PROCEED 30-day advisory)
