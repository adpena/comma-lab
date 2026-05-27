<!-- HISTORICAL_PROVENANCE: APPEND-ONLY per Catalog #110/#113. Retroactive sweep per Catalog #348 for newly-landed Catalog #367 (inflate raw bytes fail-open). -->
<!-- HISTORICAL_SCORE_LITERAL_OK: this memo cites historical Cascade C' fc-call_id anchors; no NEW score literal claims. -->

# Retroactive sweep for Catalog #367 — inflate raw bytes fail-open (frame-count / resolution mismatch)

**Date:** 2026-05-27T00:29:31Z
**Per:** CLAUDE.md "Operator gates must be wired and used" non-negotiable + Catalog #348 "EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP" self-protection.

## 1. Bug-class symptom signature

A substrate `inflate.py` file emits raw bytes that don't match the contest output contract:
- Expected: 3,662,409,600 bytes (1164 × 874 × 1200 × 3 = RGB × full-resolution × 600 pairs × 2 frames-per-pair)
- Actual: any other byte count (resolution mismatch OR frame-count mismatch)

Per `experiments/contest_auth_eval.py::_run_inflate` the upstream check is strict — any mismatch raises:
```
RuntimeError: [inflate] WRONG-SIZE .raw file(s): 0.raw=<actual>B (expected 3662409600B).
Each must be 3,662,409,600 bytes (1164x874x1200x3). Likely truncated mid-decode.
```

The bug-class symptom: inflate.py emits raw bytes via `.write_bytes()` / `.tofile()` / `_write_sparse_zero_raw()` / equivalent without a fail-closed check (`if raw_bytes != CONTEST_RAW_BYTES: raise AssertionError(...)`) BEFORE the write.

## 2. Pre-fix window

The bug-class drift was empirically demonstrated **once** in the recent session: Cascade C' WAVE-3 Modal T4 dispatch `fc-01KSKB4B30DCYTCP883XYV5BNV` (2026-05-26 18:55:39, rc=0 elapsed=11.62s) failed at stage 7 contest_auth_eval with:
```
RuntimeError: [inflate] WRONG-SIZE .raw file(s): 0.raw=707788800B (expected 3662409600B).
```
Empirical ratio: 707788800 / 3662409600 = 19.33%. Root cause: WAVE-3 inflate.py at commit `aaf0b1eb6` emitted 1200 frames at **384×512×3** resolution (1200 × 384 × 512 × 3 = 707,788,800) instead of 1200 frames at **1164×874×3** (the contest output resolution). This was a resolution mismatch, not a frame-count mismatch.

Fix landed Cascade C' commits `5bcb53070` ("Fix Cascade C prime full-frame inflate scaffold") + `d0c4517ea` ("Audit repair waterfill queue integration"); current `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py` at HEAD has `CONTEST_OUT_H=874`, `CONTEST_OUT_W=1164`, `CONTEST_NUM_FRAMES=1200`, `CONTEST_FRAME_BYTES = CONTEST_OUT_W * CONTEST_OUT_H * 3`, `CONTEST_RAW_BYTES = CONTEST_FRAME_BYTES * CONTEST_NUM_FRAMES`, plus the fail-closed check:
```python
if raw_bytes != CONTEST_RAW_BYTES:
    raise AssertionError(f"contest raw byte contract drifted: {raw_bytes}")
```

## 3. Historical-KILL/DEFER/FALSIFY search results

Searched repo for all inflate.py files under `submissions/*/inflate.py` (excluding `submissions/exact_current/` per CLAUDE.md mutation frontier) + `src/tac/substrates/*/inflate.py` that emit raw bytes AND reference the contest output contract.

Live count at landing: **5 pre-existing files** without fail-closed check:
- `src/tac/substrates/pr101_lc_v2_clone/inflate.py`
- `src/tac/substrates/pr95_lora_dora/inflate.py`
- `src/tac/substrates/sabor_boundary_only_renderer/inflate.py`
- `submissions/factorized_hnerv_v1/inflate.py`
- `submissions/pr103_pr106_final_runtime/inflate.py`

These 5 files predate the Catalog #367 landing. They reference `1164` (the contest width) or `CONTEST_RAW_BYTES` and emit raw bytes via `.tofile()` or `.write_bytes()` but lack an explicit fail-closed check on the byte count.

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion":** these 5 files are NOT killed; they are deferred for backfill. The Catalog #367 wire-in is **WARN-ONLY initial** per CLAUDE.md "Strict-flip atomicity rule" — strict-flip pending operator-routed backfill sweep that EITHER adds the canonical fail-closed check to each file OR adds a substantive `# INFLATE_FRAME_COUNT_FAIL_OPEN_OK:<rationale>` waiver documenting why the file is exempt (e.g., delegates to a sister module that does the check).

**Cross-checked all 4 historical KILL / DEFER / FALSIFY memos:** None of the 5 currently-flagged files have historical KILL / DEFER / FALSIFY verdicts. They are L1 / L2 SCAFFOLD / RESEARCH-only lanes that never reached paid-CUDA promotion (per the Catalog #220 / #240 lane registry classification).

Search of empirical anchors database (.omx/research/*falsified*.md + *killed*.md + *deferred*.md):
- 0 hits cite raw-byte-mismatch symptom signature pre-WAVE-3 anchor
- 0 hits cite the WAVE-3 anchor (it is empirical evidence for a NEW bug class, not a historical verdict)

## 4. Per-finding RE-EVAL-priority assignment

| Historical Finding | RE-EVAL Priority | Rationale |
|---|---|---|
| `pr101_lc_v2_clone/inflate.py` (no historical kill/defer/falsify; L1 scaffold) | LOW | L1 scaffold predating Catalog #367; backfill candidate for operator-routed sweep. Per Catalog #220 + #240 the lane is research_only / substrate_engineering. |
| `pr95_lora_dora/inflate.py` (no historical kill/defer/falsify; L1 scaffold) | LOW | Same as above. |
| `sabor_boundary_only_renderer/inflate.py` (no historical kill/defer/falsify; L1 scaffold) | LOW | Same as above. |
| `submissions/factorized_hnerv_v1/inflate.py` (no historical kill/defer/falsify; submitted-runtime variant) | MEDIUM | If this submission directory is ever dispatched at full Modal CUDA, the missing fail-closed check could produce a WRONG-SIZE crash post-train. Operator-routable: add fail-closed check OR explicit waiver per Catalog #367. |
| `submissions/pr103_pr106_final_runtime/inflate.py` (no historical kill/defer/falsify; submitted-runtime variant) | MEDIUM | Same as above. |
| Cascade C' WAVE-3 IMPL-LEVEL falsification (commit 39e1db080) | RESOLVED | Fix landed at commits 5bcb53070 + d0c4517ea; current inflate.py at HEAD passes Catalog #367 check. Per Catalog #307 paradigm-vs-implementation classification: this was IMPL-LEVEL (resolution mismatch), NOT paradigm-level refutation of the Cascade C' Atick-Redlich doctrine. |

## 5. Cross-references

- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable
- CLAUDE.md "Forbidden score claims" + "Apples-to-apples evidence discipline"
- Catalog #146 sister gate (contest-compliant inflate runtime template at the contract-template surface; #367 is the runtime-effect surface)
- Catalog #205 (canonical select_inflate_device)
- Catalog #220 (substrate L1+ scaffold operational mechanism)
- Catalog #272 (distinguishing-feature integration contract)
- Catalog #295 (PYTHONPATH self-containment for submissions/*/inflate.py)
- Catalog #307 paradigm-vs-implementation falsification classification
- Catalog #348 retroactive verdict-taint sweep discipline
- Catalog #287 placeholder-rationale rejection
- Cascade C' WAVE-3 verdict commit `39e1db080` + fix commits `5bcb53070` + `d0c4517ea`
- contest auth_eval inflate-side check at `experiments/contest_auth_eval.py::_run_inflate` (canonical raw-byte size validator)

## 6. Discipline declarations

- Catalog #229 PV: full audit of all 120 inflate.py files (78 submissions + 42 substrates) at preflight wire-in time
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is NEW; zero mutations
- Catalog #287 substantive-rationale rejection — placeholder literals rejected throughout
- Catalog #348 4-field contract: bug-class symptom signature ✓ + pre-fix window ✓ + historical search results ✓ + per-finding RE-EVAL-priority assignment ✓

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
