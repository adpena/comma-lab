# ddm_js1c — Stage-0 CUDA custody verdict: NOT_ADMITTED_RHO_GATE (2026-08-14)

One Modal T4 dispatch (fc-01M001XSKF61HCBVG8TBRFXBGG, ~$0.16, 15 min) bought the
custody-clean n600 SegNet argmax field for the T1R1 C1-composed candidate
(archive sha 12a5b181…, 187,046 B) on the matched worker-family instrument
(base cp135 34,970 flips). Status: component rows only, score_claim=false,
frontier unmoved. Store:
`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814/`
(STAGE0_RESULT.json sha 472fc816…; all four fields retained per the payload law).

## THE ROW (contest-CUDA T4, n600, deterministic)

| quantity | value |
|---|---|
| base flips (cp135) | 34,970 |
| candidate flips (T1R1) | **55,807** (+20,837 WORSE) |
| target flips (C1) | 27,330 |
| rho = (base−cand)/(base−target) | **−2.727** vs gate **0.827795** |
| determinism | fresh candidate field **BYTE-IDENTICAL** to the prior js1b run (sha 7c3750d9…) |

Verdict: **NOT_ADMITTED_RHO_GATE** (verdict_scope: INSTANCE — exact T1R1
archive 12a5b181 through runtime tree 3bb8da9f on the retained T4 batch-16
instrument). The local-renderer axis-mismatch is resolved: the prior js1b field
was NOT stale — the T4 instrument reproduces byte-for-byte across dispatches,
and the failure is REAL, not a custody artifact.

## Per-edge decomposition (the Road hub, m91)

Every major Road edge is NEGATIVE — the C1-composed overlay degrades the
partition it was meant to fix:
- Road↔MyCar: base 2,194 → cand 8,561 (rho undefined: target 7,483 ABOVE base)
- Road↔Lane: base 15,178 → cand 21,062 (rho −0.786)
- Road↔Undrivable: base 6,972 → cand 12,545

## ROUTING (pre-registered, executed at consume)

1. **V0–V5 ladder: FOLDED** (`V0_V5_DISPOSITION.json`) — do not execute for
   this T1R1 instance.
2. **#1043 trigger receipt EMITTED and SATISFIED** — the full-n600 CUDA
   Stage-0 decomposition IS retained, which fires js8's queued successor with
   the re-route the charter pre-registered: build the IMPLICIT
   decoder-derived edge-state conditioning consumer (no explicit edge mask
   shipped; consumer store `pr135_joint_solve_20260810/edge_conditioned/`).
   This is the trained-receiver (#982) / implicit-conditioning route — the
   explicit-overlay half of the js1 joint line is now closed on BOTH axes
   (local AND T4 custody), converging with fd135's finding that PR135's own
   explicit seg overlays were dead.
3. Convergence state: pk4 + ps135b + pz4a + js8 + js1c-explicit all terminal ⇒
   the remaining pose AND seg routes are **implicit joint conditioning
   (trained receiver)** and **coupled multi-token (#978)**.

## Apparatus banked

Modal module-identity law (fc-01M001GR1F postmortem, fix ac5d230137): a
dispatcher that reuses ANOTHER module's remote function MUST import it
BARE-first — `modal run` loads the entry by stem and images ship bare local
python sources; a package-qualified local import serializes an unresolvable
remote reference. Sister note: `modal volume get <vol> <file> <dest>` treats a
non-existent dest directory as the target FILENAME — pre-create the dest dir.

Lane closed terminal (`completed_js1c_t4_measurement_recovered`); call ledger
`harvested`. Modal ≈ $5.0/$20.
