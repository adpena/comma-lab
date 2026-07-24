# FEED-603-MENU1-postcharter-addendum

date_utc: 2026-07-24
lane_id: lane_ddm_menu1_realized_flip_menu_20260723
research_only: true
score_claim: false
evidence_axis: `[macOS-CPU frozen-scorer advisory]`
pointer: `0.1910828242 [contest-CPU]`
pointer_moved: false
main_landing_review_required: true

## Trigger

The resumed MENU1 lane already had its 15,894-row compiler, five exact V19C
measurements, typed receipt, and three-pass review merged. The 2026-07-24
delegated authority added four later landed receipts: MC1, WS1, RD1, and E4.
This feed records their deterministic join without re-running or overwriting
the settled n600 evidence.

## DAG delta

```text
MENU1 v1 receipt ───────────────┐
MC1 static reassert receipt ────┤
WS1 seglex96 receipt ───────────┼─> postcharter compiler
RD1 v5 dual receipt ────────────┤      ├─ curve:v19c_menu1_joint
E4 Brotli Q11 receipt ──────────┘      ├─ curve:ws1_seglex96
                                       └─ curve:e_line_export
```

The three curves are intentionally separate. A row can telescope only inside
its `price_domain`; no post-coder E-line byte delta is applied to the V19C or
WS1 archives.

## Joined rows

1. **MC1 static reassert, MEASURED:** +1,747,057 realized Seg corrections for
   +139 counted bytes from the MENU1 joint parent, but the advisory joint
   objective worsens by +4.850055382139988 because pose collateral remains
   large. It is priced and waterfill-readable but not admitted.
2. **WS1 W_seg, MEASURED separate base:** d_seg
   0.024124510023328993 at 138,031 B with pose present. The curve remains
   2,709,004 errors above the #613 allowance; Road binds.
3. **RD1, DERIVED advisory prior:** 162 typed cells remain non-actionable.
   The three non-null aggregate exchange-rate rows are retained only as
   waterfill priors; they cannot authorize a train decision.
4. **E4 Brotli Q11, MEASURED E-line rate:** 344,203 B post-coder; −95,100 B
   versus E3. The semantic section saves 96,172 B (23.383924099262293%) while
   the chart costs 644 B. Decoded raw payload is byte-identical.

## Verdict and route

`MENU1_POSTCHARTER_JOINED_BOX_NOT_REACHED`, verdict scope FORMULATION. The
V19C/MENU1 joint curve is MyCar-bound; the WS1 seg-lexicographic curve is
Road-bound; the MC1 child remains MyCar-bound and is joint-negative. No curve
enters d_seg <= 0.00116, so this is not an R6 candidate.

FIRST-RUNG: measure one Fisher-margin-ranked, corrected-inner-Jacobian
actuator independently on each base curve, preserving exact parent custody.
Do not use Fourier residuals and do not mix base curves.

## Durable surfaces

- `.omx/research/configs/ddm_menu1_postcharter_addendum_20260724.json`
- `.omx/research/ddm_menu1_postcharter_addendum_20260724T051016Z/ddm_menu1_postcharter_addendum_receipt.json`
- `src/tac/optimization/ddm_realized_flip_menu.py`
- `tools/compile_ddm_menu1_postcharter_addendum.py`
- `.omx/research/ddm_menu1_postcharter_addendum_canonical_equations_20260724.md`
- `.omx/research/ddm_menu1_postcharter_addendum_directive_consumption_20260724.md`
- `.omx/research/codex_findings_ddm_menu1_postcharter_20260724_codex.md`

MAIN must review the full branch diff and independently rederive every joined
price, curve boundary, and false-authority label before merge.
