# MENU1 post-charter canonical-equations note

date_utc: 2026-07-24
status: research-only equations leg
authority: `[macOS-CPU frozen-scorer advisory]`
score_claim: false

## 1. Curve-local telescoping law

For an exact parent `p` and child `c`,

```text
Delta_E(p -> c) = E_p - E_c
Delta_B(p -> c) = B_c - B_p
Delta_S(p -> c) = S_c - S_p.
```

A delta is composable only when both endpoints share the same `price_domain`
and `c.parent_candidate_id == p.candidate_id`. This is the executable
invariant enforced by `compile_postcharter_addendum`.

The three domains in this receipt are:

```text
V19C_MENU1_EXACT_CHAIN
WS1_SEG_LEXICOGRAPHIC_EXACT_CHAIN
E_LINE_EXPORT_ONLY.
```

No algebra authorizes deltas across these domains.

## 2. MC1 realized row

MEASURED endpoints:

```text
E_parent = 8,318,787
E_child  = 6,571,730
B_delta  = 139
S_parent = 26.28022355199344
S_child  = 31.13027893413343.
```

DERIVED:

```text
Delta_E = 8,318,787 - 6,571,730 = 1,747,057
Delta_S = 31.13027893413343 - 26.28022355199344
        = +4.850055382139988.
```

Therefore the row is a real Seg correction but is not admitted by the joint
objective. This is an INSTANCE result for the static-stored MC1 support, not a
family negative.

## 3. WS1 separate-base gap

MEASURED:

```text
B_ws1     = 138,031
E_ws1     = 2,845,843
d_seg_ws1 = 0.024124510023328993
d_pose    = 146.3649324958955.
```

With the #613 allowance `E_box = 136,839` and `d_seg_box = 0.00116`,

```text
E_gap     = 2,845,843 - 136,839 = 2,709,004
d_seg_gap = 0.024124510023328993 - 0.00116
          = 0.022964510023328992.
```

The per-class maximum is Road at 1,870,275 errors. WS1 is a base state, not a
delta to be summed with the V19C/MENU1 joint curve.

## 4. E4 post-coder price

MEASURED byte custody:

```text
B_e3               = 439,303
B_e4               = 344,203
Delta_B             = -95,100
semantic_before     = 411,274
semantic_after      = 315,102
chart_before        = 17,825
chart_after         = 18,469.
```

DERIVED:

```text
semantic_saving = 411,274 - 315,102 = 96,172
semantic_fraction = 96,172 / 411,274
                  = 0.23383924099262293
chart_delta = 18,469 - 17,825 = +644.
```

The coder is lossless and its decoded raw sections are byte-identical, hence
`Delta_E = 0` within the E-line domain. It does not define a byte delta for a
V19C or WS1 archive.

## 5. RD1 prior status

MEASURED/DERIVED receipt structure:

```text
typed dual cells = 162
actionable typed cells = 0
aggregate non-null priors = 3.
```

The aggregate `lambda_bytes_per_D` and
`marginal_D_reduction_per_byte` columns are advisory acquisition priors.
They do not become train-decision prices until the typed G4 class and shared
dimension-rate home are custodied.

## Epistemic status

- Source endpoints and section bytes: MEASURED in the SHA-pinned receipts.
- Differences, gaps, ratios, and binding maxima: DERIVED here and recomputed by
  the compiler.
- Cross-domain composition: FORBIDDEN, not unmeasured.
- Contest score/promotion: NOT CLAIMED; pointer unchanged.
