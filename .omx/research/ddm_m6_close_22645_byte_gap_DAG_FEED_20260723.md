# FEED — DDM M6 close the 22,645-byte gap

`research_only=true` · `$0` · no launch · no scorer run · no config change ·
`score_claim=false` · pointer unchanged · MAIN landing review required.

## Executable evidence DAG

```text
M4 receipt: baseline 177169 / cap 154524 / gap 22645
      │
      ├── #580 ker(A) measured counted hideable bytes = 0
      │       └── P_NULL_GAUGE joint credit = 0
      │
      ├── exact #575 FP11/CTXR archive
      │       ├── strict legacy parser
      │       ├── implicit framing packer
      │       ├── retained-field mutation/consumption proof
      │       └── exact legacy member + ZIP reconstruction
      │               └── P_TEMPORAL_DESCRIPTION joint credit = 13
      │
      └── g2g2 full receipt
              ├── 13/13 prefixes factor2_uint8_exact
              ├── different 121128-byte vehicle
              ├── 0/6 admitted
              └── P_REALIZE transferable byte credit = 0

three joint pool credits + one exact compact artifact
      └── compose_gap_closure
              ├── Y = 13
              ├── final = 177156
              ├── residual = 22632
              ├── sub015_reached = false
              └── candidate/eval edge remains closed
```

The deterministic receipt binds the M4 and g2g2 receipts, source archive, source
member, compact member/archive, every retained semantic section, and reconstructed
archive by SHA-256:

`.omx/research/ddm_m6_close_22645_byte_gap_20260723_receipt.json`.

## Triality

- **DAG:** this file owns the exact source-to-pool-to-composed-verdict dependency
  graph and the closed candidate/eval successor edge.
- **DSL:** N/A with rationale. This lane adds no trainer lever, launch flag,
  curriculum state, or submission config. The generic adapter is a deterministic
  packet function; inventing a typed training flag would add false state.
- **Equations:** `tac.canonical_equations.ddm_m6_gap_closure_20260723` owns typed
  joint-pool credits, the one-final-artifact admission rule, strict cap, residual,
  and sub-0.15 predicate.

Canonical composition:

```text
Delta B_p = joint receiver-closed credit for pool p
U = sum_p Delta B_p
Y = B0 - B_final if final_same_artifact_receiver_closed else 0
require 0 <= Y <= U
R = max(0, B0 - Y - B_cap)
sub015_reached iff B0 - Y <= B_cap
```

For `B0=177169`, `B_cap=154524`, and pool credits
`P_REALIZE=0`, `P_TEMPORAL_DESCRIPTION=13`, `P_NULL_GAUGE=0`:
`U=Y=13`, `B_final=177156`, and `R=22632`.

## Unified-stack wire-in disposition

1. **Sensitivity map:** no new pixel, pair, gradient, or scorer response was
   measured. Existing ker(A) and g2g2 receipts are consumed by hash; no new
   sensitivity row is authorized.
2. **Pareto constraint:** the hard rate residual is updated from 22,645 B to
   22,632 B only for the structural adapter. The distortion coordinates are reused,
   not remeasured.
3. **Bit allocator:** only exact one-artifact byte deltas enter. Nullity and
   scheduled score debt cannot be converted into byte allocation credit.
4. **Cathedral/autopilot:** no candidate or dispatch hook is enabled because the
   177,156 B structural form remains above the 154,524 B cap. MAIN review is the
   only successor edge.
5. **Continual learning:** the dated findings memo permanently records the
   13-byte framing boundary and the g2g2/n16 premise conflation.
6. **Probe disambiguator:** the deterministic derivation plus retained-field
   mutation proof arbitrates “generic framing” versus “video-derived content” and
   fails closed on magic, version, lengths, section isolation, ZIP metadata, receipt
   SHA, or pool-bound drift.

## Exact blocker

`SUB015_NOT_REACHED_Y13_RESIDUAL22632`.

No delegated lever supplies the remaining 22,632 receiver-closed bytes. This is
scoped to the exact #575 FP11/CTXR vehicle and the three delegated levers. It is not
a global MDL lower bound, not a representation-family death verdict, and not a
statement that other description-line work cannot close the gap.

## MAIN landing review edge

MAIN must verify the framing classification, exact archive reconstruction,
g2g2/n16 scope separation, zero ker(A) credit, pool composition, test receipts, and
the absence of score/promotion/dispatch claims before landing.
