# Codex Findings — DDM PC1 Pose Stream Admission

`[macOS-CPU frozen-scorer advisory]` · `research_only=true` ·
`score_claim=false` · `pointer_moved=false`

## Disposition

PC1 is **ADMITTED AS A TYPED, COUNTED, DESCENT-TRAINABLE COMPONENT**. It is not
promoted, not tube-qualified, and not a score improvement. MAIN landing review
is mandatory.

The admitted packet is a 40-byte zero home with a 32-knot smooth twist curve,
four luma-phase residual controls, and 320 stable #366 coordinates. Its complete
nested compositions add 734 exact bytes to either W parent. Both archives parse
and re-emit exactly; inactive decode is byte-identical; each output effect has
one #417 owner; and a separate nonzero packet changes 410,468 composite-R cells
against the active zero-q home for each parent.

## Fresh n600/batch32 rows

| Exact parent | Parent `(d_seg,d_pose)` | PC1 zero-home `(d_seg,d_pose)` | Direct conditional Delta S |
|---|---:|---:|---:|
| W_seg | `(0.024124510023329, 146.364932454878)` | `(0.024602940877279, 163.049668916925)` | `+2.1700709084033565` |
| W_joint | `(0.070519231160482, 36.618184751411)` | `(0.024602398342556, 163.049745071518)` | `+16.652345570764727` |

These are direct n600, batch32 comparisons against the settled exact parent
rows. The negative result is only for the **zero-home / no-descent instance**.
It does not close the smooth ξ-stream formulation. MS4d observed zero tube
members for both parents, but `membership_claim=false`: no PC1 descent ran.

## Scorer-plane correction

The first completed pass used 600 independent pose rows, produced a 5,014-byte
packet, and initialized ξ from scorer-center directions. It measured
`Delta S=+2.156215331028895` on W_seg and `+16.63850501406813` on W_joint.
The task inbox then supplied the solved-plane existence proof: pose constraints
cost only 770 marginal bytes in W_joint versus W_seg, and smooth ξ should be
sub-kilobyte. That first pass is therefore retained only as a measured
coordinate-artifact upper bound.

The admitted replacement stores no invented pose solution. It exposes the exact
parent-derived solved-plane YUV6 values to #366 at zero target bytes and stores
only smooth descent controls. This is the optimal-form queue required before
any family-level negative:

1. descend the 32 ξ knots against the exact parent-derived YUV6 target and MS4d
   quadratic;
2. open residual phases only when ξ marginal value saturates;
3. waterfill counted coordinates until direct conditional Delta S reaches
   break-even;
4. rerun n600/batch32, then receiver-closed contest CPU/CUDA only if the
   advisory row becomes negative.

## Durable receipts

- admission:
  `.omx/research/ddm_pc1_pose_stream_admission_receipt_20260724.json`, SHA-256
  `df456b0a77504d55322e0230a47be5ba5289773556f00da9a9c419d51d8d9950`;
- packet: SHA-256
  `05a98facbd7f467f39c279c2e89e44a449cc9f11bbbdd4e0df031e2d4b222e9e`;
- W_seg composition: 138,765 bytes, SHA-256
  `123d1988a122ba56f301689c6d3cc0a1b51fe87da728577f3677883f3c6298c6`;
- W_joint composition: 139,535 bytes, SHA-256
  `25e4d7b56a718311e2dfc480acf03fde99f3f1c6542337f539e08880a4476652`.

Pointer remains `0.1910828242 [contest-CPU]`.
