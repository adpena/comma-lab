# Canonical equations note — Task #574

For the settled quantized coherent-slot Lane description `Q_t` and decoded corrected composed screw `xi_t`, this measured planar projection is

`Qhat_t = quantize(Advect(dequantize(Q_{t-1}), rho_z(t), rho_x(t), omega_y(t)))`,

`e_t = Q_t - Qhat_t`, with `e_0 = Q_0`, and

`Q_t = Qhat_t + e_t` at decode.

The identity control is `Qhat_t = Q_{t-1}`. Signed innovations use zigzag plus self-delimiting uvarint, then the measured xi-derived context partition and `tac.shared_pmf_model` range coding. Reconstruction of the settled `(Q,presence)` lattice is exact, so predictor error affects rate only. The other three screw coordinates are counted but are not predictor actuators in this formulation.

The counted description objective for the Lane section is

`B_XTDL1 = B_xi + B_presence + B_model + B_range + B_framing`,

and archive authority is

`A(theta) = len(DetZip9(SerializeS4(M(theta), seed, base(theta), causal, events, components)))`.

The shortcut `451191 - 216207 + D_new` is invalid because 216,207 B is only `base+components` and outer deflate is non-additive.

MEASURED n600 anchors:

- `B_terminal(identity LBND2) = 35,393 B`.
- `B_terminal(identity + counted xi context) = 42,413 B`.
- `B_terminal(planar-3 xi predictor) = 43,901 B`.
- `D_base+components: 216,207 → 224,715 B`.
- `A: 451,191 → 460,168 B`.

Therefore, on this formulation,

`B_planar3_xi > B_identity_xi_context > B_LBND2`

and the strict admission condition `Delta B < 0` fails. Verdict scope is **FORMULATION**, not family: a chart whose ego degree of freedom has not already been absorbed can reopen the inequality.

Authority: `[macOS-CPU advisory]` exact bytes and repository decode, not a contest score. Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN landing review required.
