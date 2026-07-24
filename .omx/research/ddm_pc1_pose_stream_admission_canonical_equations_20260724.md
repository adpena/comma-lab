# Canonical Equations — DDM PC1 Pose Stream

Equation ID: `ddm_pc1_pose_stream_laws_v1`

## Smooth counted stream

For knot controls `q_k in Z^6`, coordinate scales `s`, `K=32`, and pair
`i in {0,...,599}`,

`u_i = i (K-1) / 599`,

`xi_i = (1-a_i) (s ⊙ q_floor(u_i)) + a_i (s ⊙ q_ceil(u_i))`,

where `a_i = u_i - floor(u_i)`. The four luma-phase controls use the same
interpolant. This makes all 600 pair values deterministic consequences of one
sub-kilobyte control curve.

## Multi-depth receiver

At scorer row `v`, camera height `h`, vertical focal length `f_y`, and horizon
`c_y`, continuous ground depth is

`D_ground(v) = clip(h f_y / max(v-c_y,1), h, D_far)`.

All decoded Movable pixels receive the depth at their contact row. This gives a
distinct parallax stratum rather than the previously tested one-depth plane.

The receiver law is

`x_0 = W_{-xi_i/2,D_i}(R x_parent,0) + P_4 r_i`,

`x_1 = W_{xi_i,D_i}(x_0)`,

followed by deterministic camera-grid bicubic realization and uint8 rounding.
`P_4` broadcasts the four luma residuals on the 2x2 stem phases. Adding the same
RGB scalar is luma-directed and chroma-null away from clipping.

Inactive packets bypass this law and return both parent frames byte-for-byte.

## Solved-plane descent target

Let `A` be the frozen evaluator bilinear resize and `YUV6` its exact BT.601
four-luma/two-chroma polyphase map. The target exposed to #366 is

`T_W = YUV6(A(decode(W)))`.

`T_W` is recomputed from the counted W parent and therefore costs zero target
bytes. It prevents PC1 from treating an arbitrary RGB program as pose truth.

## Pose metric and conditional action

For MS4d center `c_i`, landed factor `L_i`, and frozen PoseNet output `p_i`,

`Q_i = ||L_i (p_i-c_i)||_2^2`.

Observation of `Q_i` does not authorize a tube claim unless descent ran.

For an exact parent `W` and composition `C`,

`Delta S(C|W) = 100(d_seg(C)-d_seg(W))`

`             + sqrt(10 d_pose(C)) - sqrt(10 d_pose(W))`

`             + 25 (bytes(C)-bytes(W)) / 37,545,489`.

This is evaluated directly for each parent; intermediate deltas are never
telescoped.
