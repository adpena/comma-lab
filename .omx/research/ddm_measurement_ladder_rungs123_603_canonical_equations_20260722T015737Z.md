---
title: DDM measurement ladder canonical-equations disposition
utc: 2026-07-22T02:25:04Z
task: 603
status: NO_NEW_CANONICAL_LAW_APPARATUS_ONLY
evidence_axis: "[macOS-CPU full-resolution real-plane apparatus]"
---

# Canonical-equations disposition

No canonical-equations registry row is appended because no scorer or contest law was measured.

For 32x32 chart `q` and RGB channel `c`, the deterministic fit stores the half-up integer mean

`m[q,c] = floor((sum_{p in q} Y[p,c] + 512) / 1024)`.

Each pair-plane stores a global anchor `a[c]`, integer row/column gradients `g_y[c], g_x[c]`, and one
residual per chart. The receiver evaluates

`Y_hat[q,c] = a[c] + round(g_y[c](2q_y-11)/22) + round(g_x[c](2q_x-15)/30) + r[q,c]`

and broadcasts this chart value to its 32x32 full-resolution support. Residual records are partitioned
by deterministic within-pair-plane variation rank into low/mid/high tertiles; no pixel payload exists.

The measured quantity bridge reports

`E_pixel = count_p[all_c Y_hat[p,c] = Y[p,c]] / count_p`,

`E_channel = count_{p,c}[Y_hat[p,c] = Y[p,c]] / count_{p,c}`,

`Delta_rgb_argmax = count_p[argmax_c Y_hat[p,c] != argmax_c Y[p,c]] / count_p`,

and `D_pose6 = ||pose_codes(z) - ordinal_n600(Pose6)||_1`.

At n256 the measured tuple is `(1,095,272, 0.001762549082, 0)`. The third coordinate is integer
ordinal-code debt, not `d_pose`; `Delta_rgb_argmax` is RGB-channel input apparatus, not SegNet
argmax. None of these values can rank, kill, promote, or move the contest pointer.

CONSUMED-BY: `ddm_describe_line_rate_distortion_bracket_v1` apparatus provenance; registration landing `.omx/research/ddm_structured_carriers_law_registration_20260722T142000Z.md`; MAIN review required.
