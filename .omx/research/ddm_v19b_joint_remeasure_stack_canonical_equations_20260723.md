# DDM v19b joint-remeasurement canonical equations

`research_only=true`

`execution_allowed=false`

`score_claim=false`

`evidence_axis=[macOS-CPU frozen-scorer advisory]`

## Pure-priced authority

For a receiver-closed state `x`,

```text
S(x) = 100 d_seg(x) + sqrt(10 d_pose(x)) + 25 B(x) / 37,545,489.
```

For current stack `s` and candidate move `m`,

```text
Delta S(m | s) = S(compile(s plus m)) - S(compile(s)).
admit(m | s) iff Delta S(m | s) < 0.
```

The 405 winner is the preregistered start `s_0`. The nine remaining source-v19
winners are ordered by their independently measured single-step `Delta S`,
then applied recursively:

```text
s_(k+1) = s_k plus m_k   if Delta S(m_k | s_k) < 0
          s_k            otherwise.
```

Every `compile` above emits one exact archive and invokes the receiver, uint8,
R, frozen SegNet, official-YUV6 PoseNet, and archive-length terms together.
There is no additive score approximation.

## Common-master merge algebra

The stack has three typed stages:

```text
s = (q_template, q_sparse, g_track, q8_preuint).
```

Compact post-quantization moves add candidate deltas relative to the sealed v17
origin, followed once by wire-domain projection:

```text
q_template = clip_[0,255](q0_template + sum_i Delta q_template_i)
q_sparse   = clip_[-127,127](q0_sparse + sum_i Delta q_sparse_i).
```

For a grammar move `(axis, sign)`, only tracks that were feasible for that sign
in the sealed source carrier receive the integer translation. Per-track
translations then add across moves:

```text
g_track(t, axis) =
  sum_i sign_i * 1[axis_i = axis]
                   * 1[t active on the bound screen]
                   * 1[sign_i in exact bounds(t, axis)].
```

The cumulative translation must remain inside the exact polygon bounds for
every lifecycle. This retains the source-v19 proposal semantics at the
boundaries; a global all-track shift would be a different proposal.

All admitted Q8 variants name the same correction field at the same stage.
Their union is therefore summed in Q8 and rounded once:

```text
q8_preuint = sum_i scale_i * Delta q_405
camera_u8  = clip(floor((256 camera_compact + q8_preuint + Bayer8) / 256)).
```

Repeated dither/uint8 wrappers are forbidden because they would change the
application-stage contract.

## Non-additivity accounting

For source single-step gain `g_i = max(0, -Delta S_i(single))` and measured
conditional gain `h_i = max(0, -Delta S(m_i | s_i))`,

```text
survived_i = min(g_i, h_i)
degraded_i = max(0, g_i - h_i)
amplified_i = max(0, h_i - g_i)
survival_fraction_i = h_i / g_i.
```

For the nine conditional moves:

```text
sum g_i         = 0.03742127019557973
sum survived_i  = 0.03591006678308081
sum degraded_i  = 0.001511203412498918
sum amplified_i = 0.08049672121725288
survived / sum g = 0.9596164586450241.
```

The telescoping conditional gains, not `sum g_i`, produce the final stack.

## c1 bucket attribution

Let `C={Lane,Movable}` and `R={Road,Undrivable,MyCar}`. For bucket `b`,

```text
helpful_b = count(error_before and correct_after and label in b)
harmful_b = count(correct_before and error_after and label in b)
net_flips_b = helpful_b - harmful_b
Delta errors_b = -net_flips_b.
```

At n600:

```text
net_flips_C = 29,377
net_flips_R = 73,945
net_flips_total = 103,322.
```

The exact correction-line rates are

```text
net_flips_per_added_byte = 103,322 / 3,884
                         = 26.60195674562307
bytes_per_net_flip       = 3,884 / 103,322
                         = 0.037591219682158686.
```

The residual above c1's integer target is

```text
3,137,206 - 136,839 = 3,000,367 errors.
```

If the first 16,384-byte downstream budget had to close that residual alone,
its required exact yield would be

```text
3,000,367 / 16,384 = 183.12786865234375 net flips/B.
```

J3's later requirement is conditional on exact v18b replay; independent credits
must not be summed.

## Atom-order rate gauge

The current template payload uses six fixed-width 2x2 RGB records in a
ZIP-stored member:

```text
B(as_emitted) = 140
B(canonical_order) = 140
Delta B_order = 0.
```

This is a zero rate actuator in the current coder. A future order-sensitive
delta/entropy code may optimize the ordering only if it remaps every counted
template index and proves bit-identical receiver camera bytes.

## Verdict scope

The equations govern only the SHA-bound v19b instance and its n64/n600
macOS-CPU frozen-scorer advisory measurements. They do not establish a
contest-CPU/CUDA score, family optimum, promotion, or pointer movement.
