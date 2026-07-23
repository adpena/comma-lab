# DDM v19c correction-saturation canonical equations

`research_only=true`

`execution_allowed=false`

`score_claim=false`

`evidence_axis=[macOS-CPU frozen-scorer advisory]`

## Receiver-priced authority

For an exact receiver-closed state `x`,

```text
S(x) = 100 d_seg(x) + sqrt(10 d_pose(x)) + 25 B(x) / 37,545,489.
```

For current stack `s_k` and proposal `p_k`,

```text
Delta S_k = S(compile(s_k plus p_k)) - S(compile(s_k)).

s_(k+1) = s_k plus p_k   iff Delta S_k < 0
          s_k             otherwise.
```

`compile` includes the archive, receiver, one final uint8 realization, frozen
SegNet, official-YUV6 PoseNet, and exact archive length. Family-local Fisher or
source priors only rank proposals inside a family. They never replace the
cross-family realized `Delta S_k` verdict.

## Recursive proposal stream and saturation

Let `I=(p_0,...,p_(M-1))` be the deterministic interleaving of all six typed
families. The recursive stream revisits coordinates without changing order:

```text
proposal(k) = apply_scale_cycle(I[k mod M], floor(k/M)).
```

The stopping counter is

```text
f_(k+1) = 0       if Delta S_k < 0
          f_k + 1 otherwise,
```

including infeasible and over-budget proposals as failures. With `K=64` and
v19b archive size `B_0`,

```text
stop iff f_k >= 64
     or B(s_k) - B_0 >= 200,000.
```

This is a coordinate-stream saturation certificate, not a proof that no
unrepresented correction family can improve the instance.

## Typed correction state

The v19b state is extended additively:

```text
s = (
  Delta q_template,
  Delta q_sparse,
  track translations,
  pair-lifecycle translations,
  Q8 region directives,
  template permutation,
  grammar-reference substitutions
).
```

Inverse-solved row-band direction `r` uses the sealed candidate endpoint
relative to the sealed origin:

```text
Delta q_r = q_r - q_origin,
q_next = project_wire(q_current + alpha Delta q_r).
```

Regional pre-uint8 corrections are summed in Q8 and rounded once:

```text
q8(kind,pair) = sum_j alpha_j
                  1[scope_j contains (kind,pair)] Delta q_405(kind,pair)

camera_u8 = clip(floor((256 camera_compact + q8 + Bayer8) / 256)).
```

Worldsheet events add integer translations only to their named track or pair
lifecycle, subject to the exact polygon feasibility bounds:

```text
g_next(t,p) = g_current(t,p) + (dx,dy).
```

Template swaps act on the current template state, and grammar substitutions
replace the named reference while retaining parse-back and receiver closure.
Every actuator is therefore applied to the current accepted joint state rather
than scored as an independent alternative.

## Fisher-margin ordering

For DEV pair `p`, frozen top-1/top-2 margin `m_i`, error indicator `e_i`, and
Lane prior weight `w_i=1.25` (`1` otherwise),

```text
EV_pair(p) = sum_(i in p) e_i w_i / max(m_i, 1e-8).
```

This implements the registered Fisher-margin ordering within pair-bearing
families. Exact receiver-priced `Delta S` remains the admission authority.
No Fourier residual basis is used.

## DEV-to-n600 replay

Let `A_DEV` be the ordered DEV admissions. n600 begins again from the exact
v19b n600 state and replays only that ordered list:

```text
t_0 = v19b_n600
t_(j+1) = t_j plus A_DEV[j] iff
          S_n600(compile(t_j plus A_DEV[j])) - S_n600(compile(t_j)) < 0.
```

Every n600 decision is measured over all 600 pairs in preserved 16-pair
batches. For decoded receiver support `supp(p)`, a batch `b` outside the
support reuses the current exact scorer row:

```text
b intersect supp(p) = empty  => camera_b(t_j plus p) = camera_b(t_j).
```

For a support-bearing batch, reuse is allowed only after exact NumPy array
identity. Changed batches receive a fresh frozen-scorer forward. After the
sequential decisions close, the final archive is strict-decoded and all 600
pairs are independently replayed; every camera/cell/Pose digest and error row
must equal the assembled endpoint. DEV admission never implies n600 admission.

## Bucket attribution and c1 handoff

For role bucket `C={Lane,Movable}` and residual bucket
`R={Road,Undrivable,MyCar}`,

```text
helpful_b = count(error_before and correct_after and label in b)
harmful_b = count(correct_before and error_after and label in b)
net_flips_b = helpful_b - harmful_b
Delta errors_b = -net_flips_b.
```

The sealed continuous c1 in-box debt is `3,103,688.832`, the integer target is
`136,839`, and the residual after perfect role-bucket closure is `2,377,273`.
The measured n600 endpoint reports both correction buckets separately; it
does not infer independent credit for #366.

## Atom-order gauge

For payload atom permutation `pi`,

```text
Delta B_order(pi) = B(encode(pi(atoms))) - B(encode(atoms)).
```

An order change is a rate actuator only when the encoded byte count changes
and the remapped receiver emits bit-identical camera bytes. A fixed-width
payload with `Delta B_order=0` remains a measured no-op.

## Verdict scope

These equations govern only the SHA-bound v19c instance and its exact
macOS-CPU frozen-scorer advisory DEV/n600 measurements. They do not establish
a contest-CPU/CUDA score, global family optimum, promotion, or pointer
movement.

## Measured endpoint

The preregistered recursive DEV stream stopped at

```text
M_unique = 231 coordinates
k_stop = 1,002 proposals
admissions_DEV = 153
f_stop = 64 consecutive failures
Delta B_budget_DEV = 200,000 B (not exhausted).
```

Sequential exact n600 replay yielded

```text
admissions_n600 = 104
measured_nonnegative_rejections_n600 = 47
compile_infeasible_failures_n600 = 2

B_final = 137,827 B
d_seg_final = 0.024786978828
d_pose_final = 163.061210029156
Delta B_vs_v19b = +2 B
Delta S_vs_v19b = -0.18073912464057892.
```

Its exact component decomposition is

```text
100 Delta d_seg = -0.180744595
Delta sqrt(10 d_pose) = +0.000004138641514828123
25 Delta B / 37,545,489 = +0.0000013317179062443428.
```

The c1 incremental error reduction is

```text
role gain = 697,039 - 658,180 = 38,859
residual gain = 2,440,167 - 2,265,811 = 174,356
total gain = 213,215
residual fraction = 0.8177473442300026.
```

Strict final replay over 38 independent 16-pair batches matched every
sequential acceptance row, with digest
`3efcc943769209a5393ee03ee59b80d4287b689d16f03dfcc29589f52ded6cc3`.
