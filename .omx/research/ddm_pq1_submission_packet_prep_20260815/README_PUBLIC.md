# jg5_joint_waterfill — submission packet

This directory is a prepared submission packet held at
`submissions/jg5_joint_waterfill/`. It is **not submitted**. Every claim below is
backed by a retained custody receipt; those receipts are held outside the public
submission directory.

The one number this packet carries: on the exact bytes in this directory, the
measured `[contest-CUDA]` 600-sample score is **0.14839100138338618**.

## Evidence boundary

- **What is measured.** One exact evaluation: `upstream/evaluate.py` at
  `--device cuda` on a Tesla T4, Linux x86_64, over all 600 public samples,
  driven through this directory's own `inflate.sh` on this directory's own
  `archive.zip`. Score, both distortion legs and the rate leg all come from that
  run. The score published here is **recomputed from the reported components**,
  not read off the evaluator's rounded `final_score` field — that field prints
  `0.15`, a 2-decimal display that rounds up across the very boundary this
  submission sits on.
- **The precision of that claim.** The evaluator reports its distortion
  components at 8 decimal places, so the recomputed score carries a worst-case
  absolute error bound of `3.63296497868841e-06`. The claim is
  `0.14839100138338618 ± 3.633e-06`. The distance from that interval to 0.15 is
  about 443 times the bound, so the sub-0.15 statement is not a rounding artifact.
- **What is NOT measured for the possible final object.** The composed rider plus
  clean native-corrector runtime is locally decode-proven at full n600 and sealed,
  but it has no fresh contest-CUDA or contest-CPU receipt yet. Its score, timing
  and CPU boundary are therefore **GATED-ON-RC2**; none is transferred from jg5,
  an instrumented tree, or a contended local receiver.
- **The open runtime risk.** Populate the final runtime declaration only from the
  composed candidate's fresh `[contest-CUDA T4, n600]` receipt and fresh
  `[contest-CPU, n600]` attempt receipt. Until then no PASS/WARN/REFUSE timing
  verdict is made for the shipping object. The requested runner remains
  `linux-nvidia-t4`.
- **What is not authority.** Any local macOS number, any advisory row, and any
  projection appearing in our own research notes is not a score and is not used
  here.

## Exact identity

| Property | Value |
|---|---|
| Archive SHA-256 | `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` |
| Archive size | 180,625 bytes |
| Member | `p`, 180,525 bytes, stored, SHA-256 `54b445da3a1a4b4c7012c83b25c3e0d87daab5ce10cd54a1598cfb239ab05b4a` |
| Members in archive | 1 |
| Runtime tree SHA-256 | `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b` |
| Portable runtime content tree SHA-256 | `3ba9987771e1be967cf80942faedc7c5f6641f15039e03dd2b0909fd6613ab99` |
| Upstream snapshot SHA-256 | `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008` |
| Upstream `evaluate.py` SHA-256 | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` |
| Candidate seal SHA-256 | `96e9860aad9021e6dc9a9619036b54bd0a2205f60468e8585089db1d8044a7d0` (`SEAL_VALID`) |

The runtime tree hash is load-bearing, and the reason is measured. See "Runtime
tree pin" in `report.txt`: the previous generation's archive scored 79.40 under one
receiver tree and 0.157 under another, on byte-identical archive bytes. For this
receiver, the archive hash alone does not determine the score.

## What this submission is

The same inherited vehicle as the prior candidate. What changed is how the seg
edits and the pose carrier are decided — together instead of one after the other.

1. **The problem.** The prior composition applied all 573 seg token edits and then
   re-solved the pose carrier. Measured, that direction is seg-descending but not
   pose-null: the edits bought −0.012847 S on seg and cost +0.172 S on pose, a
   13.4× loss, and the composed result scored far worse than either part
   suggested. Solving one axis and then repairing the other is the failure; the
   two are one problem.
2. **Joint admission by waterfill (this candidate's mechanism).** Edit admission
   is swept over a Lagrange multiplier on pose damage instead of fixed in advance,
   so each edit is kept only if it pays for the pose it costs. **455 of the 573
   edits are admitted**; the other 118 are dropped and those pairs keep the prior
   carrier's codes.
3. **Carrier re-solve against this candidate's own renders.** The frame-0 pose
   carrier is re-solved against the edited renders this archive actually decodes
   to, not against the base renders, so the compensation is fitted to the state
   that ships. The descent uses a derived materiality stop rule instead of a fixed
   iteration budget: **600 of 600 pairs stopped on `no_improving_step`, with zero
   budget hits**, so the stopping criterion was never the binding constraint.

The net effect is the reverse of every prior generation in this packet: it
**spends** 3,443 bytes and buys both distortion legs, for a net `-0.008710980`.
The leg split is published in `report.txt`.

The tail sections, the HPAC stream and the 13-context fixed-point integer
log-odds mixer that produces the token stream are carried over from the prior
candidate.

**Read `BORROWED_SUBSTRATE_ACCOUNTING.md`, shipped in this directory, before
treating any of the learned content as ours.** Most of it is not: the semantic
renderer state and the pose carrier state originate in PR #130 and PR #135, and
the compressed model container, the HPAC probability object's architecture, the
residual payload and the range-coder backend all come from that lineage. This
candidate ships a lossy re-representation of their trained state, which raises the
attribution question without settling it, and the accounting says so. The same
table is reproduced inline in the pull-request body.

## Where this approach has headroom

**Rate is 81.05% of the score.** One archive byte costs `25 / 37,545,489 =
6.658590e-07` score units. On rate alone, 27,620 further bytes would reach 0.13 and
42,638 would reach 0.12.

**The archive is two sections.** Parsing the immediately prior 176,420 B body
through its own receiver gives: token stream 109,696 B (62.2%), three compressed
models 66,528 B (37.7%), fixed table 96 B, ZIP overhead 100 B. The token stream
runs at 0.0074392 bits per token over 117,964,800 tokens. The model half is
semantic 36,130 B, carrier 22,246 B and probability object 17,952 B before
compression. This budget is measured on the 176,420 B body, not on the shipped
180,625 B body.

**The probability-model axis paid three times after the coder search was believed
exhausted.** A fixed-point log-odds mixer measured −560.07 B, a widened causal
context −710.84 B, and a within-miss corrector −104.584 B. All three are
decode-identical: the decoded token field does not change, so neither distortion
leg can move.

**That axis is now measured as nearly spent.** An earlier framing called the miss
sector a "77,241 B reservoir". That quantity is an entropy and is overwhelmingly
irreducible; the arm that built the corrector withdrew the framing as a vacuous
denominator. Measured remainder on the current probability law: about 400 B on the
hit event and about 75 B in the within-miss sector, roughly `−4e-04` score units.
The recorded next step on this section is a change of structure, not of
calibration.

**The segmentation residual is measured as a tie, not a wall.** On the immediately
preceding vehicle, 99.22% of scored flips sit on the boundary of the transmitted
labels, while the label interior — 88.4% of the field — carries 7 flips in 104
million pixels, a boundary rate 203,000× the interior. 93.9% of flips have the
correct class one pixel away. At 98.3% the wanted class is already the runner-up,
at a median logit deficit of 0.105; 84.5% need under 0.3 logits and none needs more
than 3, against margins of 3 to 10 on correct pixels. One edge, Road against Lane
markings, is 43.4% of the total; three edges carry 80.4%. These shares are scoped
to the vehicle they were measured on.

**Label fidelity is measured at 4.1% of the debt or less.** 95.9% of the
segmentation leg is render and re-segmentation loss. The transmitted labels are
99.9985% correct against the ground-truth decode the scoring axis uses — 1,714
wrong cells in 117,964,800.

**The pose carrier is at a fixed point of its own basis.** The final solve stopped
on `no_improving_step` for 600 of 600 pairs, with zero budget hits and zero stops
at the materiality floor. The remaining `d_pose` is a property of the carrier's
basis and lattice rather than of the search.

**Floor.** No nontrivial proven lower bound exists for this problem; every proven
figure in the literature is an achiever upper bound. Our own floor ledger records
an ESTIMATE band of 0.07 to 0.13, rate-dominated.

## Directions we consider necessary or promising

Each item states the measurement that exists and the recorded next step. None is a
prediction of success.

1. **The third admission branch.** The shipped admission is two-way: keep the edit,
   or revert to the base frame. A third branch — drop the token — is specified and
   priced at `−0.002929` score units. It is **not shipped**: dropping a token is a
   receiver change, and the receiver this candidate ships has no such path, so
   implementing it would invalidate the byte-identity control chain the seal rests
   on. Recorded next step: a receiver revision, then a re-run of the same waterfill.
2. **The token-drop rung that was refused only on pose.** A rung-4 token drop
   measured a combined rate-and-segmentation gain of `−3.243e-3` score units with
   the rate leg exact. It was refused on its pose leg, and it named its own door:
   the compensation would have to cancel 99.807% of the pose perturbation. A later
   arm measured the same mechanism at 99.9874% cancellation, at 1.00× the same
   amplitude. Status: **REOPENED**, which means the negative no longer binds — not
   that the positive holds. The aggregate rests on 3 pairs, one of which reaches
   99.6705% alone; the two arms priced different archives (182,759 B and
   176,420 B); the carrier's re-coding cost across 600 pairs is unmeasured.
   Recorded next step: the carrier re-solve against the retained drop deltas on at
   least 60 seeded-random pairs, aggregated by ratio of sums, with its band
   reported.
3. **The model half.** 66,528 B — 37.7% of the archive — is three compressed
   models. The token stream's residual calibration is measured as nearly spent
   (above). The model half has not been treated as a representation. Recorded next
   step: none priced. This is the largest section with no design against it.
4. **Boundary placement in the renderer rather than after it.** The post-hoc form
   was built and measured: a correction channel on the free boundary band passed
   its coder gate at 32,270 real verified bytes, then measured realization
   efficiency `η = 0.6235` on 9 seeded-random pairs against a required 0.753, with
   0 of 9 above the bar, for a net of `+0.00252` score units. A flat repaint of the
   same band measures `+1.3808`. Recorded next step: pre-distortion in the
   renderer, which is item 5.
5. **The realization race.** The inherited renderer disagrees with its own
   transmitted token plane at `d_seg = 0.00029639352578669786`, or 99.9704%
   fidelity. Four treatments are specified against one common receiver, with every
   learned weight counted in the archive. Status: designed, unrun.
6. **Wider causal context.** The shipped wavefront assigns
   `group(x, y) = (x & 63) + 2·(y & 63)`; under it the up-right neighbour is
   98.6945% causal. Widening the template bought −710.84 B and is free receiver
   code. Recorded next step: further members from the same scan-order read.
7. **Decode cost.** A shipping constraint, not a score move. See the runtime risk
   above: the native port is measured and does not ship.

### Directions we measured and closed, with the mechanism that closed them

Negatives are signal, so each is stated with the reason it failed rather than as a
bare verdict. All are scoped to the formulation and vehicle named.

**Averaging in odds space.** Four context-mixture architectures over the shipped
correction law all lost, the best at +359 B. The mechanism is the hull inequality
below: every blend they could express was a weighted arithmetic mean, so none could
beat its best member. This closed the *formulation*, not the family — the
geometric-mean form built afterwards measured −560.07 B on the same field with the
same members.

**Mixing the neural prior's own log-odds as a member.** The highest-expected-value
hypothesis going into the mixer race, and it lost twice, at +552.32 B and
+253.28 B. The mechanism recorded is inferred rather than measured: the prior's
odds span roughly 1e-9 to 1e9, so any exponent other than 1 moves the confident
tail by a large factor, and 70% of the stream's bits live in that tail.
Recalibrating the *correction* is safe; recalibrating the *prior* is not. The
refusal is member-level, not family-level.

**A byte-carrying correction channel on the boundary band.** It passed its coder
gate at 32,270 real verified bytes against a 35,117 B bar, then failed on
realization (item 4 above). The sub-finding refuted our own prior guess that a
structured coder could halve the i.i.d. floor: the flips are isolated single
pixels, mean run length 1.110, so the best real coder beats i.i.d. by 2.5% and the
ceiling of *all* free conditioning on that channel is 12.2%. The channel is a
non-supplier because the geometry gives a coder nothing to exploit.

**Flat repaint of the free boundary band.** `+1.3808` score units, and worse than
repainting the whole frame flat. Mechanism: a local flat patch manufactures an edge
the scorer reads as real, so the repaint adds boundary where it removes it.

**Re-orienting the 12 stored basis dimensions.** A measured null — the reachable
pose correction is invariant to `1.9e-08` (machine precision, 24 random pairs)
under re-mixing. The same measurement corrected an earlier attribution of ours: the
reason a single-coordinate ±2 search could not reach the residual is not the basis.
The residual demands a **multi-coordinate** step of 57 to 14,079 int12 code units,
and the ±2 rail cannot travel that far. That is why the search was replaced with a
damped Gauss-Newton step on the same basis, lattice and bytes, which measured
`d_pose` → `6.993157e-06` at +9 B, with 204 of 600 pairs improved and 0 worsened.

**A semantic serialization split.** Worth −515 to −520 B when measured, then
dropped: the row-prune that landed alongside it changed the semantic body's length,
and re-measured on the edited body the split is negative. A byte credit measured in
one regime does not survive into another. Its receiver support ships and is inert.

**A lossless rider on the carrier's basis stream.** Measured at 183 B
(`−1.2185e-4` score units) against a budget of `−1.85e-4` — the same cross-regime
transfer failure, one memo downstream. It was then declined for a second,
independent reason: folding a lossless change still changes archive bytes, so the
resulting score would be derived rather than measured. It must be re-run last on
the final body.

**Search quality on the token codebook.** Both pathologies we suspected are present
and both are priced out: under-convergence is worth `3.238e-04` score units and
basin-trapping `6.104e-05`, so the token solver's total search headroom is about
0.06% of the gap it was measured against. A segmentation lever on that actuator
therefore has to clear that 0.06% headroom by roughly three orders of magnitude to
matter.

**Deep structured pruning.** Rate credit is measured and real — keep25 at −2,051 B,
`−1.3657e-3` score units. The projected pose leg is `+0.0264`, which is 2.8× the
gap it was measured against, in the wrong direction. That pose leg is unmeasured and the slope comes from n=2, so this
is closed pending a pose-sensitivity probe, not refuted.

**Finishing-stage distillation.** A 12-epoch mini-race selected an attack-weighted
distillation arm at `d_seg = 0.0050507`; over the matched long window it reversed to
`0.0054967` at slope `+1.37e-5` per epoch, against a control at `0.0051147` and
slope `−6.80e-6` — a deficit 12.8× the measured `2.99e-5` noise floor. Two things
closed: the formulation on that endpoint, and treating a short race as a verdict.

## Method: canonicalized move classes

Every mechanism that has moved this score belongs to one of five families. Each
family is one module with one accounting and one set of refusals. The families'
mathematics stays separate; the loop, the bookkeeping and the refusals are shared.

| Family | Mechanism |
|---|---|
| Realized-acceptance lattice descent | Propose a move, apply it, **decode for real**, accept only if the realized joint score improves. Stop when no neighbour improves. |
| Terminal joint compile | Segmentation edit, then carrier re-solve against the **edited** frames, then compensation, then rate re-encode, then container search. |
| Container and re-encode | Encoder-only knobs at the archive layer, with an identity control on every option. |
| Model-axis recoder | The probability model changes; the payload and the container are fixed. |
| Local authority instruments | Frozen scorer, declared ground-truth lineage, canonical score arithmetic, seeded pair draw. |

Measured worth, on the axis each was taken on. Descent: `d_pose` `7.769484e-06` →
`7.649247e-06` at zero archive bytes, 429 pairs improved and **0** worsened.
Container: −657 B and −105 B, `−5.074e-04` score units at zero distortion. Model
axis: −560 B and −104.584 B. Local instruments: 0.99993× of the contest axis on
pose, 0.99995× on segmentation. The five modules carry 239 tests.

### Realized acceptance, and why the rule is the whole safety argument

A proposal is accepted only if the realized joint score improves after a real
decode — not a surrogate loss, not a component read in isolation, not a
pre-round-trip field.

The consequence changes what else is allowed to be approximate. Once acceptance is
realized-only, every *proposal* mechanism becomes free to be heuristic, learned or
wrong — the worst a bad proposal can do is consume a realization. That is what
admits a learned ranker into a pipeline whose output must be exact, and it is
proved rather than argued: an adversarial ranker that orders the worst candidate
first still reaches the same optimum. The one honest caveat is truncation — with a
top-k cut, "no improving neighbour" becomes "no improving neighbour among the k
kept", so such a configuration carries a flag recording that its convergence proof
is weakened.

### Probabilistic micro-edits: log-odds mixing is a weighted geometric mean

Our micro-edit engine enumerates and prices small changes to the coded field. Its
four context architectures all lost, the best at +359 B. The cause is a theorem.

Every blend those architectures express is a weighted **arithmetic** mean of the
odds multipliers, and

```
min_k m_k  ≤  (Σ_k w_k m_k) / (Σ_k w_k)  ≤  max_k m_k
```

so a mean cannot leave the convex hull of its members. When one member is already
the best model available, blending it with weaker members moves the estimate away
from the answer. An average cannot beat its best member.

Context mixing that works adds in the log-odds domain, so members that agree
reinforce past the confidence of either one. That is a weighted **geometric** mean
of the odds multipliers,

```
m_mix = Π_k m_k ** w_k          (the weights need not sum to 1)
```

and because the weights are unconstrained it leaves the hull.

The classical realization reaches this through `log` and `exp`. Those are library
routines, they are not correctly rounded, and they differ by one unit in the last
place between platforms. One such unit at `p ≈ 0.5` moves an arithmetic coder's
integer frequency by 128 counts and desynchronises the decoder for the rest of the
stream. That failure is in our record at a score of 27.83.

Restricting the mixing weights to a dyadic grid `w = W / 2^b` turns the weighted
geometric mean into a **radical**:

```
m ** (W / 2^b)  =  m**k · Π_{i ∈ bits(j)} m ** (1 / 2^(i+1)),     W = k·2^b + j
```

Every factor on the right is reachable from `m` by repeated square roots and
multiplication. IEEE-754 requires `sqrt` to be correctly rounded; it does not
require that of `log`. The mixer computes a genuine weighted geometric mean and
stays bit-identical on every conforming platform. No `log`, `exp`, `pow` or `**`
appears on the decision path, and a test walks the module's syntax tree to enforce
it.

Two structural consequences are the controls. An integer weight costs no radical,
so a single member at weight exactly 1.0 returns the incumbent multiplier
bit-identically — the generalisation collapses onto the law it generalises at
0.000000 bits, and every delta afterwards is the mixture rather than the plumbing.
All weights at zero give exactly the uncorrected model.

The theorem showed up as a sign flip. One family cost +359.47 B under the
arithmetic mean and saves 36 B under the geometric mean; another cost +689.11 B and
saves 22.6 B. Same members, same estimator, same field.

The learned weights also measure the inherited law: its implicit weight of exactly
1.0 is wrong in most cells. The optimum ranges 0.23 to 1.87, so the correction is
over-applied about 4× in some regions and under-applied about 2× in others.

Where the bits are, over all 117,964,800 positions: 0.190% of positions carry
70.01% of the stream. Across four different prior families that **bit share** stays
between 68.92% and 76.56% while the miss **fraction** spans 0.190% to 50.48%, a
266× range. The share is the transferable quantity; the fraction is not.

**Selection on the scored clip.** The mixer's member set, context and learning rate
were chosen by racing on the video that is scored. The family is negative at 30 of
33 raced rows, and the smaller fallback configuration — one member, one context —
measures −340.82 B. Optimality on another clip is not claimed.

### Micro-edits priced by the exact inverse of the shipping decoder

A token edit's rate cost was previously modelled at a bits-per-token constant.
Modelling it moved one internal headline from `−0.0104` to a measured gap of
`0.006526`. Edits are now priced with a re-encoder built line-for-line from the
shipping decoder with decode replaced by encode, importing the model, the group
plan, the boundary map, the fixed table and the corrector from the shipped runtime.
It returns the exact archive delta for any edited token field, and its identity
control reproduces the shipped stream byte-for-byte at 109,696 B. At the scale
shipped here, edits cost 3.8373 measured bits per changed token.

### The seal

A candidate becomes fire-ready through one typed document pinning the archive and
member digests, the runtime file digests, the admission bar with its derivation,
the evidence axis and the pre-registered falsifiers. A validator re-derives every
pin from disk at consumption and refuses a paid evaluation on any drift; sealed
values are removed from the command line, so a hand-typed argument cannot
contradict the seal. Each field corresponds to a named failure in our own record.

## Repair and compensation

**Segmentation moves that reduce `d_seg` raise `d_pose`, and the measured sizes are
large.** On this candidate the segmentation token edits bought `−0.012847` score
units on segmentation and cost `+0.172` on pose, a 13.4× loss. 571 of 573 edited
pairs got worse on pose; 2 got better. Composing the finished segmentation solve
with the finished pose carrier produced `S = 0.3192` against a composition estimate
of 0.156.

An instrument control fixes the attribution: swapping only the odd frames back to
the base decode reproduces the banked `d_pose` to 6 significant figures, so the
467× move is the token edit's effect on frame 1 rather than a lineage or decode
change.

The same shape is recorded elsewhere: through the photometric frame, segmentation
token edits multiply `d_pose` by 387×, and direct quantization of a hard object
costs about 29× on pose.

**The mechanism, because it is what makes the repairs designable.** Two facts
combine.

*First, the two scorers share a resize.* The pose network resizes the frame
**first** and converts colour space **second**, to the identical target size the
segmentation network uses. Our own doctrine recorded this order backwards until it
was corrected at source. The consequence removes a whole class of tempting
argument: you cannot claim a perturbation is cheap for pose because it was built to
be invisible to segmentation *at a different resolution*, because there is no
different resolution. Both scorers read through the same operator.

*Second, a null space is a property of a lattice, not of a field.* The natural
design is to build a perturbation inside the resize operator's null space, so the
scorer that applies that operator cannot see it. That works within one frame. But
the pose scorer reads a **pair**, and the pose warp resamples frame 1's field onto
frame 0's lattice, where the operator no longer annihilates it. We measured the
survival directly: attenuation is only 1.662×, so the leakage arriving in the other
frame's scorer plane is **2.12× the debt being paid off**. Null-space membership
does not survive a change of lattice.

Together these say the coupling is structural rather than incidental. A direction
that descends segmentation is not pose-null, and no amount of care in constructing
the edit makes it so. That is the difference between a **penalty** and a
**projection**: adding a pose penalty to a segmentation objective trades the two
axes against each other at a rate you chose; projecting onto the pose-null
directions would remove the coupling, and the measurement above says those
directions do not exist in the frame-pair geometry. So the coupling has to be
*repaired after the fact*, and the repairs are what the rest of this section
describes.

Four repairs, in the order they run.

**Repair 1 — compensate inside the compile.** The structural reason this works is a
counting argument, and it establishes that the repair exists before any of it is
built. The pose scorer reads a pair of frames and produces **6 scored numbers**.
The carrier stores **12 free integer coefficients per pair**. So the system that
must be cancelled is underdetermined by a factor of two: for any leakage a frame-1
edit induces in those 6 outputs, there is generically a 6-dimensional family of
frame-0 coefficient moves that cancels it. The repair is not a search for a lucky
configuration; it is a solve of an underdetermined linear system, and the remaining
freedom is what pays for the lattice being integer rather than real.

Concretely: exact signed integer moves on frame 0 cancel a frame-1 edit's leakage
into the pose network's six scored outputs, solved as a Schur-coupled system and
folded into the lattice the carrier already uses. Measured pose-energy
cancellation: 99.995054%. Three independently solved pairs cancelled 98.33%, 99.94%
and 99.93% of their leakage energy `[macOS-CPU advisory]`.

The rate route matters as much as the algebra, and it follows from the same
observation. Carrying the correction as a new sidecar section costs about 7,000 B.
Expressing it as moves of coefficients the archive *already transmits* costs 41 B,
because only the residual entropy of the changed coefficients is new. A repair that
has to buy its own section is usually not worth its bytes; a repair routed through
existing coefficients usually is.

The compensation is bound to the object it was solved for, and this binding is
load-bearing rather than hygienic. A Schur solve is a linearisation about one
specific frame-1 token stream; move the tokens and the Jacobian it inverted no
longer describes the object. Each pair therefore carries a content fingerprint over
its index, its exact token bytes, its exact camera-resolution master and the
archive identity. The compiler re-checks that fingerprint before it touches the
code lattice, and a changed token object with a missing binding, a mismatched
master or a stale fingerprint fails closed. An earlier archive of ours is refused by
this rule, because its frame-0 compensation was solved for a different frame-1
token stream — the exact failure the rule exists to catch.

Credit: the edit-then-recompensate pattern is PR #135's. Ours is the fingerprint
binding, the frame-0/frame-1 disjointness argument, the step-matched Jacobian and
the rate route.

**Repair 2 — re-solve the carrier against the frames the edits produced.** With all
573 edits kept, re-running the pose carrier's own solver against the edited renders
takes `d_pose` from `3.268e-3` to `4.089e-4`, an 8.0× recovery at zero archive
bytes. On a three-pair test of the same mechanism the recovery reaches 1.073× of
the original, a 99.9874% cancellation.

The recovery is bimodal. One pair lands below its own unedited value; another
barely moves. That split is what the admission in Repair 3 acts on. An inherited
constant would have removed it: an earlier ceiling of 0.7347 on free pose
correction, measured where `d_pose` was already about `7e-6` and the lattice was the
binding limit, predicts at most 27% recovery here. It is a near-floor constant and
does not transfer to this amplitude.

**Repair 3 — admit jointly.** After the re-solve, keeping all 573 edits still
leaves `d_pose` at 58× the incumbent carrier's `d_pose` and `S` at 0.2023. The carrier cannot rescue
the full edit set, so the remaining work is selection: decide which edits to keep.

Each edited pair has exactly two states, and both legs of both states are measured.
**DROP** reverts frame 1 to the base render and the carrier to the incumbent codes,
so the pair costs no tokens and takes its base pose value. **KEEP** ships the edited
frame 1 and this arm's re-solved carrier. The DROP branch is a measurement rather
than a model: on all 8 unedited pairs checked, the odd frame is byte-identical
between the base and candidate decodes, while every edited pair checked differs — so
pricing a dropped pair at its base pose value is justified by bytes.

The selection cannot be a greedy ratio, and the reason is the shape of the score.
The pose leg enters as `√(10·d_pose)` over the *mean* across pairs, so per-pair pose
costs do not add in score units: the marginal score cost of one more unit of
`d_pose` **falls** as total `d_pose` rises. An exchange rate computed at the current
operating point is therefore wrong at every other operating point, and a
fixed-ratio greedy that admits pairs in benefit-per-damage order prices the last
admission at the first admission's rate. The concavity also has a useful sign — once
you are already paying the pose axis, further pose is cheaper at the margin than the
first unit, so a joint solve can admit edits that any per-pair rule refuses.

The admission therefore sweeps a Lagrange multiplier on pose damage and scores
**every** candidate subset through the exact contest formula, at matched batch
shape. Result: **455 of 573 admitted**, net `−0.00776976` against the prior pointer,
at zero counted archive bytes for the admission itself. Two arithmetic controls
anchor the sweep — drop-everything reproduces the pointer and keep-everything
reproduces the measured composite — and both offsets are quoted rather than folded.

The bimodality from Repair 2 is what makes the selection profitable: because
recovery is bimodal rather than graded, the sweep is separating two populations
rather than trimming a tail. Two of the largest recoveries, pair 240 at
`2.5345e-03 → 5.4316e-06` (467×) and pair 221 at `2.1402e-03 → 1.0997e-05` (195×),
both flip from DROP to KEEP once the carrier is refined.

**Repair 4 — derive the stop rule.** The inherited solver stopped after six
iterations. The accepted-step histogram is `[98, 131, 146, 111, 63, 35, 16]`, and
all 16 pairs at the cap accepted an improving step on every iteration and were still
improving by more than 2% on the last one. The replacement iterates while the
extrapolated remaining improvement, priced through the exact derivative of the
contest score and each pair's own measured geometric decay ratio, exceeds the
measurement band. The threshold is `5.588639e-09` in `d_pose` units, and no
hand-set tolerance remains in the stopping decision. The rule never bound: all 600
pairs stopped because the receiver refused every proposed step, with zero budget
hits.

Two composition laws govern how these combine, and they run opposite ways. On the
**rate** axis edit costs add: the union of three edit sets measured 1.0258× the sum
of its parts, and `10 + 6 + 14` changed tokens measured exactly 30. On the
**compensation** axis they do not: solving jointly beat the naive union by 3.705×.
A per-chunk rate may be summed; a per-pair compensation may not.

The pose forward is deterministic at a fixed batch shape and its value moves with
the shape — up to 7.7e-3 relative across shapes 1, 8 and 32 on the same pair. Every
keep-or-drop decision is therefore made on values re-measured at one declared shape
for both code sets.

## Realization walls and in-loop technique

**Realization here means the numbers the receiver produces.** A candidate must
survive the whole path: render at 384×512 float, bicubic resize to the 874×1164
camera grid, clamp and round to `uint8`, bilinear resize back to 384×512, then the
frozen scorer.

Three placements are enforced independently, and a candidate that misses any of
them does not reach an admission gate:

1. **The resize is in the loop.** Both resizes sit inside the autograd graph, with
   camera-resolution quantization between them.
2. **The `uint8` is in the loop.** The forward pass is an exact clamp and round;
   the backward pass uses a registered straight-through estimator. Float-only
   candidates are refused.
3. **The colour conversion is in the loop.** When pose is armed, a differentiable
   RGB-to-YUV6 feeds the real pose graph, and no `no_grad` or in-place clamp may
   sever the renderer gradient.

A candidate is invalid unless the training hard-forward's camera bytes equal the
public receiver's camera bytes on real frames, and the receiver's ordered argmax
vector equals the reference at the pinned batch shape.

**The measured walls.**

- **Representability does not imply survival.** A causal ladder separated exact
  semantic geometry from paint, camera placement, amplitude, resize and final
  argmax. The exact target grid reached `d_seg = 0.000282948812`; the repaired
  realized output stayed near `0.0274`. One horizon realized 5.2867% of its
  forecast cell gain. The ladder now runs before any weight changes.
- **Realization efficiency is measurable and below 1.** The one byte-carrying
  boundary candidate measured `η = 0.6235` on 9 seeded-random pairs against a
  required 0.753, with 0 of 9 above the bar.
- **The resize supplies exactly zero flips** on piecewise-constant content. The
  loss is manufactured in the paint.
- **Post-hoc repaint is closed.** A flat-anchor repaint of the free boundary band
  costs `+1.3808` score units, and is worse than repainting the whole frame flat,
  because a local flat patch manufactures an edge the scorer reads as real. Flat
  prototype paint reads back 35.4× worse than the trained render.

**Weight-space error is the wrong metric for choosing what to quantize or edit.**
Measured per tensor, the amplification from weight perturbation to *rendered*
damage spans about 15×: tensors named "dead" by weight-space error amplified at
**38,700×** while the "live" ones amplified at **2,518×**. A depth ladder ordered by
weight-space error therefore had its mildest rung 90× underwater on segmentation.
The contrast is measured on the contest axis: a recipe chosen by *rendered*
sensitivity survived at `d_seg + 2e-6`, while the same depth on the weight-error
"dead" tensors produced `+3.4e-3`. Targets are ranked by realized damage through
the render and scorer path.

## Upstream score dynamics, and the order of operations they set

The pipeline order below is derived from measured interactions between the axes,
not chosen. Four of those interactions set it:

- Edits are a pose actuator: 13.4× loss on this object.
- Rate costs of independent edits add (union over sum, 1.0258). Compensation costs
  do not (joint beats union by 3.705×). So one may be summed per chunk and the
  other may not.
- A global cascade can be a credit, so a dependency argument bounds where an effect
  reaches, not how large it is.
- Pose's estimate band is a median 13.4× segmentation's at equal sample count,
  wider in 100.0% of 2,000 shuffles — and prefix bias inverts by axis, with the
  first pairs of this clip 2.54–4.21× harder on pose and 0.95–0.97× *easier* on
  segmentation. Subsets are therefore seeded and stratified, never prefixes.

**The order, and the reason each step holds its place.**

1. **Measure the real rate first.** On the token axis rate is the binding leg: at
   one receipt the segmentation leg was `−2.712674e-5`, the pose leg `+1.126177e-7`,
   and the rate leg `+5.127114e-5` — rate was 455× pose.
2. **Solve segmentation** against the frozen scorer, with realized acceptance and
   ground-truth targets on the lineage the scoring axis uses.
3. **Re-solve the pose carrier against the edited renders.** Composing two finished
   candidates gave 0.3192 where the composition estimate was 0.156.
4. **Compensate inside the compile**, bound by content fingerprint, before the
   carrier is re-coded. A compensation solved for a different token stream is
   refused.
5. **Admit jointly** by waterfill over the pose multiplier, scoring every subset
   through the exact formula, because the pose leg is concave and a fixed ratio is
   wrong at the margin.
6. **Re-encode the tail** with the exact inverse of the shipping decoder, so the
   rate leg is measured rather than modelled.
7. **Search the container last.** It acts on section bodies that are already decided
   and restores each byte-for-byte, so both distortion legs are zero by
   construction.
8. **Seal, then fire once.**

Steps 3 to 5 follow 2 because edits move pose. Step 6 follows 5 because an edit's
rate depends on which edits survived. Step 7 is last because it is the only step
whose distortion legs are free. Steps 6 and 7 may be summed per chunk; steps 4 and
5 may not, because the two axes obey opposite composition laws.

## Two names a reader will meet in the receiver

Both are cosmetic and neither changes behaviour.

- `CP135` and `F26` appear in `inflate.sh` as an error string, environment
  variables and file names. They are internal codenames for the inherited
  PR130/PR135 lineage, kept because renaming them would change the evaluated
  runtime-tree hash.
- `inflate.sh` carries a `Darwin` branch that calls `brew --prefix libomp`. It is
  **unreachable on the contest runner**: it requires `F26_TOKEN_DECODER` to equal
  `native-hpac`, and the script defaults that variable to `python` with nothing
  setting it otherwise. This submission assumes Linux.

## Dependency closure

`inflate.sh` is **not fully self-contained**:

- It requires **`Brotli==1.2.0`** exactly. If the interpreter does not already
  have that version, the script calls `uv pip install --only-binary :all:
  "Brotli==1.2.0"`, which **reaches the network at decode time**. If `uv` is
  absent the script exits 69 — it fails closed, but it does fail.
- It invokes a **C compiler** (`${CC:-cc}`) at decode time to build
  `runtime/entropy/rc64_backend.c`. The compiler is assumed present on the runner.
- Otherwise: PyTorch and NumPy, both already required by the evaluator itself.

The declared-dependency approach follows the precedent set by earlier accepted
submissions in this contest, which likewise declared a pinned Brotli. It is
flagged because "no network at decode time" is a reasonable thing for a judge to
expect, and this submission does not meet it.

## How to verify

From this submission directory. These commands verify the current hosted jg5
object; if RC2 replaces it, update every URL and expected identity from the fresh
receipts before running them.

```bash
shasum -a 256 -c MANIFEST.sha256
# expect 33 lines ending in: OK

ARCHIVE_URL=https://raw.githubusercontent.com/adpena/comma-lab/2d61b51988799ec3561d5f8a6f659aeb88cc99d9/submissions/robust_current/jg5_sub015_runtime/runtime/archive.zip
LOCAL_SHA=$(shasum -a 256 archive.zip | awk '{print $1}')
HOSTED_SHA=$(curl -fsSL "$ARCHIVE_URL" | shasum -a 256 | awk '{print $1}')
test "$LOCAL_SHA" = "$HOSTED_SHA"
test "$LOCAL_SHA" = f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e

T4_RECEIPT_URL=https://raw.githubusercontent.com/adpena/comma-lab/2d61b51988799ec3561d5f8a6f659aeb88cc99d9/submissions/robust_current/jg5_sub015_runtime/t4_receipts/MODAL_REMOTE_RESULT.json
curl -fsSL "$T4_RECEIPT_URL" | python3 -c 'import json,math,sys; r=json.load(sys.stdin); s=100*r["avg_segnet_dist"]+math.sqrt(10*r["avg_posenet_dist"])+25*r["archive_size_bytes"]/37545489; assert r["n_samples"]==600 and r["gpu_t4_match"] is True and r["score_axis"]=="contest_cuda"; assert r["expected_archive_sha256"]=="f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e" and r["expected_runtime_tree_sha256"]=="2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b"; assert abs(s-r["score_recomputed_from_components"])<1e-15; print(f"recomputed_score={s:.17g}")'

bash ../../evaluate.sh --submission-dir . --device cuda
```

Expect `Average PoseNet Distortion: 0.00000637`, `Average SegNet Distortion:
0.00020139`, and `Final score: 0.15` at the evaluator's 2-decimal display. The
score claimed here, `0.14839100138338618`, is those components recomputed. The
shipping runtime and its budget verdict remain GATED-ON-RC2.

## Reproduction

The end-to-end rebuild entry point has **not been re-run for these bytes, and it
cannot rebuild them.** That entry point rebuilds the token stream and carries the
other seven sections through verbatim; this candidate's chain also re-decides
content in sections it copies — the seg token edit solve, the edit splice, the
admission waterfill and the pose-carrier re-solve. No configuration closes that
gap, so the script refuses this archive by name and names the builders that do
produce it.

`compress.py` and `COMPRESS.md` ship in this directory anyway, because a
submission you cannot inspect is a submission you cannot trust. `compress.py`
rebuilds the token stream exactly and carries the other seven sections through
byte-identically; run it against this archive and it refuses **by name, before
doing any work**, naming the four stages it does not express and the builder that
performs each one. `compress.py provenance` emits the full lineage with a SHA-256
for every input.

What does exist for this candidate is the seal binding archive to receiver, the
staging proof that this directory is byte-identical to the evaluated tree, and the
authority receipt.

**Source is available.** All **33 of the 33 files** enumerated by this candidate's
evaluated runtime manifest, plus the exact `archive.zip`, are in version control at
`submissions/robust_current/jg5_sub015_runtime/runtime/` in
<https://github.com/adpena/comma-lab>, each byte-identical by SHA-256 to the tree
the score was measured on — the receiver modules, `inflate.py` and `inflate.sh`
included. An earlier draft of this README disclosed that 24 of those 34 had no
source in version control; that was true when written and has since been closed.
The shipped bytes are pinned by hash and the decode is deterministic, so what a
judge runs is fully determined either way.
