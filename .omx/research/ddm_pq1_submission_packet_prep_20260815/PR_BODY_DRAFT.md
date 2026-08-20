# submission name: jg5_joint_waterfill

> **STATE: PREPARED, NOT SUBMITTED — delete this block before posting.** No
> archive has been hosted, no pull request has been opened, and the score below is
> not published anywhere. Hosting the archive and opening the PR are the
> repository owner's to do.
>
> **THIS FILE IS SOURCE MATERIAL, NOT THE BODY TO PASTE.** The contest's
> coding-agents-and-LLMs policy names "write full PR description and public facing
> comments" as a banned use, and four PRs were closed under it on 2026-08-07. This
> draft was produced with agent assistance, so **the repository owner writes the
> final description**, using this file as verified source for the numbers, the
> receipts and the boundaries. Two things the owner also owes and no draft can
> supply: an honest answer to the "most of the code" test, and a disclosure of the
> LLM setup — the policy's own optional bullet invites it, and disclosing is the
> difference between compliance and concealment.
>
> Structure the final body the way PR #135 was asked to: **this** is the baseline
> submission and its score, **this** was changed and it achieved **this** score,
> and **this** is what did not work better. Precise, not verbose. The
> "Baseline, change, score" block below is written to be lifted in that shape.

# upload zipped archive.zip

**Download:** not yet hosted; no URL is claimed here. The exact bytes are pinned:

| Property | Value |
|---|---|
| SHA-256 | `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` |
| Size | 180,625 bytes |
| Members | 1 — `p`, 180,525 bytes, stored, SHA-256 `54b445da3a1a4b4c7012c83b25c3e0d87daab5ce10cd54a1598cfb239ab05b4a` |

The submission is only valid against runtime tree
`2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`. For this
receiver the archive hash alone does not determine the score: the previous
generation's bytes scored 79.40 under one receiver tree and 0.157 under another.
See "Runtime tree pin" in the report below.

# report.txt

```text
=== Exact result identity ===
Evidence axis: [contest-CUDA]
Hardware: Tesla T4, Linux x86_64
Samples: 600
Archive SHA-256: f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e
Archive size: 180625 bytes
Member: p, 180525 bytes, stored, SHA-256 54b445da3a1a4b4c7012c83b25c3e0d87daab5ce10cd54a1598cfb239ab05b4a
Runtime tree SHA-256: 2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b
Portable runtime content tree SHA-256: 3ba9987771e1be967cf80942faedc7c5f6641f15039e03dd2b0909fd6613ab99
Upstream snapshot SHA-256: cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008
Upstream evaluate.py SHA-256: 7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b

=== Evaluation results over 600 samples ===
Average PoseNet Distortion: 0.00000637
Average SegNet Distortion: 0.00020139
Seg contribution: 0.020139
Pose contribution: 0.007981227975693965
Rate contribution: 0.1202707734076922
Recomputed score: 0.14839100138338618
Reported (2 dp display): 0.15
Report-8dp worst-case absolute score error bound: 3.63296497868841e-06
Inflation wall time: 1419.9042126240001 seconds
Evaluation wall time: 51.427507448999904 seconds
Total authority wrapper wall time: 1484.80307526 seconds
Budget: the 1800 s CI limit covers the WHOLE job, not inflation alone -- see below.

The printed "0.15" is a 2-decimal DISPLAY of the evaluator's own final_score field.
The score claimed on this submission is the value recomputed from the reported
components, 0.14839100138338618, which is what the line above records. The
display rounds up across the 0.15 boundary; the components do not.

=== Relationship to the prior candidate ===
Prior packet generation 4 measured S = 0.15710198138050818 at 177182 bytes on the
same axis and the same hardware class. This archive is +3443 bytes and
delta S -0.008710980 against it.

This candidate SPENDS rate to buy distortion, the opposite direction from every
prior generation in this packet. The measured leg split against generation 4:
  rate +2.2926e-03 (+3443 bytes)
  seg  -1.0170e-02
  pose -8.3353e-04
  net  -8.7110e-03
The legs sum to the net. They are DISPLAYED at 5 significant figures, so adding
the printed strings need not reproduce the net digit for digit.

Sign determinacy: the net is a DELTA between two independently-8dp-rounded rows,
so both rows' error bounds apply and they ADD -- 3.336608e-06 + 3.632965e-06 =
6.969573e-06. The net is 1249.86x that summed bound. One row's bound alone gives a
2x larger margin; the sign of the net is the same either way.

Like generation 4 and unlike generation 3, this candidate does NOT hold decoded
state constant: both distortion legs move, and here they move in our favour while
the rate leg is paid.

=== Why the admission has to be joint ===
Segmentation token edits are a pose actuator, and the size is measured: the full
edit set bought -0.012847 score units on segmentation and cost +0.172 on pose, a
13.4x loss, with 571 of 573 edited pairs worse on pose and 2 better. Both scorers
read through the SAME resize, so an edit cannot be made cheap for pose by hiding
it from segmentation. The repair runs after the fact: re-solve the carrier against
the edited renders, then admit jointly by sweeping a Lagrange multiplier on pose
damage and scoring every candidate subset through the exact contest formula.
455 of 573 edits are admitted, net -8.7110e-03 against the prior row, at zero
counted archive bytes for the admission itself.

=== Runtime tree pin ===
These archive bytes have been evaluated on contest-CUDA T4 exactly ONCE, under
runtime tree 2103073d..., which is the tree shipped here. No superseded row
exists on these bytes.

The pin is load-bearing, and the reason is measured, not theoretical.
The previous generation's archive was evaluated TWICE on identical bytes and
scored 79.40216174747616 under one receiver tree and 0.15710198138050818 under
another. Arithmetic decoding under a mismatched probability model does not error:
it returns rc=0 and emits wrong symbols from the first divergent bin onward, so
the decode "succeeds" and produces garbage frames, and structural parse-back
cannot see it because sections and hashes round-trip correctly either way.

For this receiver the archive hash alone does NOT determine the score. A score
claim on these bytes is valid only against runtime tree 2103073d..., the tree
shipped here and the one the authority receipt validated.

=== Evaluation-time budget: a disclosed live risk ===
The official evaluation has a 30-minute limit (upstream/README.md:114), and the
CI job carries that limit as timeout-minutes: 30 on the WHOLE job
(upstream/.github/workflows/eval.yml:30) -- not on inflation alone.

Measured on the authority run: inflation 1419.9 s plus evaluation 51.4 s =
1471.3 s of the 1800 s job wall.

We grade that measurement in THREE FRAMES, and publish all three. They are one
measurement seen three ways, not three results:

  A. Charged total (inflation + evaluation) against our canonical residual window
     for the remaining CI steps on the CUDA path, [822, 1302] s. 1471.3 s is
     169.3 s OVER the ceiling. Verdict REFUSE.
  B. Inflation alone against that same window re-derived with the MEASURED
     evaluation time in place of the estimated one, [890.6, 1430.6] s. 1419.9 s
     fits, by 10.7 s. Verdict WARN.
  C. Absolute job wall. 1471.3 s of 1800 s leaves 328.7 s of headroom for
     checkout, dependency installation and archive download.

Frames A and B are related exactly by
  [890.6, 1430.6] = [822, 1302] + (evaluate_est 120...180 - evaluate_measured 51.4)
so they do NOT disagree: B is A corrected for evaluation coming in well under its
own estimate -- 51.4 s measured against a 120-180 s allowance, i.e. 2.3x to 3.5x
under. An earlier draft of this packet described them as two
windows that disagree and were unreconciled. That was wrong, and it is withdrawn:
there is one derivation and two framings of it. Quoting either one alone hides
the evaluation correction, so both are stated. Only the 1471.3 s is measured; the
residual window for the non-inflation steps is our own projection in both frames.

Where the time goes: token decode is 1341.5 s of the 1419.9 s inflation, 94.5%
(95.72% against the 1401.58 s instrumented-stage sum). One hot stage, not diffuse
overhead.

We built the obvious cure, measured it on the contest axis, and it lost. A native
port of that stage's integer half reproduces this candidate's decode BIT-FOR-BIT:
the same 3,662,409,600-byte raw output by SHA-256, the same distortion components,
the same score. On local hardware it runs 1.77-1.83x faster -- a range that
straddles the 1.804x our own bar requires, and our own receipt declines to call it
a PASS because run-to-run variance exceeds the distance to the bar. On a contest
T4 the sign INVERTS, and both denominators are worth stating because they differ:
on the TOKEN STAGE the port took 1546.6 s against 1341.5 s for the stage it
replaces, 15.3% slower (1546.6/1341.5); on WHOLE INFLATE the ported tree took
1612.6 s against 1419.9 s for the tree shipped here, 13.6% slower (1612.6/1419.9),
because the rest of inflate is unchanged and dilutes the stage ratio. Either way
the sign is negative. The split moves the decode off the GPU onto host
vCPUs far weaker than the laptop cores the local number came from. So the port
does NOT ship, and lowering this stage to native CPU is closed as a wall-clock
lever on this hardware. That measurement cost no score: the row is byte-identical
to the one claimed above.

A second port -- lowering the decode-time corrector as well -- moves still more of
the same work onto the same host vCPUs that just lost by 205 s, so it inherits
that result rather than escaping it. It is UNMEASURED on the contest axis: the
shipping report emits no native-versus-Python sub-stage split, so this row cannot
be decomposed to say how much of the token stage is still Python. Pricing that
split on the contest axis is the next measurement. No build claim is made here
before it exists.

On the CPU path the same assessment projects 1414-1913 s of inflation against a
residual of [1044, 1332] s, which is over budget in every corner. The prior
lineage recorded contest-CPU inflation at 3422.711146813 s against the 1800 s
budget; that figure is INHERITED from an earlier body and is NOT measured on these
bytes. No contest-CPU row exists on this archive at all. This candidate ships the
same token decoder, so the CPU path is expected to remain infeasible, and that
expectation is inherited too.

This is the single largest risk on this submission. It is a runtime risk, not a
correctness or score risk: the score above is measured on the exact submitted
bytes.

=== CPU boundary ===
Status of the [contest-CPU] axis on these exact bytes: NO ROW EXISTS. No CPU
score exists and none is claimed. This submission is GPU-required for evaluation;
the requested runner is linux-nvidia-t4.

=== Provenance ===
Candidate seal: jg5_joint_waterfill_455, seal SHA-256 96e9860aad9021e6dc9a9619036b54bd0a2205f60468e8585089db1d8044a7d0
Seal validation at fire time: SEAL_VALID
Torch (auth wrapper): 2.5.1+cu124; CUDA 12.4; driver 580.95.05; Tesla T4 confirmed
Evaluation environment: torch 2.9.0+cu128, torchvision 0.24.0+cu128, numpy 2.3.4, timm 1.0.22
Source commit pinned into the eval container: 56e239829091e56ced913b464f3a6d4e9d5127c5
Provider job identifiers are retained privately with the authority receipts and
are deliberately not reproduced on this public surface.
```

# eval host info

Modal, Tesla T4 (confirmed by the harness, not assumed), Linux x86_64, driver
580.95.05, CUDA 12.4. Evaluation environment: torch 2.9.0+cu128,
torchvision 0.24.0+cu128, numpy 2.3.4, timm 1.0.22. All 600 public samples.

# build cost info

No model was trained for this candidate; it starts from an already-trained
inherited model state. The work that produced these bytes is a compile-time solve
— edit admission plus a carrier re-solve — run on local hardware. The
exact-evaluation row is a single T4 run of about 25 minutes.

# does your submission require gpu for evaluation (inflation)?

**Yes. Requested runner: `linux-nvidia-t4`.** The GPU path is measured but tight,
so please read this before scheduling the run.

`inflate.py` performs a neural render, so this submission is GPU-routed. On the
authority run, inflation took 1,419.9 s and evaluation 51.4 s = 1,471.3 s of the
1,800 s job wall.

We grade that in three frames and publish all three, because publishing only the
friendliest one is exactly the thing we would not want done to us:

| Frame | Reading | Verdict |
|---|---|---|
| **A** | charged total 1,471.3 s vs our residual window `[822, 1302] s` | **REFUSE**, over by 169.3 s |
| **B** | inflation 1,419.9 s vs `[890.6, 1430.6] s` (same window, measured evaluation time) | **WARN**, fits by 10.7 s |
| **C** | 1,471.3 s of the absolute 1,800 s wall | 328.7 s headroom |

A and B are one derivation, not two: `[890.6, 1430.6] = [822, 1302] + (evaluate_est
120…180 − evaluate_measured 51.4)`. Only the 1,471.3 s is measured; the residual
window is our projection in both frames.

Token decode is 1,341.5 s of that 1,419.9 s, **94.5%** — one hot stage. We built
the port, and on the axis that matters it lost. It reproduces this candidate's
decode **bit-for-bit** (identical 3.66 GB raw output by SHA-256, identical score),
runs 1.77–1.83× faster locally — a range that straddles the 1.804× our own bar
requires, and our receipt declines to call it a PASS because run variance exceeds
the distance to the bar — and on a **contest T4 it is slower on both
denominators**: **15.3% slower on the token stage it replaces** (1,546.6 s against
1,341.5 s) and **13.6% slower on whole inflate** (1,612.6 s against 1,419.9 s), the
smaller figure because the rest of inflate is unchanged and dilutes the stage
ratio. The split moves decode onto host vCPUs much weaker than the laptop cores.
**It does not ship**, and that measurement cost no score because the row is
byte-identical to the one claimed here.

Porting the decode-time corrector as well would move more of the same work onto
those same vCPUs, so it inherits that result. It is unmeasured on the contest
axis and we make no claim for it.

**These exact bytes have not been measured on a contest CPU** and no CPU score is
claimed. The 3,422.7 s contest-CPU inflation sometimes quoted for this lineage is
**inherited from an earlier body, not measured on these bytes**; this candidate
ships the same token decoder, so we expect the CPU path to remain over budget, and
that expectation is inherited too.

Full numbers are in the report above under "Evaluation-time budget".

# did you include the compression script? and want it to be merged?

**Included — `compress.py`, with `COMPRESS.md` beside it. Merging is your call, and
you should read the boundary first, because it does not rebuild these bytes.**

`compress.py` reconstructs an archive from pinned inputs and refuses to exit 0
unless the rebuilt bytes hash to the pinned SHA-256. Run it against **this**
archive and it refuses **by name, before doing any work**, and prints the four
build stages it does not express plus the script that performs each one. That
refusal is deliberate: a script that runs, produces different bytes and reports
success is worse than no script.

What it does express: it rebuilds the **token stream** — replaying the shipped
decode order, re-encoding the token field under the free decode-time corrector,
splicing the new stream in and repacking — and carries the other seven sections
through byte-identically by construction. For a candidate inside that grammar the
byte-close is genuine and end-to-end.

What it does not express is this candidate's own chain, which re-decides content
in sections the script only copies: the segmentation token edit solve, the edit
splice, the admission waterfill and the pose-carrier re-solve. Those are missing
**stages**, not missing options, so no configuration closes the gap and the script
says so rather than offering a flag that cannot help. `COMPRESS.md` names the four
builders that do produce them.

**We claim no verified end-to-end rebuild for these bytes.** Training the
checkpoint is upstream of all of this and is documented rather than re-run:
`compress.py provenance` emits the lineage with a SHA-256 for every input.

# changes from upstream

None. The pinned upstream snapshot
(`cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`,
`evaluate.py` `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`)
is unmodified. The scorer was not touched.

What changed is on our side. Every mechanism that moved this score sits under one
rule: a proposal is accepted only if the **realized** joint score improves after a
real decode. The token mixer is a weighted geometric mean in log-odds reached by
repeated square roots rather than `log`/`exp`, so it stays bit-identical across
platforms. The resize, the `uint8` round and the colour conversion are enforced
**in loop, independently**, and the pipeline order is derived from measured axis
interactions rather than chosen.

# competitive or innovative?

**Competitive, on one measured row.**

On the exact submitted bytes the measured `[contest-CUDA]` 600-sample score is
`0.14839100138338618`, recomputed from the reported components. That is below the
best ranked leaderboard score as read on 2026-08-20 (PR #135,
`semantic-pose-HPAC_CPR1_polished`, 0.162) and below every prior row in our own
custody.

Four qualifications:

1. **The printed score is `0.15`; the claim is `0.14839100138338618`.** The
   evaluator prints a 2-decimal display that rounds up across exactly the
   boundary this submission sits on. The claim rests on the components.
2. **The claim carries a bound.** The components are reported at 8 decimal
   places, giving a worst-case absolute score error of `3.633e-06`. The claim is
   `0.14839100138338618 ± 3.633e-06`; the distance to 0.15 is about 443× that
   bound, so the sub-0.15 statement is not a rounding artifact.
3. **The improvement is a re-decision over borrowed content, not a new model.**
   No artifact was trained for this candidate. Ours is the decision rule; it
   operates on PR #130 / PR #135's trained state. The accounting table below says
   which is which, section by section.
4. **One axis is measured, one is not.** `[contest-CUDA]` is measured on these
   bytes. `[contest-CPU]` is not, and we grade the GPU-path evaluation-time
   budget WARN ourselves.

Where the headroom is, measured: rate is **81.05%** of this score, and **37.7%** of
the archive is three compressed models. The token stream's residual calibration is
measured as nearly spent; the model half has no design against it. The reason a
joint solve is needed at all:
segmentation edits cost **13.4× more pose than they buy segmentation**, so the
result is admissible only because of the compensation and joint admission, not
because the edits are cheap.

# additional comments

## Baseline, change, score

**Baseline.** Our own prior candidate on this vehicle: **S 0.15710198138050818 at
177,182 bytes** `[contest-CUDA T4, n600]`. It applied all 573 seg token edits and
then re-solved the pose carrier. Those edits bought −0.012847 S on seg and cost
+0.172 S on pose — a 13.4× loss.

**Change.** Edit admission and the pose carrier are now solved **jointly**, in the
compile stage rather than one after the other.

- *Admission* is swept over a Lagrange multiplier on pose damage and every
  candidate subset is scored through the exact contest formula, so an edit is kept
  only if it pays for the pose it costs — **455 of 573 admitted**. A greedy
  benefit-per-damage rule cannot do this: the pose leg enters as `√(10·d_pose)`
  over the mean, so per-pair pose costs do not add in score units and a ratio
  computed at one operating point is wrong at every other one.
- *The carrier* is re-solved against **the edited renders this archive actually
  decodes to**, not against the base renders, so the compensation is fitted to the
  state that ships. The descent uses a derived materiality stop rule instead of a
  fixed iteration count: **600 of 600 pairs stopped on `no_improving_step`, zero
  budget hits**, so the stop was never the binding constraint.
- Both run inside the same solve, before the tail is re-encoded, because an edit's
  rate cost depends on which edits survived.

**Score.** **S 0.14839100138338618 at 180,625 bytes**, same axis: **+3,443 bytes**
for a net **−8.7110e-03**. It spends rate and buys both distortion legs, the
reverse of every earlier candidate here. Leg split and error bounds are in the
report above.

**What did not work better.** The three-way `{edit, drop, keep}` solve shipped only
two branches — `drop` needs a receiver change this body has no path for. A
12-dimensional pose-basis re-orientation is a measured null: re-mixing the basis
leaves the reachable correction invariant to 1.9e-08, so it ships nothing. And the
native token-decode port, which is byte-identical and 1.77–1.83× faster locally, came
back **15.3% slower on a contest T4** on the token stage it replaces (1,546.6 s
against 1,341.5 s), so it does not ship either.

**Priced and unbuilt.** A third admission branch that drops the token outright is
worth `−0.002929` score units and is blocked on a receiver path this body lacks;
a reopened token-drop rung is worth `−3.243e-3` on a 3-pair aggregate whose
carrier re-coding cost is unmeasured.

## Borrowed-substrate accounting

Most of the learned content in this archive is **not ours**, and the packet ships a
full table (`BORROWED_SUBSTRATE_ACCOUNTING.md`). Summary:

| Section / mechanism | Class | Note |
|---|---|---|
| Semantic renderer state | mechanism-adopt-with-attribution | PR #130/#135 trained values, lossily re-represented in our format. **Not byte-identical to theirs after decode.** |
| Pose carrier state | mechanism-adopt-with-attribution | Their solver form, our binding, their lattice re-solved |
| Compressed model container | inherited-substrate | PR #130/#135 |
| HPAC probability object | inherited-substrate | Architecture PR #130/#135 |
| Compensation blob | mechanism-adopt-with-attribution | Edit-then-recompensate is PR #135's pattern |
| Residual payload + table codes | inherited-substrate | PR #130/#135 |
| RC64 token stream | mechanism-adopt-with-attribution | Model-axis work ours; coder theirs |
| RC64 backend (encoder + shipped receiver) | inherited-substrate | PR #130/#135 |
| Receiver binding / assembly / custody | ours-original | |
| **Joint admission waterfill** | **ours-original** | **0 counted archive bytes** — selects and re-solves inside existing sections |

The joint admission waterfill described above is the one mechanism we claim at this
candidate. It is a better decision rule over someone else's representation: a
contribution to the decision, not to the representation.

## What else in this work is ours

The table above is section-scoped — what is in the archive and whose it is. These
mechanisms shaped the candidate without owning a section. **None adds a counted
archive byte** and none changes a classification above.

- **The instruments that priced every decision.** A tail re-encoder that is the
  exact inverse of the shipping decoder, so an edit's cost is the measured archive
  delta, not a bits-per-token estimate (3.8373 measured bits per changed token at
  this candidate's scale); and the measured superposition law — token-edit rate
  costs add, interactions under 3% — which lets the waterfill sum per-chunk rate
  instead of re-encoding every subset.
- **Two zero-distortion rate steps in this archive's ancestry.** A parameter-free
  container transform that re-lays out four already-decided section bodies before
  the Brotli pass, with the receiver restoring each byte-for-byte before parsing, so
  both distortion legs are zero by construction (−657 B); and a tail-override build
  step (−105 B) without which the token-stream rate wins measured elsewhere were
  structurally unreachable from the shipping body.
- **The pose solve.** A damped Gauss-Newton carrier solve. The residual demands a
  multi-coordinate step of 57 to 14,079 integer code units, which the previous
  single-coordinate ±2 search could not travel. Plus an uncapped convergence proof
  over all 600 pairs at zero added bytes, and the un-interleave finding that turned
  two byte-close blockers into one missing transform.
- **A decode-time probability corrector on the miss class**, online and
  decode-identical. The shipped `runtime/free_corrector.py` is the tip of an
  `rr2→rr4→fx1→fx2→ma1` lineage. PR #138 `opal_v1` published this mechanism class
  first, on 2026-08-17 at 08:31Z; our first measured result is the same day at
  14:41Z and we first read that PR at 19:32Z, so the base corrector is concurrent
  independent work and **we claim no priority**. The mechanisms also differ:
  PR #138's correction stops at the rank-one split and preserves the complement's
  relative law exactly, while ours models that complement — which is where our
  −105 B comes from. One later refinement in that lineage **did** read our harvest
  of PR #138 and used its `exp()` state-sync warning as a design check, then solved
  it by exact rounding rather than by their quantisation.
- **Custody apparatus, so the numbers above are checkable**: a seal contract that
  re-derives every pin from disk and refuses a paid evaluation on drift, one
  canonical score arithmetic byte-identical to `upstream/evaluate.py:92`, a
  manifest-driven packet stager whose census reports its own denominator, and a
  registry that hashed all 241 copies of `rc64_backend.c` across our custody roots
  and separated four distinct bodies by role.

The full table, with a receipt on every row, is
`BORROWED_SUBSTRATE_ACCOUNTING.md` §9.5.

## Credits and prior work

This submission runs on PR #130's vehicle and PR #135's trained state. Most of what
decodes here is theirs, and the semantic-token plus HPAC design is a good one to
build on.

- **PR #130 `semantic-pose-HPAC_CPR1`** by Fesal Fayed (`fesalfayed`) — the origin
  of this vehicle.
- **PR #135 `semantic-pose-HPAC_CPR1_polished`** by Shreyan Mohanty (`codexblack`)
  — the trained state this submission re-represents, and the edit-then-recompensate
  pattern.
- **PR #133 `cpr1_cbq_matched8`** by `JasonMo123` — in this ancestry via PR #135.
- **PR #138 `opal_v1`** — published the decode-time-corrector mechanism class
  first. We make no priority claim on it.

## How to verify

From a checkout of the contest repository, with the archive downloaded to
`submissions/jg5_joint_waterfill/archive.zip`:

```bash
sha256sum submissions/jg5_joint_waterfill/archive.zip
# expect f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e

bash evaluate.sh --submission-dir ./submissions/jg5_joint_waterfill --device cuda
```

Expect `Average PoseNet Distortion: 0.00000637`, `Average SegNet Distortion:
0.00020139`, `Final score: 0.15` printed at 2 dp. The score claimed here,
`0.14839100138338618`, is those components recomputed. Budget about 25 minutes on
a T4; see the runtime risk below before scheduling.

## Known limits

1. **Evaluation-time budget.** 1,471.3 s of the 1,800 s job wall is measured;
   token decode alone is 1,341.5 s. Read in one frame this is WARN (fits by
   10.7 s); read in the more conservative frame it is REFUSE (over by 169.3 s).
   Both frames are above; this is the largest open risk on the submission.
2. **`inflate.sh` is not fully self-contained.** It requires `Brotli==1.2.0`
   exactly and calls `uv pip install` — **network at decode time** — if that
   version is absent, exiting 69 if `uv` is missing. It also invokes a C compiler
   at decode time. This follows the declared-dependency precedent of earlier
   accepted submissions here, but "no network at decode time" is a reasonable
   expectation and this submission does not meet it.
3. **No end-to-end rebuild for these bytes**, and the included `compress.py`
   cannot produce them (see the compression-script answer above).
4. **No `[contest-CPU]` row exists on these bytes.** GPU-required.
5. **One published mechanism is a measured null.** Re-mixing the 12 stored pose
   basis dimensions leaves the reachable correction invariant to `1.9e-08`
   (machine precision). It ships nothing, and we report it because it corrected
   our own attribution of why the earlier search failed.
6. **The mixer was tuned on the scored clip.** Its member set, context and
   learning rate were chosen by racing on the video that is scored. We do not
   claim the configuration is optimal on any other clip.

## Public source

- Public source repository: **https://github.com/adpena/comma-lab** — the research
  repository itself, public.
- **All 34 files of the evaluated runtime tree are in version control**, at
  `submissions/robust_current/jg5_sub015_runtime/runtime/`, and each is
  byte-identical by SHA-256 to the tree the score was measured on. An earlier
  draft of this packet disclosed that 24 of those 34 had no source in version
  control; that was true when written and is no longer true.
- Source commit pinned into the evaluation container:
  `56e239829091e56ced913b464f3a6d4e9d5127c5`
- The submitted bytes are bound to their receiver by hash: archive SHA-256
  `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` against
  runtime tree SHA-256
  `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`. Either one
  alone is insufficient to reproduce the score; the pair is the identity.

Thanks for running the contest. Happy to answer questions or re-run anything here.
