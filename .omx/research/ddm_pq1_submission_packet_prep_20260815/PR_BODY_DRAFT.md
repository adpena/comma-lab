# submission name: jg5_joint_waterfill

> **STATE: PREPARED, NOT SUBMITTED — delete this block before posting.** No
> archive has been hosted, no pull request has been opened, and the score below is
> not published anywhere. Hosting the archive and opening the PR are the
> repository owner's to do.

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
Inflate budget: 1800 seconds; inflation alone uses 1419.9 s, a 1.268x margin.
That margin is not the binding one: the CI limit is 30 minutes on the WHOLE job,
so read the budget section below before using this line.

The printed "0.15" is a 2-decimal DISPLAY of the evaluator's own final_score field.
The score claimed on this submission is the value recomputed from the reported
components, 0.14839100138338618, which is what the line above records. The
display rounds up across the 0.15 boundary; the components do not.

=== Relationship to the prior candidate ===
Prior packet generation 4 measured S = 0.15710198138050818 at 177182 bytes on the
same axis and the same hardware class. This archive is +3443 bytes and
delta S -0.008710980 against it.

This candidate SPENDS rate to buy distortion, the opposite direction from every
prior generation in this packet. A reader comparing byte counts alone would read
the +3443 as a regression. The measured leg split against generation 4:
  rate +2.2926e-03 (+3443 bytes)
  seg  -1.0170e-02
  pose -8.3353e-04
  net  -8.7110e-03
The legs sum to the net. They are DISPLAYED at 5 significant figures, so adding
the printed strings need not reproduce the net digit for digit.

Sign determinacy: the net is a DELTA between two independently-8dp-rounded rows,
so both rows' error bounds apply and they ADD -- 3.336608e-06 + 3.632965e-06 =
6.969573e-06. The net is 1249.86x that summed bound. Dividing by one row's bound
alone would overstate the margin by about 2x; at this magnitude the sign of the
net is unaffected either way.

Like generation 4 and unlike generation 3, this candidate does NOT hold decoded
state constant: both distortion legs move, and here they move in our favour while
the rate leg is paid.

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
1471.3 s of the 1800 s job wall, leaving 328.7 s for checkout, dependency
installation and archive download. Against the internally derived residual window
for those remaining steps on the CUDA path, [890.6, 1430.6] s, this candidate
fits only at the most optimistic end, by about 10.7 s. Our own wall-clock
assessment therefore grades this WARN, not PASS: a margin of 10.7 s on a
warm-cache assumption is not a margin.

Where the time goes: token decode is 1341.5 s of the 1419.9 s inflation, 94.5%.
(Against the 1401.58 s sum of instrumented stages the same figure is 95.72%; the
denominator is stated because both appear in our notes.) The cost is one hot
stage, not diffuse overhead. A native port of that stage's integer half exists and
reproduces this candidate's decode bit-for-bit on the full 600-frame field at
1.77-1.83x on local hardware. That range STRADDLES the 1.804x speedup our own bar
requires, and local run-to-run variance is wider than the distance to it, so the
port is not shown to close the budget. It is also NOT in the tree evaluated here,
and folding it would move the runtime-tree hash and require a new exact
evaluation. It is available work, not a fix claimed on these bytes.

Two residual windows for the non-inflation steps on the CUDA path disagree in
verdict: [822, 1302] s grades this candidate REFUSE, and [890.6, 1430.6] s -- that
same window re-derived with a larger evaluation-time allowance -- grades it WARN.
They are ours, they are projections rather than measurements, and they are not
reconciled. Both are stated.

On the CPU path the same assessment projects 1414-1913 s of inflation against a
residual of [1044, 1332] s, which is over budget in every corner. The prior
lineage MEASURED contest-CPU inflation at 3422.711146813 s against the 1800 s
budget. This candidate ships the same token decoder, so the CPU path is expected
to remain infeasible -- that expectation is INHERITED, not measured on these bytes.

This is the single largest risk on this submission. It is a runtime risk, not a
correctness or score risk: the score above is measured on the exact submitted
bytes.

=== CPU boundary ===
Status of the [contest-CPU] axis on these exact bytes: NO ROW EXISTS. No CPU
score exists and none is claimed. This submission is GPU-required for evaluation.

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

**Yes, and the GPU path is measured but tight. Please read this before scheduling
the run.**

`inflate.py` performs a neural render, so this submission is GPU-routed. On the
authority run, inflation took 1,419.9 s and evaluation 51.4 s: 1,471.3 s of the
30-minute job wall, leaving about 328.7 s for checkout, dependency install and
archive download. We grade our own submission WARN, not PASS, on this axis.

Token decode is 1,341.5 s of that 1,419.9 s, 94.5% — one hot stage, not diffuse
overhead. We have a native port of its integer half that reproduces this
candidate's evaluated decode bit-for-bit on the full 600-frame field at 1.77–1.83×
on local hardware. That range straddles the 1.804× our own bar requires and local
variance is wider than the gap, so the port is **not** shown to close the budget;
it is also not in the tree evaluated here, and folding it would move the
runtime-tree hash and require a new exact evaluation.

Two of our own residual windows for the non-inflation CI steps disagree:
`[822, 1302] s` grades this REFUSE, and `[890.6, 1430.6] s` — the same window
re-derived with a larger evaluation-time allowance — grades it WARN. Both are
projections, not measurements, and they are not reconciled. Both are stated.

**These exact bytes have not been measured on a contest CPU** and no CPU score is
claimed. The prior lineage measured contest-CPU inflation at 3,422.7 s against the
1,800 s budget; this candidate ships the same token decoder, so we expect the CPU
path to remain over budget. That expectation is inherited, not measured here.

Full numbers are in the report above under "Evaluation-time budget".

# did you include the compression script? and want it to be merged?

No. The repository contains an end-to-end rebuild entry point that reconstructs an
archive from pinned inputs and refuses to exit 0 unless the rebuilt bytes hash to
the pinned SHA-256. **It has not been re-run for this candidate, and it cannot
rebuild it.**

That entry point rebuilds the token stream (optionally plus a declared container
repack) and carries the other seven sections through verbatim. This candidate's
chain also re-decides content in sections it copies: the seg token edit solve, the
edit splice, the admission waterfill and the pose-carrier re-solve. No
configuration closes that gap, so the script refuses this archive by name and
cites the builders that do produce it.

We claim no verified end-to-end rebuild for these bytes.

# changes from upstream

None. The pinned upstream snapshot
(`cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`,
`evaluate.py` `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`)
is unmodified. The scorer was not touched.

# competitive or innovative?

**Competitive, on one measured row.**

On the exact submitted bytes the measured `[contest-CUDA]` 600-sample score is
`0.14839100138338618`, re-derived from the reported components rather than read
off the evaluator's rounded `final_score` field. That is below the best ranked
leaderboard score as read on 2026-08-20 (PR #135,
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

# additional comments

## What this candidate changes

Same inherited vehicle as the prior candidate. What changed is how the seg token
edits and the pose carrier are decided: **jointly, not one after the other.** The
predecessor applied all 573 edits and then re-solved the carrier; the edits bought
−0.012847 S on seg and cost +0.172 S on pose, a 13.4× loss. Here admission is swept over a
Lagrange multiplier on pose damage, so an edit is kept only if it pays for the pose
it costs (**455 of 573 admitted**) and the carrier is then re-solved against the
edited renders this archive actually decodes to, under a derived materiality stop
rule (**600 of 600 pairs stopped on `no_improving_step`, zero budget hits**).

Against the prior candidate (177,182 bytes, S 0.15710198138050818) this archive is
**+3,443 bytes** for a net **−8.7110e-03**: it spends rate and buys both distortion
legs, the reverse of every earlier candidate here. A reader comparing byte counts
alone would misread the larger archive as a regression. The leg split and its error
bounds are in the report above.

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
  decode-identical; the shipped `runtime/free_corrector.py` is that corrector. The
  mechanism class was published first by PR #138 (credited below); ours is an
  independent implementation on this vehicle, not a priority claim.
- **Custody apparatus, so the numbers above are checkable**: a seal contract that
  re-derives every pin from disk and refuses a paid evaluation on drift, one
  canonical score arithmetic byte-identical to `upstream/evaluate.py:92`, a
  manifest-driven packet stager whose census reports its own denominator, and a
  registry that hashed all 241 copies of `rc64_backend.c` across our custody roots
  and separated four distinct bodies by role.

**Two corrections to our own record, both against us.** A 12-dimensional pose-basis
re-orientation we investigated is a measured null — re-mixing the basis leaves the
reachable correction invariant to 1.9e-08 — and ships nothing. And the three-way
`{edit, drop, keep}` solve shipped only two branches: `drop` needs a receiver change
this body has no path for, so it is headroom, not a delivered mechanism.

The full table, with a receipt on every row, is
`BORROWED_SUBSTRATE_ACCOUNTING.md` §9.5.

## Credits and prior work

- **PR #130 `semantic-pose-HPAC_CPR1`** by Fesal Fayed (`fesalfayed`) — the
  origin of this vehicle.
- **PR #135 `semantic-pose-HPAC_CPR1_polished`** by Shreyan Mohanty
  (`codexblack`) — the trained state this submission re-represents, and the
  edit-then-recompensate pattern.
- **PR #133 `cpr1_cbq_matched8`** by `JasonMo123` — transitively in this ancestry
  via PR #135; named because a reader tracing our substrate reaches it.
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
   token decode alone is 1,341.5 s. We grade this WARN, and one of our own residual
   windows grades it REFUSE. This is the largest open risk.
2. **`inflate.sh` is not fully self-contained.** It requires `Brotli==1.2.0`
   exactly and calls `uv pip install` — **network at decode time** — if that
   version is absent, exiting 69 if `uv` is missing. It also invokes a C compiler
   at decode time. This follows the declared-dependency precedent of earlier
   accepted submissions here, but "no network at decode time" is a reasonable
   expectation and this submission does not meet it.
3. **No end-to-end rebuild for these bytes**, and the rebuild entry point cannot
   produce them (see the compression-script answer above).
4. **No `[contest-CPU]` row exists on these bytes.** GPU-required.

## Public source

- Public source repository: https://github.com/adpena/comma-lab
- Source commit pinned into the evaluation container:
  `56e239829091e56ced913b464f3a6d4e9d5127c5`
- The submitted bytes are bound to their receiver by hash: archive SHA-256
  `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` against
  runtime tree SHA-256
  `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`. Either one
  alone is insufficient to reproduce the score; the pair is the identity.
