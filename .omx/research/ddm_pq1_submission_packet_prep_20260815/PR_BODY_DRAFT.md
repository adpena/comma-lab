# submission name: jg5_joint_waterfill

> **STATE: PREPARED, NOT SUBMITTED.** This body is staged for review. No archive
> has been hosted, no pull request has been opened, and the score below is not
> published anywhere. Opening the PR and hosting the archive are reserved to the
> repository owner.

# upload zipped archive.zip

**Download:** pending operator-authorized hosting. No URL is claimed here, because
no URL exists yet. The exact bytes a judge would download are pinned:

| Property | Value |
|---|---|
| SHA-256 | `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` |
| Size | 180,625 bytes |
| Members | 1 — `p`, 180,525 bytes, stored, SHA-256 `54b445da3a1a4b4c7012c83b25c3e0d87daab5ce10cd54a1598cfb239ab05b4a` |

The submission is only valid against runtime tree
`2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`. That is not
boilerplate — see "Score and runtime boundary" below for the measured reason.

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
Inflate budget: 1800 seconds; measured headroom 1.268x

The printed "0.15" is a 2-decimal DISPLAY of the evaluator's own final_score field.
The score claimed on this submission is the value recomputed from the reported
components, 0.14839100138338618, which is what the line above records. The
display rounds up across the 0.15 boundary; the components do not.

=== Relationship to the prior candidate ===
Prior packet generation 4 measured S = 0.15710198138050818 at 177182 bytes on the
same axis and the same hardware class. This archive is +3443 bytes and
delta S -0.008710980 against it.

This candidate SPENDS rate to buy distortion, which is the opposite direction
from every prior generation in this packet, and it is worth stating plainly
because a reader scanning byte counts alone would read the +3443 as a regression.
The measured leg split against generation 4:
  rate +2.2926e-03 (+3443 bytes)
  seg  -1.0170e-02
  pose -8.3353e-04
  net  -8.7110e-03
The legs sum to the net; the values above are DISPLAYED at 5 significant figures,
so adding the printed strings need not reproduce the net digit for digit. The
rounding is in the display, not in the arithmetic.

Sign determinacy: the net is a DELTA between two independently-8dp-rounded rows,
so both rows' error bounds apply and they ADD -- 3.336608e-06 + 3.632965e-06 =
6.969573e-06. The net is 1249.86x that summed bound. Dividing by one row's bound
alone would overstate the margin by about 2x; at this magnitude the conclusion is
unaffected, but the arithmetic is stated the correct way regardless.

Like generation 4 and unlike generation 3, this candidate does NOT hold decoded
state constant: both distortion legs move, and here they move in our favour while
the rate leg is paid.

=== Runtime tree pin ===
These archive bytes have been evaluated on contest-CUDA T4 exactly ONCE, under
runtime tree 2103073d..., which is the tree shipped here. No superseded row
exists on these bytes.

The pin is still load-bearing, and the reason is empirical rather than theoretical.
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

On the CPU path the same assessment projects 1414-1913 s of inflation against a
residual of [1044, 1332] s, which is over budget in every corner. The prior
lineage MEASURED contest-CPU inflation at 3422.711146813 s against the 1800 s
budget. This candidate ships the same token decoder, so the CPU path is expected
to remain infeasible -- that expectation is INHERITED, not measured on these bytes.

This is disclosed rather than discovered. It is the single largest risk on this
submission, and it is a runtime risk, not a correctness or score risk: the score
above is measured on the exact submitted bytes.

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

The candidate was produced from an already-trained inherited model state; no new
model was trained for it. The work that produced these bytes is a compile-time
solve (edit admission plus a carrier re-solve), run on local hardware. The
exact-evaluation row itself is a single T4 run of about 25 minutes.

# does your submission require gpu for evaluation (inflation)?

**Yes — and this answer carries a measured caveat that a judge should read before
scheduling the run.**

`inflate.py` performs a neural render, which is why this submission is
GPU-routed. **These exact bytes have not been measured on a contest CPU**, and no
CPU score is claimed. Our own projection for the CPU path is that inflation would
exceed the 30-minute budget; the immediately prior lineage measured contest-CPU
inflation at 3,422.7 s against an 1,800 s budget, and this candidate ships the
same token decoder.

**The T4 path is measured but tight.** Inflation took 1,419.9 s and evaluation
51.4 s, so 1,471.3 s of the 30-minute job wall is accounted for, leaving about
328.7 s for checkout, dependency installation and archive download. We grade our
own submission WARN rather than PASS on this axis and say so here rather than
letting a judge discover it as a timeout.

# did you include the compression script? and want it to be merged?

**Scope reduction, stated rather than inherited.** The repository contains an
end-to-end rebuild entry point that reconstructs an archive from pinned retained
inputs and refuses to exit 0 unless the rebuilt bytes hash to the pinned SHA-256.
**That entry point has not been re-run for this candidate**, so we are not
claiming a verified end-to-end rebuild for these bytes. We are not asking for a
compression script to be merged.

# changes from upstream

None. The pinned upstream snapshot
(`cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`,
`evaluate.py` `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`)
is unmodified. The scorer was not touched.

# competitive or innovative?

**Competitive, on a measured row, stated against what is actually verified.**

On the exact submitted bytes the measured `[contest-CUDA]` 600-sample score is
`0.14839100138338618`, which we re-derived from the reported components
independently rather than reading the evaluator's rounded `final_score` field.
That is below the best ranked score on the leaderboard at the time of writing
(PR #135, `semantic-pose-HPAC_CPR1_polished`, 0.162) and below every prior row in
our own custody.

Four qualifications, so the claim is not read as more than it is:

1. **The printed score is `0.15`; the claim is `0.14839100138338618`.** The
   evaluator prints a 2-decimal display that rounds up across exactly the
   boundary this submission sits on. The claim rests on the components.
2. **The claim carries a bound.** The components are reported at 8 decimal
   places, giving a worst-case absolute score error of `3.633e-06`. The claim is
   `0.14839100138338618 ± 3.633e-06`; the distance to 0.15 is about 443× that
   bound, so the sub-0.15 statement is not a rounding artifact.
3. **The improvement is a re-decision over borrowed content, not a new model.**
   No artifact was trained for this candidate. What is ours is the decision rule;
   what it operates on is PR #130 / PR #135's trained state. The accounting table
   below is explicit about which is which.
4. **One axis is measured, one is not.** `[contest-CUDA]` is measured on these
   bytes. `[contest-CPU]` is not, and the evaluation-time budget on the GPU path
   is tight enough that we grade it WARN ourselves.

# additional comments

## Score and runtime boundary

The runtime-tree pin is load-bearing and the reason is empirical, not theoretical.
The previous candidate in this lineage was evaluated twice on **byte-identical
archive bytes** and scored 79.40216174747616 under one receiver tree and
0.15710198138050818 under another. Arithmetic decoding under a mismatched
probability model does not raise: it returns rc=0 and emits wrong symbols from the
first divergent bin onward, so the decode "succeeds" and produces garbage frames,
and structural parse-back cannot see it because sections and hashes round-trip
correctly either way. **For this receiver the archive hash alone does not
determine the score.** These bytes have exactly one row, under the tree shipped here.

## What the distortion legs cost

Against the prior candidate (177,182 bytes, S 0.15710198138050818) this archive is
**+3,443 bytes** and the legs are rate +2.2926e-03, seg −1.0170e-02,
pose −8.3353e-04, for a net of **−8.7110e-03**. This candidate spends rate and buys
both distortion legs — the reverse of every earlier candidate in this lineage — so
a reader comparing byte counts alone would misread the larger archive as a
regression. Sign determinacy: the net is a delta between two independently
8dp-rounded rows, so both bounds apply and add (3.336608e-06 + 3.632965e-06 =
6.969573e-06); the net is 1249.86× that summed bound.

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

The one mechanism we claim at this candidate: edit admission and pose compensation
are solved **jointly** — admission swept over a Lagrange multiplier on pose damage,
455 of 573 edits admitted, and the carrier re-solved against this candidate's own
edited renders under a derived materiality stop rule (600/600 pairs stopped on
`no_improving_step`, zero budget hits). The predecessor kept all 573 edits and paid
roughly 13× more pose than the edits bought in seg. A better decision rule over
someone else's representation is a contribution to the decision, not to the
representation, and we do not claim otherwise.

## Credits and prior work

- **PR #130 `semantic-pose-HPAC_CPR1`** by Fesal Fayed (`fesalfayed`) — the
  origin of this vehicle.
- **PR #135 `semantic-pose-HPAC_CPR1_polished`** by Shreyan Mohanty
  (`codexblack`) — the trained state this submission re-represents, and the
  edit-then-recompensate pattern.
- **PR #133 `cpr1_cbq_matched8`** by `JasonMo123` — transitively in this
  ancestry via PR #135; named because a reader tracing our substrate reaches it
  whether or not we mention it.
- **PR #138 `opal_v1`** — published the decode-time-corrector mechanism class
  first. We make no priority claim on it.
- **One caveat worth stating on the record: `inflate.sh` is not fully
  self-contained.** It requires `Brotli==1.2.0` exactly and will call
  `uv pip install` — reaching the **network at decode time** — if that version is
  not already present, exiting 69 if `uv` is absent. It also invokes a C compiler
  at decode time. This follows the declared-dependency precedent set by earlier
  accepted submissions here, but "no network at decode time" is a reasonable
  expectation and this submission does not meet it.

## Public source and reproducibility

- Public source repository: https://github.com/adpena/comma-lab
- Source commit pinned into the evaluation container:
  `56e239829091e56ced913b464f3a6d4e9d5127c5`
- The submitted bytes are bound to their receiver by hash rather than by
  description: archive SHA-256
  `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` against
  runtime tree SHA-256
  `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`. Either one
  alone is insufficient to reproduce the score; the pair is the identity.

Two items are open by construction and must not be quietly closed: the end-to-end
rebuild has not been re-run for these bytes, and no contest-CPU row exists on them.
