# submission name: jg5_joint_waterfill

> **STATE: PREPARED, NOT SUBMITTED — delete this block before posting.** The
> current jg5 archive is publicly hosted at the commit-pinned URL below and its
> downloaded bytes were verified by SHA-256. No pull request has been opened and
> the score below is not published as a contest submission. Opening the PR remains
> the repository owner's one-line decision.
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

## LLM disclosure — DRAFT ONLY; OPERATOR MUST REWRITE IN THEIR OWN WORDS

> **Do not paste this section.** It is a factual scaffold, not the policy answer.
> The repository owner must replace it with their own account after reviewing the
> final submitted diff and the contest's current coding-agents-and-LLMs policy.

- **Setup scaffold:** coding-agent/LLM assistance was used for repository search,
  code drafting, experiment orchestration, result checking, adversarial review and
  this source material. **Operator: name the exact tools/models used, what you did
  personally, and how every generated change was reviewed or tested.**
- **Borrowed-substrate scaffold:** point readers to
  `BORROWED_SUBSTRATE_ACCOUNTING.md`, the NO-FAKE #7 accounting surface. Most of
  the learned content and the representation originate in PR #130/#135; the
  section-by-section table distinguishes inherited, adopted-with-attribution and
  ours-original mechanisms.
- **"Most of the code" scaffold:** the measured accounting answers mechanism and
  content provenance, not code authorship by line. It therefore does **not** by
  itself justify a yes/no policy answer. **Operator: compare the final submitted
  files and their authorship history, answer the policy's test directly, and do
  not substitute the mechanism table for that judgment.**

# upload zipped archive.zip

**Download (mechanics verified):**
<https://raw.githubusercontent.com/adpena/comma-lab/2d61b51988799ec3561d5f8a6f659aeb88cc99d9/submissions/robust_current/jg5_sub015_runtime/runtime/archive.zip>

MAIN downloaded that commit-pinned URL and verified HTTP 200 plus SHA-256
`f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`.
The exact current bytes are pinned below.

> **Swap gate:** if the composed RC2 candidate confirms on T4, this URL is stale
> by definition. Commit and push the exact shipping archive, replace the URL with
> a raw URL pinned to that new commit, download it again, and verify its SHA-256
> before publication. `SWAP_PROCEDURE.md` is the instrument; no URL or receipt
> transfers across the swap.

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
Inflation wall time: GATED-ON-RC2 -- populate only from the fresh composed-tree contest-CUDA receipt
Evaluation wall time: GATED-ON-RC2 -- populate only from that same receipt
Total authority wrapper wall time: GATED-ON-RC2 -- populate only from that same receipt
Budget: GATED-ON-RC2 -- the 1800 s CI limit covers the WHOLE job, not inflation alone.

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

=== Evaluation-time budget: GATED-ON-RC2 ===
The official evaluation has a 30-minute limit (upstream/README.md:114), and the
CI job carries that limit as timeout-minutes: 30 on the WHOLE job
(upstream/.github/workflows/eval.yml:30) -- not on inflation alone.

The possible final shipping object composes the rider archive with the clean
native-corrector runtime. Its real receiver has completed two full local n600
decodes with byte-identical output, and it is sealed, but those local runs were
contended and are not transferable to either contest axis.

Populate this section only from two fresh receipts on the exact composed bytes
and runtime tree:

  A. [contest-CUDA T4, n600]: inflation, evaluation, wrapper total, runtime-tree
     SHA-256, archive SHA-256, and the recomputed score components.
  B. [contest-CPU, n600]: measured inflation outcome and whether evaluate.py ran.

The instrumented-tree T4 observation and the local receiver observations are
different objects or regimes. Neither is a shipping-runtime figure. Until the
fresh receipts exist, the runtime verdict is GATED-ON-RC2 and no PASS/WARN/REFUSE
runtime claim is made for the composed object.

=== CPU boundary ===
Status of the final shipping object's [contest-CPU] axis: GATED-ON-RC2. No CPU
score is claimed. The requested evaluation runner remains linux-nvidia-t4.

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

**GATED-ON-RC2 for the final shipping object.** Populate from the fresh composed
contest-CUDA receipt: provider, confirmed accelerator, OS/architecture, driver,
CUDA, package versions and 600-sample denominator. Do not transfer this block
from either an instrumented runtime or a local decode.

# build cost info

No model was trained for this candidate; it starts from an already-trained
inherited model state. The work that produced these bytes is a compile-time solve
— edit admission plus a carrier re-solve — run on local hardware. Exact-evaluation
duration and cost are **GATED-ON-RC2** and come only from the final authority receipt.

# does your submission require gpu for evaluation (inflation)?

**Yes. Requested runner: `linux-nvidia-t4`.** `inflate.py` performs a neural
render, so the final object is GPU-routed.

**Runtime declaration: GATED-ON-RC2.** The composed candidate's full local n600
receiver is decode-proven and sealed, but its local runs were contended. Populate
the CUDA timing, budget verdict and CPU boundary only from fresh contest-CUDA and
contest-CPU receipts on the exact composed archive/runtime pair. The prior
instrumented runtime and local receiver are not substitutes. Full source slots
are named in the report above under "Evaluation-time budget".

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
4. **The current jg5 score row is measured; the possible RC2 swap is not.** Its
   `[contest-CUDA]` and `[contest-CPU]` declarations remain GATED-ON-RC2 until
   fresh receipts bind the composed archive and runtime tree.

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
   leaves the reachable correction invariant to 1.9e-08, so it ships nothing. The
   earlier integer-only native token port also lost on its own T4 row. A broader
   clean native-corrector runtime is a different object: it is locally decode-proven,
   but its composed shipping runtime remains GATED-ON-RC2.

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

From the submission directory in a checkout of the contest repository. These
commands verify the current hosted jg5 object; if RC2 replaces it, update every
URL and expected identity from the fresh receipts before running them.

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
0.00020139`, `Final score: 0.15` printed at 2 dp. The score claimed here,
`0.14839100138338618`, is those components recomputed. The shipping runtime and
its budget verdict remain GATED-ON-RC2.

## Known limits

1. **Evaluation-time budget.** GATED-ON-RC2. The final CUDA and CPU declarations
   require fresh receipts on the exact composed archive/runtime pair; no timing
   from a different tree or axis transfers.
2. **`inflate.sh` is not fully self-contained.** It requires `Brotli==1.2.0`
   exactly and calls `uv pip install` — **network at decode time** — if that
   version is absent, exiting 69 if `uv` is missing. It also invokes a C compiler
   at decode time. This follows the declared-dependency precedent of earlier
   accepted submissions here, but "no network at decode time" is a reasonable
   expectation and this submission does not meet it.
3. **No end-to-end rebuild for these bytes**, and the included `compress.py`
   cannot produce them (see the compression-script answer above).
4. **Final `[contest-CPU]` boundary is GATED-ON-RC2.** No CPU score is claimed;
   the requested runner is `linux-nvidia-t4`.
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
- **All 33 files enumerated by the evaluated runtime manifest, plus the exact
  `archive.zip`, are in version control**, at
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
