# submission name: joint_waterfill_rider

> **STATE: PREPARED, NOT SUBMITTED — delete this block before posting.** These
> archive bytes are **not yet published**: the download field below is blank on
> purpose, because the only URL we have hosts a superseded archive. Publishing the
> exact bytes and pinning the URL to that commit is the last mechanical step; no
> pull request has been opened and the score below is not published as a contest
> submission. Both remain the repository owner's decision.
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

**Download: PENDING PUBLICATION — the URL is deliberately blank.**

> **Why there is no link here yet.** These archive bytes have not been published.
> The previous generation's URL is not reused, because it serves a **different
> archive** (`f3bce5d2…`, 180,625 B) under a **different 33-row runtime tree** — a
> download link that resolved would hand a reviewer the wrong object. Publishing
> these exact bytes, pinning a raw URL to that commit, re-downloading it and
> verifying HTTP 200 plus SHA-256 and byte count is `SWAP_PROCEDURE.md` step 4A,
> and it is held for the repository operator. No URL, receipt or manifest
> transfers across a candidate swap.

| Property | Value |
|---|---|
| SHA-256 | `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` |
| Size | 180,456 bytes |
| Members | 1 — `p`, 180,356 bytes, stored, SHA-256 `83fa979c1118499b7dd6083cb20bb66f3f8f47e32cfc16ff30ea66449d81cdf3` |

The submission is only valid against runtime tree
`fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2`, enumerated as
the 36 rows of `MANIFEST.sha256`. For this receiver the archive hash alone does not
determine the score: an earlier generation's bytes scored 79.40 under one receiver
tree and 0.157 under another. See "Runtime tree pin" in the report below.

# report.txt

```text
=== Exact result identity ===
Evidence axis: [contest-CUDA]
Hardware: Tesla T4, Linux x86_64
Samples: 600
Archive SHA-256: df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080
Archive size: 180456 bytes
Member: p, 180356 bytes, stored, SHA-256 83fa979c1118499b7dd6083cb20bb66f3f8f47e32cfc16ff30ea66449d81cdf3
Runtime tree SHA-256: fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2
Portable runtime content tree SHA-256: ccd9f7aba9b5f7837e749e856f178586ad496d51d1011306422a5740c56c915c
Upstream snapshot SHA-256: cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008
Upstream evaluate.py SHA-256: 7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b

=== Evaluation results over 600 samples ===
Average PoseNet Distortion: 0.00000637
Average SegNet Distortion: 0.00020139
Seg contribution: 0.020139
Pose contribution: 0.007981227975693965
Rate contribution: 0.12015824324461456
Recomputed score: 0.14827847122030852
Reported (2 dp display): 0.15
Report-8dp worst-case absolute score error bound: 3.63296497868841e-06
Inflation wall time: 458.752594349 s
Evaluation wall time: 39.72359129999995 s
Total authority wrapper wall time: 508.81608174799993 s
Budget: PASS. 498.476 s charged (inflation + evaluation) against a 822 s cold-uv-cache
  ceiling. The 1800 s CI limit covers the WHOLE job, not inflation alone; the ceiling is
  the residual after the other CI steps. See the budget section below for what is
  MEASURED and what remains a PROJECTION.

The printed "0.15" is a 2-decimal DISPLAY of the evaluator's own final_score field.
The score claimed on this submission is the value recomputed from the reported
components, 0.14827847122030852, which is what the line above records. The
display rounds up across the 0.15 boundary; the components do not.

=== Relationship to the prior candidate ===
Prior packet generation 5 measured S = 0.14839100138338618 at 180625 bytes on the
same axis and the same hardware class. This archive is -169 bytes and delta S
-1.1253016e-04 against it.

That delta is EXACT, not bound-limited, and the reason is measured rather than
argued. The two objects emit BYTE-IDENTICAL 600-pair inflated output on this axis:
both n600 raw streams hash to
6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883 at 3662409600 bytes.
Identical scorer input through the same deterministic CUDA scorer gives identical
scorer output, so both distortion legs cancel exactly and the entire delta is the
rate leg: 25 * (-169) / 37545489 = -1.1253016e-04, which is what the two recomputed
scores differ by. Neither row's 8dp rounding bound enters a delta whose distortion
legs are identical objects rather than independently rounded numbers.

The change that bought those 169 bytes is a lossless re-encode of the same carrier
body under an adaptive arithmetic basis. Nothing about the decoded state moves: this
is the one direction in this packet's history where bytes fall at zero distortion
cost.

Lineage against the earlier row: packet generation 4 measured S = 0.15710198138050818
at 177182 bytes. Against that row this archive is +3274 bytes and delta S -8.8235102e-03.
That comparison IS a delta between two independently-8dp-rounded rows, so both rows'
error bounds apply and they ADD: 3.336608e-06 + 3.632965e-06 = 6.969573e-06. The net is
1266.0x that summed bound, so its sign is determinate.

=== Why the admission has to be joint ===
This body is inherited unchanged from generation 5 and the reasoning below applies to
it byte-for-byte, which the identical raw output above proves rather than assumes.

Segmentation token edits are a pose actuator, and the size is measured: the full
edit set bought -0.012847 score units on segmentation and cost +0.172 on pose, a
13.4x loss, with 571 of 573 edited pairs worse on pose and 2 better. Both scorers
read through the SAME resize, so an edit cannot be made cheap for pose by hiding
it from segmentation. The repair runs after the fact: re-solve the carrier against
the edited renders, then admit jointly by sweeping a Lagrange multiplier on pose
damage and scoring every candidate subset through the exact contest formula.
455 of 573 edits are admitted at zero counted archive bytes for the admission itself.

=== Runtime tree pin ===
These archive bytes have been evaluated on contest-CUDA T4 exactly ONCE, under
runtime tree fdd57749..., which is the tree shipped here. No superseded row
exists on these bytes.

The pin is load-bearing, and the reason is measured, not theoretical.
An earlier generation's archive was evaluated TWICE on identical bytes and
scored 79.40216174747616 under one receiver tree and 0.15710198138050818 under
another. Arithmetic decoding under a mismatched probability model does not error:
it returns rc=0 and emits wrong symbols from the first divergent bin onward, so
the decode "succeeds" and produces garbage frames, and structural parse-back
cannot see it because sections and hashes round-trip correctly either way.

For this receiver the archive hash alone does NOT determine the score. A score
claim on these bytes is valid only against runtime tree fdd57749..., the tree
shipped here and the one the authority receipt validated. The 36 rows of
MANIFEST.sha256 ARE that pin.

=== Evaluation-time budget ===
The official evaluation has a 30-minute limit (upstream/README.md:114), and the
CI job carries that limit as timeout-minutes: 30 on the WHOLE job
(upstream/.github/workflows/eval.yml:30) -- not on inflation alone.

MEASURED on the exact shipping archive and runtime tree, [contest-CUDA T4, n600]:

  inflation                       458.752594349 s
    archive setup                   0.564833165 s
    frame-0 selector and I/O        3.608821572 s
    neural render and resize       41.950293628 s
    token decode                  397.876589923 s
    (decode and render subtotal   454.596335505 s, plus the raw SHA-256 pass)
  evaluation                       39.723591300 s
  charged against the residual    498.476185649 s
  authority wrapper total         508.816081748 s

PROJECTION, not measured by us: the rest of the CI job. Checkout, uv sync, apt and
upload leave a residual window of 822 s (cold uv cache) to 1302 s (warm cache) for
our decode. Only the 1800 s job wall and the install payload SIZES are measured;
every per-step second in that window is an estimate that has never been timed on a
real contest runner. The charged 498.476 s fits the COLD end with 323.524 s of
margin, which is the binding corner.

The predecessor generation is the reason this section exists. On the same axis and
the same wrapper it measured 1419.904212624 s of inflation and 1471.331720073 s
charged -- over both ends of that window. The shipping object replaces the Python
free-corrector with a C port and decodes 3.10x faster, which is what moved the
budget verdict from over-ceiling to PASS with margin.

=== CPU boundary ===
Status of the shipping object's [contest-CPU] axis: MEASURED WALL-INFEASIBLE.
No CPU score exists and none is claimed. The requested evaluation runner is
linux-nvidia-t4.

The exact bytes were run on Linux x86_64 CPU. Inflation was killed at the 1800 s
contest wall before evaluate.py ever started. The receiver's own instrumentation
completed afterwards and reports where the time went:

  token decode                   2427.166373672 s
  neural render and resize        410.182710582 s
  receiver report total          2850.781244341 s

This is a wall result, not a decode failure. The decoded token stream is
byte-identical across the two axes -- SHA-256
cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb on both, at the
same final decoder bit position 910837, with the same NativeFreeCorrector engaged.
The receiver decodes correctly on CPU; it decodes too slowly to fit the contest
wall there. Token decode alone is 6.10x its [contest-CUDA] cost and the render
stage is 9.78x, which is why no reordering of the same work fits 1800 s on CPU.

=== Provenance ===
Candidate seal: ddm_rc2_object_b_clean_port_rr5_rider, seal SHA-256 2e32079c5de2cff9e2c2e6788eb74e8152127273aa0f977cf11cb302a3547005
Seal validation at fire time: SEAL_VALID
Torch (auth wrapper): 2.5.1+cu124; CUDA 12.4; driver 580.95.05; Tesla T4 confirmed
Evaluation environment: torch 2.9.0+cu128, torchvision 0.24.0+cu128, numpy 2.3.4, timm 1.0.22
Source commit pinned into the eval container: d9af7d0f4c8aa33f8a5954e70502f233ee7a1891
Provider job identifiers are retained privately with the authority receipts and
are deliberately not reproduced on this public surface.
```

# eval host info

Measured on the exact shipping archive and runtime tree:

- **Accelerator:** NVIDIA Tesla T4, confirmed by the harness (`gpu_t4_match: true`),
  driver 580.95.05, CUDA 12.4.
- **OS / architecture:** Linux, x86_64.
- **Evaluation packages:** torch 2.9.0+cu128, torchvision 0.24.0+cu128,
  numpy 2.3.4, timm 1.0.22. The authority wrapper itself ran torch 2.5.1+cu124.
- **Denominator:** all 600 public samples, `n_samples: 600`.
- **Upstream:** snapshot SHA-256 `cdad563c…`, `evaluate.py` SHA-256 `7da71a84…`.

Provider job identifiers are retained privately with the authority receipts and are
not reproduced on this public surface.

# build cost info

No model was trained for this candidate; it starts from an already-trained
inherited model state. The work that produced these bytes is a compile-time solve
— edit admission, a carrier re-solve, and a lossless re-encode of the carrier body
— run on local hardware.

The exact evaluation that produced the claimed score is a single T4 run:
**458.752594349 s** of inflation plus **39.723591300 s** of evaluation,
**508.816081748 s** for the whole authority wrapper.

# does your submission require gpu for evaluation (inflation)?

**Yes. Requested runner: `linux-nvidia-t4`.** `inflate.py` performs a neural
render, so the object is GPU-routed.

**Runtime declaration, measured on these bytes.** Charged time is
**498.476185649 s** (inflation + evaluation) against a residual of **822 s** with a
cold `uv` cache — **PASS** at the binding corner, with 323.524 s of margin. The
residual window itself (822 s cold to 1302 s warm) is a **projection**: only the
1800 s job wall and the dependency payload sizes are measured, and no per-step CI
second has been timed by us on a real runner.

**CPU boundary, also measured on these bytes: WALL-INFEASIBLE.** Inflation on
Linux x86_64 CPU was killed at the 1800 s contest wall before `evaluate.py`
started, so **no `[contest-CPU]` score exists and none is claimed**. The decode is
correct there — the decoded token stream is bit-identical to the CUDA axis at the
same decoder bit position — it is 6.10× too slow in token decode alone. That is why
the requested runner is the GPU one.

# did you include the compression script? and want it to be merged?

**Included — `compress.py`, with `COMPRESS.md` beside it. Merging is your call, and
you should read the boundary first, because it does not rebuild these bytes.**

`compress.py` reconstructs an archive from pinned inputs and refuses to exit 0
unless the rebuilt bytes hash to the pinned SHA-256. Run it against **this**
archive and it refuses **by name, before doing any work**, and prints the build
stages it does not express plus the script that performs each one. That
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
`0.14827847122030852`, recomputed from the reported components. That is below the
best ranked leaderboard score as read on 2026-08-20 (PR #135,
`semantic-pose-HPAC_CPR1_polished`, 0.162) and below every prior row in our own
custody.

Four qualifications:

1. **The printed score is `0.15`; the claim is `0.14827847122030852`.** The
   evaluator prints a 2-decimal display that rounds up across exactly the
   boundary this submission sits on. The claim rests on the components.
2. **The claim carries a bound.** The components are reported at 8 decimal
   places, giving a worst-case absolute score error of `3.633e-06`. The claim is
   `0.14827847122030852 ± 3.633e-06`; the distance to 0.15 is about 474× that
   bound, so the sub-0.15 statement is not a rounding artifact.
3. **The improvement is a re-decision over borrowed content, not a new model.**
   No artifact was trained for this candidate. Ours is the decision rule; it
   operates on PR #130 / PR #135's trained state. The accounting table below says
   which is which, section by section.
4. **Both axes are measured on these exact bytes, and only one of them yields a
   score.** The `[contest-CUDA]` row is the claim above. The `[contest-CPU]` run
   was killed at the 1800 s contest wall before the evaluator started, so no CPU
   score exists — that is a measured infeasibility, not a missing measurement, and
   it is why the requested runner is `linux-nvidia-t4`.

Where the headroom is, measured: rate is **81.04%** of this score, and **37.7%** of
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

**Score.** **S 0.14827847122030852 at 180,456 bytes**, same axis: **+3,274 bytes**
for a net **−8.8235e-03**. The body spends rate and buys both distortion legs, the
reverse of every earlier candidate here; a lossless re-encode of the carrier then
gives 169 of those bytes back at **zero** distortion cost. Leg split and error
bounds are in the report above.

**What did not work better.** The three-way `{edit, drop, keep}` solve shipped only
two branches — `drop` needs a receiver change this body has no path for. A
12-dimensional pose-basis re-orientation is a measured null: re-mixing the basis
   leaves the reachable correction invariant to 1.9e-08, so it ships nothing. The
   earlier integer-only native token port also lost on its own T4 row — note that
   this is the *token* port, not the free-corrector port that ships here and cut
   inflation 3.10× at zero score change.

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

From the submission directory in a checkout of the contest repository. The local
legs run today; the hosted leg cannot run until these bytes are published, and it
is written to refuse rather than to resolve against the superseded archive.

```bash
# 1. Runtime integrity. 36 per-file rows (the authority tree hash fdd57749...
# additionally binds the root directory name, the tac import manifest, and
# upstream/evaluate.py, so it is not reproducible from these rows alone; the
# reviewer-reproducible content digest is runtime_content_tree_sha256
# ccd9f7ab..., see MANIFEST.sha256 header + PACKET_TARGET.json).
shasum -a 256 -c MANIFEST.sha256
# expect 36 lines ending in: OK

# 2. Archive identity, from the bytes in the directory.
shasum -a 256 archive.zip
# expect df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080
wc -c < archive.zip
# expect 180456
python3 -c 'import zipfile,hashlib; z=zipfile.ZipFile("archive.zip"); i,=z.infolist(); assert (i.filename,i.file_size,i.compress_type)==("p",180356,0), i; print("member p", i.file_size, hashlib.sha256(z.read("p")).hexdigest())'
# expect member p 180356 83fa979c1118499b7dd6083cb20bb66f3f8f47e32cfc16ff30ea66449d81cdf3

# 3. HOSTED LEG — set both URLs from the publication commit; unset, this refuses.
ARCHIVE_URL=${ARCHIVE_URL:?raw URL pinned to the commit carrying THIS archive}
T4_RECEIPT_URL=${T4_RECEIPT_URL:?raw URL of THIS row's T4 receipt at that commit}
LOCAL_SHA=$(shasum -a 256 archive.zip | awk '{print $1}')
HOSTED_SHA=$(curl -fsSL "$ARCHIVE_URL" | shasum -a 256 | awk '{print $1}')
test "$LOCAL_SHA" = "$HOSTED_SHA"
test "$LOCAL_SHA" = df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080
curl -fsSL "$T4_RECEIPT_URL" | python3 -c 'import json,math,sys; r=json.load(sys.stdin); s=100*r["avg_segnet_dist"]+math.sqrt(10*r["avg_posenet_dist"])+25*r["archive_size_bytes"]/37545489; assert r["n_samples"]==600 and r["gpu_t4_match"] is True and r["score_axis"]=="contest_cuda"; assert r["expected_archive_sha256"]=="df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080" and r["expected_runtime_tree_sha256"]=="fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2"; assert abs(s-r["score_recomputed_from_components"])<1e-15; print(f"recomputed_score={s:.17g}")'

# 4. The evaluation itself.
bash ../../evaluate.sh --submission-dir . --device cuda
```

Expect `Average PoseNet Distortion: 0.00000637`, `Average SegNet Distortion:
0.00020139`, `Final score: 0.15` printed at 2 dp. The score claimed here,
`0.14827847122030852`, is those components recomputed. Measured wall on the
reference T4: 458.752594349 s inflation, 39.723591300 s evaluation.

## Known limits

1. **Evaluation-time budget: measured on our side, projected on yours.** Our
   charged time on these exact bytes is 498.476185649 s. What that has to fit
   inside — the residual left by checkout, `uv sync`, apt and upload — is an
   estimate of 822 s (cold cache) to 1302 s (warm) that we have never timed on a
   real contest runner. We fit the cold end with 323.524 s to spare, but the
   window itself is the projection, and we label it as one.
2. **`inflate.sh` is not fully self-contained.** It requires `Brotli==1.2.0`
   exactly and calls `uv pip install` — **network at decode time** — if that
   version is absent, exiting 69 if `uv` is missing. It also invokes a C compiler
   at decode time. This follows the declared-dependency precedent of earlier
   accepted submissions here, but "no network at decode time" is a reasonable
   expectation and this submission does not meet it.
3. **No end-to-end rebuild for these bytes**, and the included `compress.py`
   cannot produce them (see the compression-script answer above).
4. **The `[contest-CPU]` axis is measured infeasible.** Inflation on Linux
   x86_64 CPU was killed at the 1800 s wall before the evaluator started, so no
   CPU score is claimed. The decode is correct there and bit-identical to the CUDA
   axis; it is 6.10× too slow in token decode alone. Requested runner:
   `linux-nvidia-t4`.
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
- **33 of the 36 files enumerated by the evaluated runtime manifest are already
  in version control**, byte-identical by SHA-256, across
  `submissions/robust_current/jg5_sub015_runtime/runtime/`, `experiments/`,
  `src/tac/` and `runtime-rs/native/f26-corrector/`. The three that are not yet
  published are the two entry points and one receiver module the composed object
  rewired — `inflate.py`, `inflate.sh` and `runtime/residual_archive.py`.
  Publishing all 36 plus the exact `archive.zip` is part of the same operator step
  that pins the download URL, and this line is written to be re-measured then
  rather than assumed.
- Source commit pinned into the evaluation container:
  `d9af7d0f4c8aa33f8a5954e70502f233ee7a1891`
- The submitted bytes are bound to their receiver by hash: archive SHA-256
  `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` against
  runtime tree SHA-256
  `fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2`. Either one
  alone is insufficient to reproduce the score; the pair is the identity.

Thanks for running the contest. Happy to answer questions or re-run anything here.
