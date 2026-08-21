# ddm_dc1 decode-time compute — ship questions, not answers

Date: 2026-08-21
Axis: `[contest-CUDA]` for the inherited fx5 timing/score receipts;
`[macOS-CPU advisory / scorer-free]` for this arm's token-region measurements
Verdict: **READY-RESEARCH, NOT READY-TO-SHIP**
Score claim: false for this arm; no scorer, Modal call, governed launch, or exact evaluation ran

## Result

Decode-time compute is a real rate axis, but it is not a free source of thousands of bytes.
The strongest legal instance I found is a **fixed-grid combinatorial hash sieve**:

- the decoder assumes the HPAC maximum-probability answer for each fixed eight-symbol block;
- the counted payload combinatorially identifies the non-MAP blocks and gives each one a
  truncated SHA-256 question;
- the scorer-free decoder enumerates that block's real HPAC candidates in probability order and
  returns the first hash match.

This is a real solver. It consumed persisted packet bytes and reproduced all **510/510** tested
real fx5 blocks exactly. It also found a genuine finite-target **body-bit** win: group 0's sparse
body used **38 bits versus 73.910477 ideal HPAC bits**, a **35.910477-bit = 4.488810 B
ideal-body equivalent** credit before framing. This is not an archive-byte delta: the standalone
packet header makes the local packet larger. The result repeated byte-identically.

It did **not** survive the broader bounded sample. Across five fixed frame-0 groups, the strongest
sparse bodies used **167 bits versus 127.740765 ideal HPAC bits**, a **39.259235-bit = 4.907404 B
ideal-body equivalent loss** before their counted per-group headers. This is a five-group,
4,080-symbol bounded negative,
not a family kill. The full-video selection/framing problem is unmeasured, and no archive candidate
exists.

The other high-ceiling formulation—ship a scorer-equivalence-cell constraint and solve for any
member—remains blocked. No current receiver-checkable constraint proves SegNet/PoseNet cell
membership without loading the forbidden scorers, and the explicit raw constraint is about
44.244 MB. Bits-back/REC inherits that same missing quotient and refunds zero bits for the current
deterministic field.

The fx5 frontier therefore did not move.

## Live object and exact rate asks

The common contract's old frontier paragraph is stale for this arm. Live authority is fx5_e1:

| quantity | value | evidence class |
|---|---:|---|
| archive | `180,386 B`, SHA `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841` | MEASURED `[contest-CUDA T4 n600]` |
| exact recomputed score | `0.14823186109359` | MEASURED `[contest-CUDA T4 n600]` |
| token stream | `113,777 B`, SHA `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` | MEASURED scorer-free bytes |
| token share | `63.074185%` | DERIVED: `113777/180386` |
| token ideal code length | `910,209.432143 bits = 113,776.179018 B` | MEASURED from real fx5 coding rows |
| rate marginal | `6.6585895312e-7 S/B` | DERIVED: `25/37,545,489` |

The member is `14 B` header + `13,515 B` HPAC + `30,856 B` semantic + `22,028 B`
carrier + `96 B` compact residual + `113,777 B` token stream.

The charter's rc2 asks and the post-fx5 asks lead to the same archive targets:

| basis | cut | target archive | cut if charged wholly to token stream | rate-score credit |
|---|---:|---:|---:|---:|
| rc2 | `12,225 B` | `168,231 B` | `10.744702%`; token `<=101,552 B` | `0.008140126 S` |
| rc2 | `42,470 B` | `137,986 B` | `37.327404%`; token `<=71,307 B` | `0.028279030 S` |
| live fx5 | `12,155 B` | `168,231 B` | `10.683178%` | `0.008093516 S` |
| live fx5 | `42,400 B` | `137,986 B` | `37.265880%` | `0.028232420 S` |

## The compute resource

Primary receipt:
`/Volumes/APDataStore/pact/ddm_fx5/t4_row_r1/MODAL_REMOTE_RESULT.json`.

| surface | seconds | evidence class |
|---|---:|---|
| token decode | `478.652183` | MEASURED `[contest-CUDA T4]` |
| neural render and resize | `55.223904` | MEASURED `[contest-CUDA T4]` |
| full inflate wrapper | `558.626257` | MEASURED `[contest-CUDA T4]` |
| evaluator | `51.902857` | MEASURED `[contest-CUDA T4]` |
| contest-auth-eval wrapper | `622.401365` | MEASURED `[contest-CUDA T4]` |
| Modal call | `629.117285` | MEASURED `[contest-CUDA T4]` |

The gross arithmetic against 1,800 seconds is `1,170.882715 s` spare after the Modal call, or
`1,241.373743 s` after inflate alone. Those are **optimistic gross reserves**, not a licence to add
that much decode. The inherited job also pays checkout, dependency, and other workflow costs.
fx5's own conservative budget predicate charges `610.529114 s` against the `822 s` cold-cache
decode/eval ceiling, leaving **211.470886 s**. The warm-cache `1,302 s` ceiling leaves
**691.470886 s**. These are the honest governed headroom endpoints.

Token decode is `85.683796%` of measured inflate and ran at `246,452.025 symbols/s`. The question
prototype's deterministic enumerate + rank + hash path materialized `781,254` candidates in
`2.149509 s`, or **363,456.964 candidates/s** `[macOS-CPU advisory]`. Assigning all remaining
time to one region buys at most:

| wall allocation | candidates | rank bits `log2(candidates)` |
|---|---:|---:|
| cold-cache governed `211.470886 s` | `76,860,566` | `26.195740 bits` |
| warm-cache governed `691.470886 s` | `251,319,909` | `27.904950 bits` |
| optimistic gross `1,170.882715 s` | `425,565,477` | `28.664806 bits` |

This is the central exchange: search time buys logarithmic rank bits, not bytes linearly. If the
budget is split over regions, `sum(t_i)` is bounded and region `i` can reach only
`log2(q*t_i)` rank bits at measured throughput `q`.

## Family A — hash-constrained decompression

For a region target `x`, let the deterministic HPAC best-first order place it after `M` earlier
candidates. A zero-error ordinal description needs roughly `log2(M+1)` bits. A `k`-bit hash has
collision-free probability `(1-2^-k)^M`; a non-adaptive reliability target `epsilon` therefore
needs approximately:

`k >= log2(M/epsilon)`.

The stronger encoder-side construction used here is target-specific: hash every earlier candidate
and choose the smallest prefix that distinguishes `x`. Its expected width is still
`log2(M)+O(1)`, and the width must be counted, but finite hash luck can beat the prior's
self-information on this one fixed object. That is what happened in group 0.

### Real-coder crosswalk

The prototype did not substitute a proxy entropy model:

- live tokens: 117,964,800 B, SHA
  `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`;
- full real RC64 inverse control: byte-identical, receipt SHA
  `b5a2668f499bc7060f5c09fa36b8435fd98bae80e62d4d5a0fc2ddc3713c2685`;
- frame-0 real RC64 control: `2,831.616980 bits` ideal, `355 B` emitted, first `354 B`
  byte-equal to the live stream prefix; retained payload SHA
  `97841c41a96cc26e37e296923677f653ce914a28c6403f12e80b89e8b04a9be8`;
- independently extracted frame-0 coding rows summed to exactly the retained fx5 frame-0 ledger
  within `1e-9 bit`.

Within an HPAC conditional group, all rows are fixed before the group is observed, so an
eight-symbol product search is the receiver's real conditional law. No scorer participates.

### Prototype results

Prototype source: `experiments/ddm_dc1_decode_time_compute.py`, working-tree SHA
`19583e7f9a5b46f5bfb0ddb3e69112885d3da539207d2a22901b70d4184ee33e` at measurement time.
It requires `--resume-from`, checkpoints each region as a distinct attempt, and retains every
materialized candidate, probability, order, digest, question, and decoded answer.

The actual sparse packet uses a combinatorial rank for the non-MAP block set plus one uniform hash
width. The generic solver and fixed grid are free code; the rank, width, and hashes are counted
video-derived payload.

| frame-0 group | symbols / blocks | non-MAP blocks | direct ideal bits | sparse body bits | ideal-body equivalent `-B` | exact |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48 / 6 | 2 | `73.910477` | `38` | `+4.488810 B` | yes |
| 76 | 1,536 / 192 | 3 | `19.377750` | `30` | `-1.327781 B` | yes |
| 114 | 1,536 / 192 | 4 | `16.690594` | `42` | `-3.163676 B` | yes |
| 152 | 912 / 114 | 5 | `12.885699` | `53` | `-5.014288 B` | yes |
| 189 | 48 / 6 | 1 | `4.876246` | `4` | `+0.109531 B` | yes |
| **total** | **4,080 / 510** | **15** | **`127.740765`** | **`167`** | **`-4.907404 B`** | **510/510** |

`sparse body` omits the local experiment's generic packet header but includes the combinatorial
non-MAP positions and hash bits. A count needs another 29 bits across these five groups if it is not
amortized elsewhere. The retained standalone packets include all headers and are therefore larger
still. This makes the table favorable to the candidate, not to the baseline.

The group-0 packet and decode repeated byte-identically:

- sparse packet SHA `3e688f11533311a307760d45176a88d99dd4b573848713ed552ddf0ac398ca77`;
- decoded SHA `cbb99eb650b335230c4f21ca21323d85eb919d79df140f0337da2c19682e52a7`;
- two independent retained runs produced both hashes exactly.

Even the favorable group-0 ideal-body equivalent repeated over all 600 frames is only a
**2,693.286 B projection**,
22.03% of the 12,225 B ask and 6.34% of the 42,470 B ask. That multiplication is not a measured
full-video result; it only shows the scale required. The observed local win would have to recur
2,724 or 9,462 times, respectively, without selection/header losses.

### Disposition

**SURVIVES CONDITIONALLY, READY-RESEARCH.** Exact whole-field hash replacement is not promising,
but fixed local regions expose model-mismatch ranks that can be much cheaper than HPAC
self-information. The next measurement is a batched full-frame sparse-grid sweep with one retained
bundle per group, not thousands of tiny files. No archive build is warranted until its actual
counted body clears at least 3 KB after headers and a native wall projection.

## Family B — constraint shipping and decode-time solve

Let `C` be a scorer-equivalence cell and `P` the shipped HPAC prior. Exact answer coding costs
`L_x=-log2 P(x)`. An ideal quotient sample costs:

`L_C=-log2 P(C)`,

with gross gain

`L_x-L_C = log2(P(C)/P(x)) = -log2 P(x|C)`.

That is the real seam: a large task cell could remove answer identity. But a legal archive pays the
constraint description, the solve index/seed, all learned/video-derived parameters, and framing.
The solver may be free code; the cell is not.

An explicit 600x384x512 five-class partition at three raw bits/cell is `44,236,800 B`; 600x6 fp16
Pose6 adds `7,200 B`, totaling **44,244,000 B = 388.866x the entire fx5 token stream**. Compression
could reduce that, but no real compressed constraint packet was built or priced here.

The tempting geometry does not yet connect to the wire:

- #580 proves **80.67% real nullity** for the factor-two camera-to-scorer resize. It is a linear
  preimage result, not a SegNet/PoseNet task-cell certificate.
- #49/S12 found 10.29% and 19.53% raw-frame coding gains from preimage choice, but current fx5 is
  procedural and has no raw-frame-pixel archive section to shrink directly.
- Task #597 built a scorer-free PREDICT->PROJECT constraint interface and measured 600 deterministic
  fixture stages, but its own receipt says the desired-cell interface, B2/hard oracle, native
  rasterization, and equal-fidelity real byte result remain measurement-blocked.
- JO1's exact full-field B/H/Pose objective is an encoder-side acceptance law. JO1U made the
  full-population materializer ready to fire, but it does not turn SegNet/PoseNet into a
  receiver-checkable proof. Loading either scorer at inflate remains forbidden.

The legal sufficient condition is stronger than “the encoder checked one answer”: every candidate
the scorer-free decoder may return must be proved inside the desired cell. No current constraint
language supplies that proof.

### Disposition

**BLOCKED, high ceiling, no honest expected byte credit.** The theoretical gross ceiling is the
whole `113,777 B` stream, but the current explicit constraint is much larger and `P(C)` is
unmeasured. Falsifier: counted constraint + solve payload reaches `101,552 B` before the 12,225 B
ask, or one satisfying decoded candidate leaves the desired Seg/Pose cell.

## Family C — bits-back / relative entropy coding on the task quotient

For proposal `P` and receiver-runnable posterior `Q`, the ideal REC price is approximately
`KL(Q||P)` plus finite-seed/index/model/table/framing costs. If `Q=P(.|C)`, then
`KL(Q||P)=-log2 P(C)`: this is exactly the same quotient price as Family B, not a second source of
savings.

The current fx5 field has a deterministic delta posterior. Its posterior entropy is zero, so it
refunds **0 bits**; the price collapses back to ordinary answer coding. A nontrivial quotient `Q`
requires receiver-runnable counted `p` and `q`, integer CDFs, support closure, canonical ANS order,
and an initial-state/seed receipt. None exists for fx5.

Naive rejection REC also inherits the compute wall: a monolithic quotient index larger than about
26.2 governed rank bits is unreachable at the measured local search rate. Any viable REC must
factor into small independent regions and then beat their accumulated indices and headers.

### Disposition

**CURRENT FORMULATION CLOSED; quotient form folded into Family B.** Reopen only when a real
receiver-checkable task-cell posterior exists. Falsifier: counted p/q/tables/state/index is not
smaller than the direct RC64 stream, support fails, output is nondeterministic, or decode crosses the
governed wall.

## Original weird formulation — the combinatorial hash sieve

The built sieve is the charter's requested original formulation. It is a hybrid of MAP defaults,
enumerative sparse-set coding, and target-specific proof questions:

1. Divide a real HPAC conditional group into a generic fixed grid.
2. Reconstruct every MAP block for zero answer bits.
3. Combinatorially encode only the set of non-MAP blocks.
4. For each such block, ship a fixed-width SHA prefix, not its symbols or ordinal.
5. Deterministically best-first search the already-shipped HPAC law until the prefix matches.

This is not data hidden in code: all non-MAP positions and hashes are in the retained packet. It is
not a fake solver: the decoder parses that packet, enumerates, consumes every meaningful bit, and
reproduces the exact target. Its local win is real; its bounded aggregate loss is also real.

## Ranked routing table

Positive `-B` means bytes saved. `UNKNOWN` is not silently treated as zero or as a projection.

| rank | family / claim | evidence class | honest `-B` on fx5 | falsifier | owner / fire order |
|---:|---|---|---:|---|---|
| 1 | fixed-grid combinatorial hash sieve | MEASURED, real fx5 regions, scorer-free | `-4.907404 B` ideal-body equivalent over five groups; group-0 instance `+4.488810 B`; **no archive delta**, full video UNKNOWN | batched full-frame body+headers fails `>=3 KB`, native wall exceeds cold bar, or any parse-back differs | ddm_dc1 successor; queued only after batched-retention rewrite |
| 2 | constraint-shipped task-cell solve | DERIVED ceiling + blocked interfaces | UNKNOWN; gross ceiling `113,777 B`, explicit raw form `-44,130,223 B` versus token stream | counted packet `>=101,552 B`, no universal cell proof, or one solver output exits cell | Task #597 / JO1 consumer; blocked on receiver-checkable proof |
| 3 | task-quotient REC / bits-back | DERIVED | current delta posterior `0 B`; nontrivial quotient UNKNOWN | counted p/q/state/index loses, support/order fails, or compute wall fails | folded into rank 2 until real `Q` exists |
| 4 | monolithic exact hash/free-run search | DERIVED compute bound | expected `<=0 B`; only `26.196..27.905` governed rank bits reachable | target rank exceeds wall or reliability/hash-width exceeds direct code | DEAD; no successor |

No family produced the charter's counter-hypothesis of **>=3 KB honest credit**. Hash questions
survive as a bounded local phenomenon, not as supply against r012 yet.

## Compliance boundaries

- **Rule 118:** hashes, sparse positions, widths, and any video-specific seed are counted. Only the
  generic grid, SHA implementation, HPAC evaluator, combinadic codec, and search are free code.
- **Determinism:** no random search is used. Candidate order is probability-descending with a
  lexical tie-break; SHA domain and packet grammar are fixed; independent runs produced identical
  packet and decoded hashes.
- **Strict scorer rule:** no scorer was loaded. Constraint/REC families remain blocked rather than
  smuggling a scorer or a scorer-derived table into inflate code.
- **Authority:** region bytes/bits are scorer-free rate evidence, not an exact score or frontier
  row. The T4 timing and fx5 score are inherited primary receipts.
- **Payload retention:** all materialized payloads remain under
  `/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/` with per-stage SHA/byte receipts. No
  materialized candidate or question was discarded.
- **Storage caveat:** APDataStore's 128 KiB allocation granularity made the many-file 504-block
  run occupy about 2.0 GiB despite much smaller logical files. Those retained bytes were not
  deleted. A successor must batch arrays by group before a full-frame sweep.

## RECALL EVIDENCE

### Scopes and queries

- Full `.omx/research` corpus and arm messages: `decode compute`, `procedural rate`, `question`,
  `hash`, `constraint`, `task cell`, `quotient`, `bits-back`, `REC`, `Wyner-Ziv`, `PREDICT`,
  `PROJECT`, `preimage`, `nullity`, `solver`, and `HPAC`.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`, then the
  visible-quotient, score-quotient, solved-object rate-dominance, rate-model direction, and
  scorer-conditional rate-distortion rows.
- Graph/state: canonical task ledger rows for #597, prior dc1, fx5, and dx1; current
  `main_hot_state.md`; canonical research index and relevant DAG feed blocks.
- Implementation/custody: actual fx5 runtime, `ddm_jg2_tail_reencode.py`, real RC64 controls,
  decoded-token checkpoint, ideal-bit ledger, packet member, and AP/Vertigo retained artifacts.

### Findings beyond the charter seeds that changed the work

1. The 2026-08-16 same-name dc1 arm proved the fixed-probability coder is within 7.8 B of ideal,
   but its correction explicitly leaves probability models and question replacement open. I reused
   its real-coder discipline and did not repeat coder swaps.
2. The 2026-06-11 inflate-compute inventory declared “compute != information” globally after a
   lossy decoder test. The real fixed-grid hash result supplies a counterexample at the individual
   sequence level: compute plus a counted 38-bit question reconstructed a 73.91-bit-prior group.
   The broader sample then bounded that exception instead of restoring the global claim.
3. Task #597's PREDICT->PROJECT interface is structurally built but explicitly not a measured
   desired-cell/B2 byte result. It became a conditional consumer, not evidence that constraints are
   already shippable.
4. #580's 80.67% resize nullity and #49/S12's raw-frame savings do not directly cut current fx5
   because the archive stores a procedural token/render program, not camera pixels.
5. The VAE harvest's bits-back admission contract already says a delta posterior refunds zero and
   counted p/q/CDF/state are mandatory. The only new application is the task quotient, which
   collapses algebraically to Family B.
6. The real fx5 stream exposes 190 within-frame conditional groups and adaptive inter-frame state.
   That changed the prototype from a surrogate iid sampler to an exact within-group best-first
   search cross-checked against the full RC64 encoder ledger.

## Verification and review status

- Real RC64 one-frame control retained an emitted payload and exact bit ledger.
- 510/510 fixed-grid blocks round-tripped exactly from persisted packets.
- Group-0 packet and decoded output repeated byte-identically across independent stores.
- Combinadic rank/unrank exhaustively round-tripped every subset for populations 1 through 11.
- Hash-prefix, bit-pack, parser, `py_compile`, and focused runtime smoke passed.
- A retained one-bit question mutation changed the decoded candidate from index `194,383` to
  `190,346`; the solver demonstrably consumes the transmitted question instead of returning a
  fixture. Receipt:
  `/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/negative_controls/mutated_group0_start24_question.json`.
- Review pass 1 found and fixed source-unbound resume reuse, then narrowed all local credits to
  ideal-body equivalents rather than archive-byte claims.
- Review pass 2 attacked packet consumption, deterministic repeat, bounded-sample scope, scorer
  absence, and the shared assumption that direct prior self-information is always the shortest
  individual-sequence description. The mutation control fired and no broader claim was promoted.
- The focused P0 measure-and-discard detector returned zero findings for the prototype.
- Developer preflight was 17/25 green. The eight red gate categories were the same inherited
  shared-state/custody/codebase-drift/dispatch/old-landing/lane/substrate-loss categories recorded
  by JO1; neither touched file is a state writer, dispatcher, lane claim, trainer, or scorer loss.
  No waiver or unrelated repair was made.
- No upstream file, frozen packet, scorer, lane claim, or shared state file was changed.
- Two `review_tracker.py` file passes are recorded; the serializer commit is the landing receipt.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_dc1` successor; consumer store: `/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/full_frame_sparse_sweep/`; fire trigger: a reviewed batched-per-group retention format replaces the tiny-file layout, storage preflight passes, and the fixed packet/source hashes above verify; action: run a scorer-free full-frame sparse-grid body/header/wall sweep and stop unless honest credit is at least 3 KB.
- **BLOCKED** — owner: Task #597 plus the JO1 consumer; consumer store: `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/harvest/`; fire trigger: the already-queued MAIN-owned materializer harvest is complete and a reviewed scorer-free constraint proof shows every decoder output remains in the desired Seg/Pose cell; action: price the counted constraint packet before any solver/archive build.

## LIVE-HYPOTHESES

- Finite-target rank mismatch may cluster in HPAC's high-surprise blocks, so a batched sparse-grid
  sweep can harvest more than the five-group sample even though expected hash length does not beat
  an optimal source code.
- Variable block sizes or branch-and-bound across adjacent conditional groups may reduce the
  combinatorial position tax while staying within the measured 26–28 reachable-rank-bit wall.
- A compact scorer-free universal cell certificate, rather than an explicit argmax partition, could
  expose enough task-cell mass to make constraint solving or REC the first representation-scale cut.

## DEAD-ENDS

- Replacing the fixed-probability RC64 coder is closed by the inherited real-coder control and
  near-zero overhead; question replacement is distinct and was the only reopened seam.
- A monolithic whole-stream hash search is closed: compute buys at most about 26–28 governed rank
  bits while the token stream carries about 910,209 ideal bits.
- Current-field bits-back is closed: a deterministic delta posterior has zero entropy to refund.
- Explicit raw argmax-plus-Pose constraints are closed for fx5: 44.244 MB is 388.866 times the token
  stream before framing.
- Resize-null preimage savings cannot be applied directly to fx5's procedural archive because it
  has no stored raw-frame section.
- Scorer-at-inflate and scorer-derived data hidden in code are closed by the strict scorer and
  rule-118 boundaries.

Own-vehicle frontier: **fx5_e1 — S 0.14823186109359 @ 180,386 B [contest-CUDA T4 n600]**, unchanged.
