# ddm_cl3 — the SMALLER-prior direction and seed selection of the HPAC prior on the shipped object

**Arm:** ddm_cl3 (Opus, spawned 2026-09-05 ~16:05Z on MAIN's charter
`.omx/research/charters/ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md`). Tokens: `[no-triality] [p0-ledger-ok]`.

**Axis of every byte number below:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]` (pack + fx1 mixer + RC64
through the shipped fs2 path; receiver-copy decode identity). Training axis: `[macOS-MPS research-signal]`. No scorer ran.
The token FIELD is bit-identical on every row (sha `cc10a7b0…63efb`), so d_seg / d_pose are HELD by construction and only
model bytes + stream bytes move. Labels: MEASURED / DERIVED / PREDICTED as marked. `score_claim=false`.

**Pointer at spawn — and it MOVED during setup.** The charter was written against fs2 (S 0.14784474152757654 @ 180,023 B).
At 16:08:42Z cl2's own λ=1.0 control repack landed its contest-CUDA T4 row and became `effective_frontier`:

| | score | archive B | archive sha | axis |
|---|---:|---:|---|---|
| fs2 (charter's pointer) | 0.14784474152757654 | 180,023 | `a8f3a379…` | contest-CUDA T4 n600 |
| **cl2 λ=1.0 control (LIVE pointer)** | **0.14781744131049854** | **179,982** | `08ec8533…` | contest-CUDA T4 n600 |

Re-derived at the live pointer (DERIVED, per [[binding-instruction-numbers-expire-and-nobody-rederives-them]]):
demand to sub-0.12 = **41,776.8 B** (was 41,817.8); rate-corner archive **138,205.2 B** (unchanged — the pointer moved by
exactly its own rate term); exchange 6.658589531221714e-7 S/B. **The bar a cl3 rung must beat is 179,982 B / joint
126,885 B — not fs2's 180,023 / 126,926.** cl2's measured T4 row also confirms the field-held claim end to end: seg
0.00020139 and pose 6.14e-06 came back at the base row's values, exactly as its seal pre-registered.

**Pointer re-derived from components, not recalled (MEASURED).** cl2's T4 receipt
(`…/ddm_cl2_t4_lambda1_control_repack_20260905/MODAL_REMOTE_RESULT.json`, `passed: true`, rc 0,
`expected_archive_sha256 = 08ec8533…`, `expected_archive_size_bytes = 179982`) carries a `final_score` field reading
**`0.15`** — the rounded field CLAUDE.md warns never to quote. Recomputing
`S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489` from `seg 0.00020139`, `pose 6.14e-06`, `179,982 B` gives
**0.14781744131049854**, matching the canonical pointer to `abs diff 0.0`. Decomposition of the live pointer:

| term | value | share |
|---|---:|---:|
| rate `25·179,982/37,545,489` | 0.11984262610083464 | **81.1 %** |
| seg `100·d_seg` | 0.020139 | 13.6 % |
| pose `√(10·d_pose)` | 0.007835815209663893 | 5.3 % |

The rate term is already **below 0.12 on its own**. Sub-0.12 therefore needs the 0.027975 of seg+pose paid for out of
rate — the 41,776.8 B demand — which is exactly the corner this ladder is probing, and why a rung is judged purely on
`ΔJ` with distortion held.

---

## 1. PRIOR-LAW PREDICTION (written before any cl3 measurement; charter §"PRIOR-LAW PREDICTION", m38)

From cl2's one measured secant (λ 1.0→0.5 bought +156 B of stream for +350 B of model; slope +0.446 against the −1
break-even), capacity is already past the point where more prior helps, so the smaller direction should pay:

| # | line | PREDICTED | MEASURED | residual |
|---|---|---|---|---|
| P1 | λ=2.0 model | −250…−400 B | _pending_ | _pending_ |
| P2 | λ=2.0 stream tax | +0…+200 B | _pending_ | _pending_ |
| P3 | λ=2.0 net joint vs the control | −350…−50 B (**PAYS**) | _pending_ | _pending_ |
| P4 | λ=4.0 model | −500…−700 B | _pending_ | _pending_ |
| P5 | λ=4.0 stream tax | +200…+900 B (falsifier boundary; either sign admissible) | _pending_ | _pending_ |
| P6 | seeds 20260717/20260718 at λ=1.0, min-of-3 | −40…−90 B beyond the control | _pending_ | _pending_ |
| P7 | seed spread | within ±20 B ⇒ the seed lever is at its floor | _pending_ | _pending_ |

**What makes P6/P7 readable at n=2 (MEASURED, from cl2, before any cl3 rung).** cl2 ran the λ=1.0 law TWICE from a
fresh root (control + twin) and got raw IHS1 sha `81728190…`, packed Brotli-q11 sha `66801b10…` (13,466 B), RC64 stream
sha `e07274ca…` (113,419 B) and candidate archive sha `08ec8533…` (179,982 B) — **byte-for-byte identical at every
layer**. So the run-to-run noise floor of `J` at a FIXED law and seed is exactly **0 B**, not "of order 41 B". Two
consequences: (i) any seed-to-seed spread cl3 measures is a pure seed effect with zero run-noise mixed in, so two extra
seeds are enough to characterise the lever; (ii) cl2's −41 B is NOT "training noise of that order" — the charter's own
phrasing for P6 is looser than the evidence. The −41 B is deterministic; the open question is only whether a different
seed lands lower.

**FALSIFIER (whole-axis, pre-registered):** if λ=2.0 nets ≥ 0 B against the control, the capacity axis is CLOSED IN BOTH
DIRECTIONS on the shipped object (verdict_scope: this vehicle, this formulation), and λ=4.0 is NOT fired — recorded as
not-run-because-falsified, not as an untested rung.

---

## 2. Instrument (cl2's, nothing re-implemented)

Trainer `tools/train_ddm_cl1_hpac_capacity.py`, profile `cl2_shipped_ladder` = the JF1 warm-start reference law on Metal:
60-epoch cosine from the shipped **epoch-634 EMA init** (`ff2d3e45…2afd9`), cache `f29c479a…`, field `cc10a7b0…`,
batch 8, QAT 0.5, lr 0.003. Pricer `experiments/ddm_cl2_hpac_prior_capacity_ladder.py` (cl2's sha `e3153943fe34239c…`
at spawn): pack (`_pack_terminal_ihs1` + Brotli q0–q11 race) → stage a copy of the fs2 fire tree carrying the new hpac
section → encode TWICE with `encode_tail` (streams must be byte-identical) → receiver-copy decode back to the exact field.

**What cl3 changed, and only this** (commits `7fe2cea8c`, `ca6db6afa`, `c65b7e003`):

1. **Trainer, purely additive.** `PREREGISTERED_RATE_LAMBDAS_BY_PROFILE["cl2_shipped_ladder"]` widened
   {1.0, 0.5, 0.25} → {4.0, 2.0, 1.0, 0.5, 0.25}; new `PREREGISTERED_SEEDS_BY_PROFILE` admits {20260716, 20260717,
   20260718} for that profile and exactly the config-pinned seed for every other profile, so `cl1` / `rx2_mc36` /
   `jf1_joint_refit` are behaviour-identical. `seed` moved out of the strict-equality loop into the set check beside
   `rate_lambda`; every other config key is still a strict equality. No cl2 row changes: a landed rung carries its own
   λ and seed in its checkpoint, and the widening only governs what a FUTURE invocation may run.
2. **Pricer.** New rung names + `RUNG_SEED` (checked beside `rate_lambda`, so a rung can never be priced from a
   checkpoint trained under a different seed than its name claims) + a fail-closed import guard that the two dicts
   agree; `rung_root()` gives cl3 its own store and leaves every cl2 rung where cl2 measured it; `report --out` so
   cl2's landed `LADDER_REPORT.json` is never overwritten; `report --pointer-archive-bytes` so the decision reads the
   LIVE pointer instead of a baked-in fs2 constant.
3. **Verified by source inspection (not assumed):** the trainer seeds its RNG from `args.seed` alone
   (`torch.manual_seed` / `np.random.seed` / `random.seed` / `torch.Generator(...).manual_seed`). The run-identity
   hashes feed resume-drift detection, never the random stream — so editing the trainer cannot move any rung's
   trajectory, and a cl3 λ=1.0/seed-20260716 run would still reproduce cl2's control tensors.

## 3. Two aborted launches, recorded plainly (both mine, both preserved)

**(a) First λ=2.0 attempt — ExFAT tier.** Launched 16:04Z (counter 907, trainer pid 1273) onto the ExFAT fallback tier;
**I** SIGTERMed it at ~16:12Z after 455.6 s / 4 epochs. The run was healthy and descending — the reason was the tier.
**MEASURED:** copying the fs2 fire tree onto ExFAT makes macOS write **44 AppleDouble `._*` companions (68 files → 85)**,
which would corrupt the staged receiver runtime tree's census and sha; and `._*.pt` companions (5 already present)
pollute the trainer's `rglob("*.pt")` artifact manifest.

**What it would have cost, concretely (traced, not guessed):** `stage_verify` check (c) computes
`tree_ok = set(differing) <= {"archive.zip", "inflate.py"} and len(inflate_diff) == 2` over `_tree_facts(receiver_copy)`
against the fs2 fire tree. Forty-four `._*` companions in the receiver copy put 44 extra names into `differing`, so
**every cl3 rung would have hard-failed the verify gate** — and it would have failed at verify time, an hour of encode
after the mistake was made. The routing fix is not tidiness; it is the difference between a rung that can be sealed and
one that cannot.

*Adjudication of the actor (asked by MAIN; MEASURED from the run's own `resource_safe_run_status.json`):*
`status=killed`, `exit=143`, `kill_action={action: SIGTERM_then_SIGKILL_process_group, reason: "external_signal",
signal: 15}`, `peak_rss_mib=1525.734` against `rss_limit_mib=118784`, `elapsed_s=455.6` against `timeout_s=7200`.
**safe_run did not kill it on its own policy and neither did the watchdog (report-only in that window) — I did**, with
`kill 1273 / 1258 / 1257`; safe_run correctly classified the result as an external signal. The receipt also makes
MAIN's point quantitatively: peak RSS was **1.5 GiB against a 116 GiB cap**, so the RSS guard was 77× away from firing
while the actual pressure was on the GPU. **Metal working sets are invisible to RSS, so an RSS cap cannot protect a
Metal launch** — that is the generalisable lesson, not the AppleDouble finding.

Cure for the tier: every cl3 rung lives on the APFS primary tier (~172 MB each, ~1.4 GiB for eight against 29 GiB
free); the ExFAT tier carries only the one 3.66 GB parse-back render — a single file, no tree census, no `*.pt` glob.
This composes the charter's disk instruction (keep the big render off the primary tier) with the measured correctness
requirement; the deviation is exactly that and nothing more.

**(b) Second λ=2.0 attempt — co-resident with md3 on Metal.** Launched 16:14:34Z (counter 908, trainer pid 19417) on
APFS; I stopped it at 16:16:42Z after 127.9 s on MAIN's **binding charter correction**: do NOT run a Metal trainer
while md3's 49.6 GiB Metal cell is live. The charter's original line ("a 2.4 GiB trainer is admitted beside it") was
withdrawn — at 16:12:53Z the memory watchdog logged a CRITICAL memory_pressure alarm (compressor 42.4 GiB, growing
5.2 GiB/s) with a cl3 trainer and md3's cell both on the GPU.

Both partials are PRESERVED, never deleted, each with an `ABORT_NOTE.md`:
`…/APDataStore/…/aborted/lambda_2p0_exfat_partial_20260905T1612Z/` and
`…/VertigoDataTier/…/aborted/lambda_2p0_metal_coresident_partial_20260905T1616Z/`.

**(c) What the two aborts bought, free:** epoch-0 telemetry is IDENTICAL across both attempts — bpp
0.00790358228417244, est. model 17,947 B, est. tokens 116,544 B — fresh-vs-fresh, across two volumes and two launches.
MPS determinism on this object is re-confirmed at no cost.

## 4. Launch discipline (MEASURED, as corrected)

**Nothing else on Metal.** The first training launch is gated on md3's terminal receipt
`.omx/tmp/codex_runs/md3_different_init_DONE.json.done` AND a process-table check that no `qbr1_born_fairform` cell
remains, polled by a background until-loop; the rungs then run serially with the GPU to themselves. CPU price stages
may still overlap a Metal training (cl2's own documented pattern — they touch no Metal); peaks measured during any
overlap are graded CONFOUNDED. The CPU JF1 profile is NOT a substitute for a Metal rung: it is a different instrument,
and cl2 measured it at +252 B against the MPS control's −41 B.

Every stage goes through `tools/launch_detached_process.py` with a distinct `--done-receipt`, waited on with a
background until-loop (a foreground wait over ~3 min is reaped rc=144). The trainer fails closed without
`PYTHONHASHSEED=0` / `TAC_ADMISSION_ENFORCE=1` / `PYTORCH_ENABLE_MPS_FALLBACK=0`; cl3 passes all three explicitly
through `--env` so they are recorded in the launch manifest instead of inherited from a shell.

**Co-residency cost, MEASURED before the stop:** ~87 s per epoch with md3 on the GPU against cl2's ~53 s alone
(1.64×) — so co-residency was not only unsafe, it was slow.

### 4a. Co-residency with md3 is NOT safe — MEASURED, and it supersedes a 45-second spot check

MAIN re-authorised Metal at 16:29Z on a 45-second observation ("compressor flat 1.8 GiB") and armed a tripwire rule:
SIGTERM the trainer if a `memory_pressure` WARN or CRITICAL fires, then resume after md3's terminal receipt. I relaunched
(counter 909, trainer 26914) at 16:18Z. **The tripwire fired 100 s later and I SIGTERMed at 16:19:43Z, at epoch 0.**
That is the rule working exactly as written.

The full alarm ledger (`.omx/tmp/memory_watchdog/launch_r6/run.log`, all `report_only`) against my three trainer
windows — A 16:05:37–16:13:13, B 16:14:34–16:16:42, C 16:18:0x–16:19:43 — is the decisive evidence:

| observed_utc | level | reason | cl3 trainer live? |
|---|---|---|---|
| 16:04:21Z | WARN | compressor 18.86 GiB ≥ 16.0 | **no** (md3 alone) |
| 16:08:14Z | CRITICAL | growing 4.27 GiB/s, already 31.11 GiB | yes (A, +2.6 min) |
| 16:08:19Z | CRITICAL | **compressor 54.89 GiB ≥ 48.0** | yes (A) |
| 16:08:24Z | WARN | compressor 24.00 GiB | yes (A) |
| 16:12:53Z | CRITICAL | growing 5.20 GiB/s, already 42.37 GiB | yes (A, +7 min) |
| 16:19:22Z | WARN | compressor 21.07 GiB | yes (C, +1.3 min) |
| 16:19:27Z | CRITICAL | **compressor 55.62 GiB ≥ 48.0**, growing 6.76 GiB/s | yes (C) |
| 16:19:33Z | WARN | compressor 28.50 GiB | yes (C) |

Reading, plainly — with one sub-claim of mine corrected below. **Every crossing of the 48 GiB
CRITICAL line happened with a cl3 Metal trainer live** — twice, independently, ~1.3 min and ~2.6 min after launch,
reaching 54.89 and 55.62 GiB. Window B is not a counterexample: it lived 2 min 8 s, shorter than the ~2.5 min the ramp
takes to develop. MAIN's 45-second window was likewise too short to see it — the spot check measured the ramp's
beginning, not its top.

**Self-correction (MEASURED 16:32:45Z, after every cl3 process was stopped).** I first wrote that "md3 alone already
sits at or over the 16 GiB WARN line", citing a 16.77 GiB compressor reading taken at 16:26Z. That reading was
CONTAMINATED: it was the decay tail of my own trainer, 7 minutes after the tripwire killed it. At true steady state
with md3 alone the machine measures **wired 49.8 GiB, compressor 9.0 GiB** — comfortably under the 16 GiB WARN line.
What survives is the 16:04:21Z WARN (18.86 GiB), which does predate every cl3 launch and shows md3's own ramp can
touch the WARN line transiently. The corrected conclusion is STRONGER, not weaker: md3 alone rests near 9 GiB, so the
54.89 and 55.62 GiB CRITICAL excursions are attributable to the cl3 Metal trainer almost in full. I record the error
because the number I cited was produced by the very process whose effect I was trying to exclude.

**The campaign's own ledger already held the answer, and it is 16× the charter's number.**
`tools/measured_peaks.py lookup --family train_ddm_cl1_hpac_capacity` governs at **38.622 GiB
system-availability delta** (RSS 1.656 GiB) — recorded 2026-09-05T13:47Z from cl2's own `lambda_1p0` launch. The
charter's "a 2.4 GiB trainer is admitted beside it" priced this trainer by its RSS, which is the one number that cannot
see a Metal working set. I recorded my three partials into the same ledger (row_count 4 → 7; the governing peak is
unchanged at 38.622 GiB):

| run | elapsed s | peak RSS GiB | **system-availability delta GiB** | attribution |
|---|---:|---:|---:|---|
| A (ExFAT) | 455.6 | 1.490 | **37.493** | CONFOUNDED_OVERLAPPING_CELL |
| B (co-resident) | 127.5 | 1.490 | **16.148** | SOLE_CELL_INFERRED_FROM_LEDGER |
| C (tripwire) | 52.9 | 1.632 | **34.751** | SOLE_CELL_INFERRED_FROM_LEDGER |

Run C is the decisive one: **34.75 GiB of system availability consumed in 52.9 seconds.** That is the ramp, and it is
why a 45-second window reads low — it samples the ramp's beginning, not its top. Add md3's declared 49.572 GiB and the
pair asks ~85–88 GiB of a 116 GiB ceiling, with the compressor measured at 55 GiB. The arithmetic and the alarms agree.

So the honest verdict is MAIN's FIRST correction, not the re-authorisation: **a cl3 Metal trainer must not run while
md3's cell is live.** The tripwire's own prescription ("resume after `md3_different_init_DONE.json.done` lands") is the
path this arm is on. The RSS guard cannot substitute for the tripwire here — the aborted run's receipt measured peak
RSS 1.5 GiB against a 116 GiB cap while the GPU-side pressure was 55 GiB, i.e. **the guard was 77× from firing on a
condition it structurally cannot see.**

## 4b. THE OBJECT CHANGED UNDER THE LADDER (2026-09-05T22:48Z) — re-grounded, not re-anchored

Between this arm's charter and its first rung the pointer moved **four times** and, more importantly, the OBJECT
changed. Two changes matter to a capacity ladder:

* **ddm_rc1** replaced the generic Brotli byte coder on both MODEL sections with an adaptive per-group binary-tree
  range coder (the `RC1H` / `RC1S` riders). The hpac section is no longer `brotli(raw IHS1)`; it is
  `brotli(ck2_interleave(apply_hpac(raw, row_counts, shift=5)), q11, lgwin24)` and measures **12,343 B**, against
  cl2's Brotli 13,466 B.
* **ddm_pc1** shrank the carrier 22,031 → 18,568 B.

Live pointer: **S 0.14411787458634504 @ 174,786 B**, sha `1de6c5d7…`, tree
`…/ddm_pc1_pose_carrier_efficiency/retained/v3x16_on_rc1_candidate_runtime/`. Section census MEASURED by the
receiver's own splitter: header 14 · hpac 12,343 · semantic 30,246 · carrier 18,568 · tail 113,515 (= 96 B residual
table + **113,419 B** token stream) = 174,686 B member.

**Consequence for this arm, stated plainly: cl2's pricer no longer describes the object.** A Brotli-packed IHS1 body
would produce a section the live receiver refuses to read, so pricing a rung through cl2's `pack_model` would have
produced a number about nothing. This is the [[m148]] object-change law arriving on my own instrument.

**The re-grounding (MEASURED, and it is an instrument control, not an assertion).** The live tree's own hpac section
is reproduced **byte for byte** — 12,343 B, sha-identical — by driving cl2's retained λ=1.0 control raw IHS1 body
(17,770 B, sha `81728190…`) through the recipe above: 517 channels, 20,416 params, RC1H rider 16,267 B, and a fresh
`restore_hpac` returns the raw body exactly. Two facts fall out of that one control:

1. The recipe is right, so every cl3 rung can be priced on the live object.
2. **The live pointer carries cl2's λ=1.0 control weights**, re-coded — independently confirmed because
   `jg2._prepare` on the live tree returns `parts.token_stream` = 113,419 B, sha `e07274ca…`, which is cl2's control
   stream byte for byte. So the ladder's control IS the pointer, and a cl3 rung competes with it directly.

**The bar, re-derived on the live object:** `J = hpac_container + token_stream`; the pointer's
J = 12,343 + 113,419 = **125,762 B**. A rung pays iff its J is lower. The archive bar is **< 174,786 B**.

**What this does to the pre-registered predictions.** P1–P5 were written in cl2's Brotli currency, which no longer
exists on this object. I did NOT restate them — that would be re-anchoring. Instead every rung records BOTH
containers: the rc1 basis (the decision) and cl2's Brotli q0–q11 race on the same raw body (advisory, so P1/P2/P4/P5
stay scoreable in the currency they were written in). The joint-byte predictions P3/P6/P7 are currency-independent in
sign and were always the operative ones.

**Instrument:** `experiments/ddm_cl3_rc1_rung_price.py` (commits `b5bdc20a1`, `3bbc11e52`) — a thin composition over
cl2's helpers (`_pack_terminal_ihs1`, `replace_hpac_section`, `patch_inflate_pins`, `decode_with_receiver`) and rc1's
codec (`apply_hpac` / `restore_hpac` / `hpac_row_counts` / `ck2_interleave`). Nothing in either is reimplemented. Each
rung: pack → rc1 container (rider losslessness proved) → stage the LIVE tree → encode TWICE (streams must be
byte-identical) → receiver-copy decode to the exact field → section census refusing any move outside the model and
the stream.

### 4c. Matched-epoch trainer surrogate, λ=2.0 vs cl2's λ=1.0 control (ADVISORY, never a row)

The trainer's own telemetry is stamped `byte_authority: ADVISORY_ESTIMATE_NOT_SERIALIZED`, so this is a direction
check, not evidence. It is recorded because it is free and because it says whether the rate multiplier is doing the
mechanical thing before an hour of encode prices it. `depth-sum` is Σ(depth × channels) over the bit-depth histogram —
the model's total allocated width.

| epoch | λ=1.0 est. model | λ=2.0 est. model | Δ model | Δ depth-sum | λ=1.0 est. tokens | λ=2.0 est. tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17,947 | 17,947 | **0** | **0** | 116,544 | 116,544 |
| 10 | 17,488 | 17,177 | −311 | −53 | 129,606 | 133,626 |
| 20 | 17,483 | 17,095 | −388 | −53 | 135,510 | 131,993 |

Epoch 0 is byte-identical on both runs, which is the shared warm start doing its job and a free determinism check.
From there λ=2.0 holds a strictly smaller model — 62 zero-width channels against the control's 54, and mass moved out
of depths 6–8 — which is exactly the rate multiplier biting on the coordinate it is defined on. The −388 B at epoch 20
sits inside P1's −250…−400 B band, though on a THIRD basis (the trainer's surrogate) that is neither cl2's Brotli
container nor rc1's, so it scores nothing; only the exact container price does. The token estimate is noisy epoch to
epoch (the control itself moved +5,904 B between epochs 10 and 20), so no reading is taken from it here.

### 4d. A SECOND prediction, pre-registered at 23:33Z — BEFORE the first exact price

The QAT phase closes the surrogate gap monotonically: joint **+1,886 B (ep 40) → +284 B (ep 50)**, the model saving
growing (−554 → −603) while the token tax collapses (+2,440 → +887). λ=2.0 is on a knife edge on the surrogate, which
is exactly the case the exact price exists to settle. Two things follow that I want on the record before I can see the
answer:

* **P8 (new, pre-registered):** the **rc1 basis should show a LARGER model saving than the Brotli basis** for this
  rung. Reason: λ=2.0 moves mass to low depths and zero-width channels (67 zero-width at ep 40/50 against the
  control's 54 at start), and rc1's adaptive per-group tree coder models the code alphabet directly whereas Brotli is
  a generic byte coder that never sees a code boundary — so the structure λ=2.0 creates is precisely the structure the
  adaptive coder can exploit and the generic one cannot. Both containers are measured per rung, so P8 is scored
  directly. If it holds, the smaller-prior direction is worth MORE on the live object than cl2's currency would have
  shown, and a rung could pay on the rc1 basis while losing on the Brotli one.
* **Honest counterweight:** the surrogate at ep 40 and ep 50 points AGAINST P3 (joint still positive). I record the
  unfavourable epochs, not only the favourable ep-20 one. If the exact price lands positive, P3 is falsified and the
  pre-registered whole-axis FALSIFIER fires: the capacity axis closes in BOTH directions on this vehicle and λ=4.0 is
  NOT run.

## 5. The ladder — NOT MEASURED (state as of 2026-09-05T16:35Z)

**No rung has been trained, priced or verified.** There is no ladder table, no residual against P1–P7, and no seal.
Every λ=2.0 launch was stopped in its first 8 minutes (§3), and the arm is now held OFF Metal by MAIN's rule until
md3's cell finishes (~19:00Z). Saying this plainly: the charter's deliverable is unmet so far. The apparatus, the
predictions and five measured setup findings are real work, but they are MEANS — the END is a ladder row, and there
is none.

The one number this arm can already state about the ladder is the bar, re-derived at the live pointer: a cl3 rung must
land **archive < 179,982 B (joint < 126,885 B)** to move the pointer, and **≤ 138,205.2 B** to reach the rate corner.

## 5b. Ready-to-fire state (resumable from disk)

Runbook `/Volumes/VertigoDataTier/pact/ddm_cl3_hpac_smaller_prior_and_seed_selection/RUNBOOK.md` carries the exact ten
steps and the pinned seal command. Rung root is clean (0 dirs). Inputs re-hashed and still pinned at
16:31Z: field `cc10a7b0…`, init `ff2d3e45…`, fs2 archive `a8f3a379…`. Every module the price and verify stages import
resolves (`jg2.encode_tail`, `rx2._pack_terminal_ihs1`, `up3.parse_shipped_body/build_archive`). Every flag of the seal
command was checked against its argparse. Checkpoint `tools/subagent_checkpoint.py read --subagent-id ddm_cl3`.

## 6. Decision rule (pre-registered)

Winner = min J over {cl2's λ=1.0 control, the cl3 rungs}. If the winner is NOT the cl2 control: twin it (a fresh-root
retrain of the exact winning law; the twin's stream and archive shas must match byte for byte, as cl2 did for its
control), then produce the contest-CUDA seal with `tools/make_candidate_seal.py` and hand it to MAIN. A candidate must
beat **179,982 B** to be a pointer move, and **≤ 138,205.2 B** would be the rate corner. This arm dispatches no Modal.

## 7. Equations leg (`tac.canonical_equations`)

This arm adds no new law. Every cl3 row is an anchor on cl2's registered
**`hpac_prior_capacity_slope_v1`** (`src/tac/canonical_equations/hpac_prior_capacity_slope_20260905.py`; guards in
`src/tac/tests/test_ddm_cl2_hpac_prior_capacity_slope.py`): with the field held, the token subsystem's counted bytes are
`J = B_model + B_stream`.

**A precision the charter's one-line framing drops, re-derived here before reading any cl3 number.** "A rung pays iff
`ΔB_stream/ΔB_model < −1`" is true only when `ΔB_model > 0`. Dividing `ΔB_model + ΔB_stream < 0` by a NEGATIVE
`ΔB_model` flips the inequality, so on the smaller-model side the same primitive gives `slope > −1`. Reading a cl3 rung
through the bigger-side rule would invert its verdict. Checked at the source: `rung_pays` already branches correctly
(`if delta_model_bytes > 0: slope < break_even; else: joint fell`), so the registered law is direction-safe and only the
prose is bigger-side-only. **Every cl3 rung is therefore judged on the primitive, `ΔJ < 0` against the control** — which
is also the currency prediction P3 is written in. cl2 anchored the law on
the BIGGER-model side (secant λ 1.0→0.5, slope +0.446 — does not pay). cl3 anchors it on the SMALLER-model side
(λ 1.0→2.0, and 1.0→4.0 if the first pays), which is the same law evaluated at the opposite sign of `ΔB_model` and is
therefore a genuine out-of-sample test of it, not a re-fit.

The seed rungs are deliberately NOT slope anchors: they hold λ fixed, so `ΔB_model ≈ 0` and the ratio is undefined. They
measure the law's own NOISE FLOOR — the run-to-run spread of `J` at a fixed law — which is what says whether cl2's −41 B
control is a mechanism or a draw. `hpac_prior_capacity_slope_v1.control_reproduces_shipped_family` is the admissibility
clause each rung is checked against.

## 8. Frontier line

**cl2 λ=1.0 control repack — S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]**, archive
`08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e` — the LIVE `effective_frontier`, moved by cl2 at
16:08:42Z (superseding fs2's 0.14784474152757654 @ 180,023 B). **ddm_cl3 has moved no pointer.**
