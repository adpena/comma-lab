# ddm_rv15 — round 2 adversarial review of the 08-20 landing wave

Date: 2026-08-20 · Arm: `ddm_rv15_wave_end_round2` · Cost: **$0**, zero dispatch, zero scorer forwards
Cycle: finding round **2** of the #1157 3-clean-pass cycle (round 1 = `ddm_pq8`, "ready for round-2")
Scope: cd1 · cpu1 · sd1 · pq8 · vr1 · cross-cutting

**COUNTER VERDICT: FINDING. The 3-clean-pass counter RESETS to 0.**

---

## THE ANSWER, FIRST

**The score is real and I could not break it.** `S = 0.14839100138338618` recomputes from its
own components to a delta of **exactly 0.0**, on a Tesla T4, n600, rc=0, `gpu_t4_match=true`,
zero validation errors. The staged packet pins **the same tree the scoring row evaluated**
(`2103073d…`, verified 33/33 by re-hash), the archive on disk is `f3bce5d2…` at 180,625 B, and
**no file in the sealed tree was touched after the seal**. The submission identity chain is closed.

**One defect sits on the submission path and it is the reason this round is not clean.** `ddm_cpu1`
measured the contest-CPU wall at 4,369.6 s and wrote verbatim replacement text for four packet
surfaces that still publish the superseded 3,422.7 s. **Zero of the four were updated**, and the
corrected figure appears **nowhere** in the packet directory. `ddm_pq8` committed five minutes
after cpu1 landed the handoff and did not apply it.

Eleven further findings follow. None overturns the BUILD verdict for the corrector port — the
port scope `P = 917.929 s` is measured on T4 alone and does not depend on any comparison I broke.
What is damaged is the **bar**: `2.03× / 2.77×` is quoted as "a MEASURED bar, not a hope" while
carrying neither the arm's own ±61 s noise band nor a matched instrument.

---

## 1. Findings, severity-ranked

| # | Sev | Finding | Receipt |
|---|---|---|---|
| **F1** | **HIGH** | cpu1 §5 packet handoff **UNAPPLIED**. 4 named surfaces, 0 updated. The public PR body, report and README still publish contest-CPU inflate **3,422.7 s**, which cpu1 measured to understate the real cost by **946.9 s (+27.7%)**. The corrected 4,369.6 s appears in **no** packet file. | `grep -rln "4369\|4,369" .omx/research/ddm_pq1_submission_packet_prep_20260815/` → **empty**. Still-stale: `PACKET_TARGET.json:59,137,304`, `PR_BODY_DRAFT.md:185,263`, `REPORT_PUBLIC.txt:144`, `COMPLIANCE_RUNBOOK.md:139,249,348`, `CONTRIBUTION_ETIQUETTE.md:21`, `SWAP_PROCEDURE.md:157,200`, `GPU_ROUTING_VARIANTS.md:52`, `CPU_AXIS_SEALED_FIRE_ORDER.json:154`. Chronology: cpu1 `9c7d1c9557` **08:15:03**; pq8's last commit `3930dde0a7` **08:20:42** (memo only). |
| **F2** | MED | **The break-even bar carries no noise band.** cd1 §6.1 measures a run-to-run floor of **≥61.4 s (4.6%)** and states "Every second quoted below carries that band." §6.4 and §9.3 then quote `2.03×/2.77×` bare, and §9.3 instructs the next arm that "a port that lands below 2.03× has not cleared anything." Re-anchored on jg5's own published inflate (1,419.904 s / 51.428 s, P scaled at the same 71.7%), the bars are **2.22× and 3.08×**. A port landing at 2.1× would be declared cleared on one anchor and failed on the other. 3.08× is above the midpoint of the memo's own "realistic 2–4×". | Recomputed: frame-B narrow 890.572 → k=2.2237; frame-A charged 822 → k=3.0774. cd1-anchored table reproduces exactly (all 7 rows) — the issue is the missing band, not the arithmetic. |
| **F3** | MED | **"#1162 = optional post-port polish, not shipping-critical" contradicts the arithmetic.** At k=2 — the low end of the memo's own realistic band — the port **MISSES** frame B by **−7.6 s**. `#1162`'s scope is 74–165 s, which covers that gap ~10×. Removing `model_d2h_sync` (74.140 s) alone drops the corrector break-even from **2.03× to 1.75×**. Ordering the port first is defensible; calling the lever that decides the marginal case "not shipping-critical" is not. | cd1 §6.4 table row k=2.0 margin −7.6. My recompute with token −74.140: `917.929(1−1/k)=392.395` → k=1.7467. |
| **F4** | MED | **The two halves of the inversion were measured on unmatched instruments.** The project's own law pins `(code, weights, threads, batch)`. Local: `torch_get_num_threads=6`, `OMP/MKL/OPENBLAS = None` (unpinned). T4: `torch_get_num_threads=1`, `OMP=MKL=OPENBLAS=1`. §6.3's per-core claim — "the T4 container vCPU is 4.4–5.0× slower than an M5 Max core on numpy" — is therefore an **upper bound**, not a measurement. **Direction matters and favours the memo on one leg:** the local model ran with 6 threads, so a matched 1-thread local run would be slower, and the rr6 §2.4 falsification survives *a fortiori*. The corrector's 4.350× is the leg that may be overstated. | `cd1_local_contest_auth_eval_advisory.json` `/provenance/instrument_tuple/threads/*` vs T4 `MODAL_REMOTE_RESULT.json` `/instrument_tuple/threads/*`. |
| **F5** | MED | **`runtime_tree_sha256` names three different values under one field name.** cd1 retention manifest `67dc3e32…`; cd1 T4 row `expected_runtime_tree_sha256 = 1728c2a9…` — **the same instrumented tree, two conventions**; packet MANIFEST `2103073d…` (shipped tree). No crosswalk exists. A reviewer cross-checking the retention manifest against the row it retains sees a mismatch and would reasonably conclude the retained artifact is not the fired one. | All three read directly from the JSON. |
| **F6** | MED | **§7.2 is under-rated: it becomes shipping-critical the moment §6.4's verdict is executed.** The shipped decoder emits no `decode_path`, so verdicts grade `unreported` — and `contest_budget.py:711` leaves `margin_depends_on_unverified_fast_path=False` on that branch. That is the *same silent-flag defect* §4 just fixed for `other`. Today it is harmless (jg5 truly has no native path). A corrector port **creates** one, and the gate will then declare a margin independent of a native path it genuinely depends on. | `contest_budget.py:707-713` (unreported note), `:711` flag stays False; native branch `:718` is the only True. |
| **F7** | MED | **`vertigo_certify_move.take_census:100-103` silently swallows `OSError` on `stat`.** An unreadable source file is dropped from the census → dropped from `rels` → never copied → never in the manifest → never in the equality proof → then destroyed by `rmtree` at `:407`. A genuine silent-data-loss path, and **not** on the memo's HARDENING OWED list. It did not bite this pass (verified below). | Found by the vr1 verification arm; code read at `tools/vertigo_certify_move.py`. |
| **F8** | LOW-MED | **"The round trips were never the problem" overstates the measurement.** `model_d2h_sync` went 0.523 → 74.140 s, so round trips cost **73.6 s = 9.8%** of the 752.084 s T4-vs-local token gap that rr6's reading tried to explain. The correct reduction is "~10%, not almost all" — a decisive demotion of rr6 §2.4, but not "never". This sentence is the stated basis for F3's demotion of #1162. | 1341.540 − 589.456 = 752.084; 74.140 − 0.523 = 73.617; 73.617/752.084 = 9.79%. |
| **F9** | LOW | **False precision on the headline.** `S` is quoted to 17 digits but derived from `avg_posenet_dist = 6.37e-06` (**3 significant figures**) and `avg_segnet_dist = 0.00020139` (5 dp). Pose sensitivity is `5/√(10·p) = 626.6`, so reporting rounding alone gives **S = 0.148391 ± 3.6e-06** — about **6** supported digits. §8's "re-proved to the last ULP" over-claims by ~11 orders. **The sub-0.15 verdict is SAFE**: the margin to 0.15 is 0.001609 = **443×** the rounding band. | Recomputed interval `[0.148387368, 0.148394633]`. |
| **F10** | LOW | **vr1 memo drifts from its own tool and ledger.** (a) Method lists a `SOURCE_MANIFEST` phase that **does not exist**. (b) Method says `rsync -a`; the landed tool uses `copy_and_hash_one` and **never calls rsync**. (c) "728 corpus refs" for `ddm_pfs1_20260729` measures **93 files / 155 occurrences**, and the labels are inverted in magnitude — `dqs1_local_first` is cited ~144× more but is described only as "many". (d) `--referenced-by` was passed **empty** on both moves. (e) "10 cert rows" is 10 **phase** rows over **2** artifacts; no single row is a complete cert. | vr1 verification arm, measured against the ledger and `git grep -o`. |
| **F11** | LOW | **Bare task ids do not resolve in the repo.** `#1157`, `#1158`, `#1162`, `#1163` are **absent** from `.omx/state/canonical_task_status.jsonl` (max id **1029**, 216 distinct). This is the memorialized task-ledger-split genus; the recorded cure — *cite CONTENT, never bare ids* — was not applied. cd1's routing line partially mitigates by naming "sync-elimination". | Parsed all 565 rows. |
| **F12** | LOW | **Contradictory poller markers coexist.** `t4_row_r1/` holds **both** `poller.done` = `"ok"` **and** `poller.failed` = `"PollDeadlineExceeded: deadline 2400s exceeded"`. cd1 §6.5 explains the history honestly, but the two files remain side by side as a trap for any consumer that reads one and not the other. | Both files read. |
| **F14** | MED | **cd1's owed item 1 is real, still live, and no longer blocked — I reproduced it by reading the code.** `tools/fire_local_advisory.py` refuses a non-empty attempt dir at `:166-167`, then `mkdir`s at `:168`, writes the pyshim at `:171` and `ADVISORY_LAUNCH.json` at `:227` — **all before** the dry-run early return at `:228`. So `--dry-run` poisons the very directory it previews, and the next real fire hits its own refusal. This is the **canonical and only sanctioned** local-advisory firer, so it taxes every arm that prudently dry-runs first. cd1 honestly declined to fix it because a sister (`ddm_cpu1`) was live on the same file — **that blocker expired at 08:15**. | Line numbers read directly; write sites `:168`, `:171`, `:227` all precede `:228`. |
| **F13** | INFO | **Token stage ≠ loop + prelude, unnamed both times.** T4: 1,280.093 vs 1,277.869 + 1.240 = 1,279.109 (**0.984 s** unnamed). Local: 676.556 vs 673.293 + 0.324 = 673.617 (**2.939 s**). Small (0.08% on T4) and almost certainly teardown, but the decomposition is presented as complete. | Arithmetic on §1.2 / §6.2. |

---

## 2. Per-target verdicts

### 2.1 `ddm_cd1` — CONFIRMED on the measurement, REFUTED on the bar's presentation

**(a) Score identity — CONFIRMED, exactly.** I recomputed from components:

```
100 × 0.00020139              = 0.020139
√(10 × 6.37e-06)              = 0.007981227975693965
25 × 180,625 / 37,545,489     = 0.1202707734076922
                          S   = 0.14839100138338618
claimed                       = 0.14839100138338618
delta                         = 0.0
```

`n_samples=600`, `gpu_model='Tesla T4'`, `gpu_t4_match=True`, `validation_errors=[]`,
`evidence_grade='contest-CUDA'`, `returncode=0`. `promotion_eligible=False` and
`adjudication_required=True` are correctly set. See **F9** for the precision caveat.

**(b) Family attribution — the no-overlap argument is AIRTIGHT for the port scope.** I read the
staged loop rather than the note. Per group the order is:

```
selected = sparse.selected_logits(...)     # ASYNC on CUDA — enqueues only
base_logits = selected.cpu().numpy()       # BLOCKING d2h — drains the queue
... corrector_group_state / coding_row / observe ...   # pure numpy, CPU
```

`.cpu()` is a hard synchronisation barrier sitting **between** all model work and all corrector
work. No corrector timer can absorb pending GPU compute, so `P = 917.929 s` cannot be inflated by
async hiding — and the same property makes `T_token(k) = T_token − P(1−1/k)` valid, because
removed corrector seconds are removed wall-clock 1:1. **The instrument's caveat is correct and its
consequence is load-bearing in the memo's favour.**

*One incompleteness:* the note names `model_d2h_sync` as the only absorber. It is not.
`orch_boundary_buckets` contains `previous[0].to(device="cpu")` — a second sync — which absorbs
`model_prepare_context`'s GPU work into the **orchestration** family. The tell is in the ratios:
`prepare_context` measures 19.815 → 2.126 (**0.107×**, impossibly fast) while `boundary_buckets`
measures 0.288 → 4.465 (**15.5×**, against a family-typical ~5×). Roughly **3 s** is mis-familied.
That is 0.23% of the token stage and does not touch the corrector. So: "the families neither
double-count nor hide overlap" is true where it matters and slightly too strong as written.

**(c) Break-even arithmetic — CONFIRMED internally, REFUTED as noise-free.** All seven rows of
§6.4 reproduce from `token = 1280.093`, `non-token = 79.757`, `evaluate = 48.685`, `P = 917.929`,
against frame B `[893.315, 1433.315]`. The frame A/B distinction is handled correctly: 822 s is
charged, and §6.4's closing paragraph carries the evaluate term properly. Two notes: the true
frame-B break-even is **2.034×** (displayed as 2.03 with margin 0.0, a display rounding), and the
band is missing — **F2**.

**(d) `classify_decode_path` — CONFIRMED.** The C emits exactly four labels
(`f26_hpac_native.c:732-752`: `scalar`, `neon`, `avx2`, `x86-scalar`). Pre-fix, `scalar` and
`x86-scalar` had no native token and no python token, so both fell to `other` — **two of four,
as claimed**. Post-fix I probed 22 labels: all four C labels → `native_dispatched`;
`scalar-python`, `python-scalar`, `scalar fallback`, `pure-python`, `fallback` → `python_fallback`;
`rust` → `other`; empty/`None`/`unknown` → `unreported`. The ordering exception works. Ambiguous
labels (`python-hpac`) read native, which is the **cautious** direction (native sets the warning
True). `44 passed` across both test files. The consumer test does pin the flag, not the prose.
Residual: **F6**.

**(e) The rr7 explanation — arithmetic CONFIRMED; the "band" is a one-sided bound, and the memo's
own hedge is adequate.** The chain reproduces: 1546.617 − 918.755 = 627.862; 267.229 + 90.365 =
357.594; predicted +270.268; measured 1546.617 − 1341.540 = +205.077; residual **+65.191**.

Is the 61.4 s band *measured* or *asserted*? It is a **legitimate one-sided lower bound**, not a
two-sided band. The instrumented tree can only add work, so a run that came in 61.4 s **faster**
proves noise ≥ 61.4 s. That inference is valid. What it does **not** license is "65.19 sits at the
edge of the band" — a lower bound of 61 s is equally consistent with a noise floor of 200 s, which
would make the residual uninformative. The memo says "read it as a consistency check, not a
derivation", which is the correct hedge. The failure is that §6.1's stronger promise — *every
second below carries that band* — is then not honoured (**F2**). Sensitivity: re-scaling rr7's
corrector by the same 4.6% collapses the residual to ~23 s, confirming it is noise-dominated.

**pq8's F1 denominator fix is arithmetically right:** 1546.617/1341.540 = **1.1529** (+15.3%,
token stage); 1612.6/1419.9 = **1.1357** (+13.6%, inflate). Both denominators now appear on all
three public surfaces (PR body 7/9, README 2/3, report 3/5 mentions) — **CONFIRMED**.

**Custody — CONFIRMED, byte-verified.** `CD1_RETENTION_MANIFEST.json`: **15 files, 240,596 B**. I
re-hashed every one from the bytes: **15/15 present, all byte counts and all sha256 match**. The
bound-in-place 3.66 GB payload exists at the recorded path at exactly **3,662,409,600 B**. The P0
keep-the-payload rule is honoured. The lane claim **was** auto-closed (`14:01:33Z`,
`completed_contest_cuda_exact_eval_harvested`), so §6.5's "the blocker was a symptom, not a claims
defect" is CONFIRMED.

### 2.2 `ddm_cpu1` — CONFIRMED on every number I could test; its handoff was dropped downstream

* **DERIVED score — CONFIRMED.** `100(3.4740024143e-04) + √(10 × 1.4701090981e-04) + rate =`
  **0.19335279929942722** → `0.19335280` at 8 dp. Exact.
* **Label discipline — CONFIRMED, and it is strong.** `0.19335280` appears in **exactly one file,
  twice**, both times tagged `(DERIVED)`, with an explicit "**This is NOT a `[contest-CPU]` row …
  It must never be quoted with a contest axis tag.**" No other surface in the repo carries it. The
  "reproduces the advisory row exactly at 8 dp" claim refers to the **distortions**, and both
  reproduce at 8 dp. Deriving the score from full-precision distortions rather than the rounded
  report fields is the right choice and explains the 1.4e-07 offset from cd1's advisory
  `canonical_score = 0.19335265651220337`. Not a discrepancy.
* **Wall arithmetic — CONFIRMED.** 4,369.6 / 1,800 = **2.4276×**. 4,369.6 − 1,332 = **3,037.6 s**
  REFUSE. 4,369.600210089 − 3,422.711146813 = **946.889 s (+27.7%)**.
* **Platform-scoped raws — CONFIRMED** as internally consistent: three lineages
  (`aff13c89…` Linux-CPU, `6bf8acf8…` T4, `7246a4ff…` macOS-CPU) against an axis-invariant token
  field (`cc10a7b0…`, bit position 910837) reproducing on both axes.
* **The four named consumer surfaces — REFUTED. Zero updated.** See **F1**.

### 2.3 `ddm_pq8` — CONFIRMED

* Staged tree verifies **33/33** against `MANIFEST.sha256` (`OK=33 FAILED=0`), run from the correct
  root (`jg5_sub015_runtime/runtime/` — the manifest paths are relative to it, not to the packet dir).
* **Archive UNCHANGED:** on-disk `f3bce5d259a0…` at **180,625 B** = the scoring row's
  `expected_archive_sha256` / `expected_archive_size_bytes`.
* **The identity chain closes.** The jg5 scoring row's `expected_runtime_tree_sha256` is
  `2103073d739fc3f2…` — **exactly** the packet's pin. The packet pins the tree that was evaluated.
* **The 33-vs-34 count reconciles cleanly:** 34 = 33 manifest rows + `archive.zip` (pinned
  separately by sha). Rows: cpr1 6 + inflate.py 1 + inflate.sh 1 + runtime 18 + runtime/entropy 7.
* **No fix re-touched sealed custody.** `find . -newermt "2026-08-20T06:22:40"` over the sealed tree
  returns **nothing**; the seal is `sealed_at_utc 2026-08-20T06:22:40Z`. Receiver pin `inflate.py`
  `78dc0386…` matches the seal exactly.
* **The tautology fix is genuine.** `d678b60c24` moves the derivation from the manifest's own
  claimed digests to the **freshly measured** digests of the staged copies, so content drift now
  moves the value. Per-file diffs correctly demoted to diagnostic. This converts a check that
  could never fail into one that can.

### 2.4 `ddm_vr1` — CONFIRMED on the reclaim; three memo details REFUTED

Verified at **100% coverage** (2 artifacts, not a sample), including a full re-hash of a 3.66 GB
raw (exact match) and independent recomputation of both manifest digests.

* **75.18 GiB — CONFIRMED.** Ledger sum 78,836,860 KiB; `df` delta 78,837,168 KiB; residual
  **308 KiB**. `df -g` now reports **76** GiB available. That 308 KiB residual is the strongest
  receipt in the audit: it is far too small to conceal a deletion.
* **Real moves, nothing deleted — CONFIRMED.** Both sources are absolute symlinks into the cold
  store; no `.RETIRING` remnants. Exactly one destructive call exists (`rmtree` at `:407`),
  reachable only after verification, `--retire-source`, symlink install and probe.
* **Verify-before-unlink — CONFIRMED.** Verification **re-reads the destination**
  (`build_manifest(dest, …)` → `sha256_file`), never reuses the source hash. `if not verified:
  return _block(...)` precedes the retire block, so `rmtree` is unreachable on mismatch.
* **Payload safety — CONFIRMED.** Content is 99.7%+ deterministic decode output. The only
  payload-shaped files are byte-identical copies of the upstream contest scorer weights and source
  video — free artifacts, not lab payload.
* **REFUTED / gaps:** **F7** (silent `OSError` data-loss path) and **F10** (a–e).

### 2.5 `ddm_sd1` — **INDETERMINATE, unresolved in this round**

I dispatched a dedicated verification arm against the five sd1 questions — re-run the guard, settle
whether the `ssd_only_code=9` snapshot is stale or live, and test whether the "0" is **vacuous**
because the exclusion buckets refined in `249bd5d891` stopped the detector from looking. That arm
had not returned when this memo was written.

**Resolving measurement, named:** run `cf3bb0b561`'s guard from a clean shell; compare its live
count against the SessionStart `ssd_only_code`; and quote the exclusion patterns, then re-run the
guard **with the exclusions removed** to establish what the gauge reads if the cure had not been
applied. Per the project's own detector law, a zero is only trustworthy once the detector is shown
to be capable of reading non-zero. **The 814 → 9 → 0 claim is NOT confirmed by this round** and
must not be cited as verified until that measurement lands.

### 2.6 Cross-cutting — MERGE_HEAD CONFIRMED, ledger routings REFUTED

* **MERGE_HEAD guard (`36f4b29476`): positive control is EXECUTED, not asserted.**
  `test_unguarded_commit_really_does_fabricate_a_false_second_parent` (`:103-118`) builds a real
  merge state, runs a **bare `git commit`**, and asserts
  `len(rev-list --parents) == 3` with the message *"git did NOT record a second parent — hazard
  absent"*. The hazard is demonstrated before the refusals are tested. `8 passed`. This is the
  correct shape and I have no finding against it.
* **Ledger routings — REFUTED. F11.**

---

## 3. The assumption-challenge axis (mandatory)

**The shared assumption the whole wave operates within:**

> *The decode wall is a property of our shipped code, measurable once per axis on a Modal T4
> container, and lowerable by porting hot numpy to C.*

`ddm_cd1` is built on the law that a fraction may not cross a regime boundary — and it applies that
law rigorously to **local → T4**, catching the 59.2%/31.4% transfer before it did damage. It then
applies it **not at all** to **Modal-T4 → contest-T4**, and treats one provider's container as the
shipping axis.

**What violating the assumption changes.** Three provider-specific facts are on the receipt:

* `platform_release = 4.19.0-gvisor` — a user-space kernel. gVisor intercepts syscalls in
  userspace, which taxes memory-management-heavy workloads. The float64 corrector allocates
  temporaries per group, 114,000 times.
* `OMP = MKL = OPENBLAS = 1`, `torch_get_num_threads = 1` — the numeric stack is pinned to one
  thread. `ddm_rr6` **measured** that this codebase's native win is thread-borne (1.007× at one
  thread vs 1.865× at four).
* `platform_processor` is **empty**. The CPU model that the entire 4.35× claim rests on is
  **not recorded**.

**Consequences, stated with their scope.**

1. **The direction survives.** The corrector dominates the T4 token stage at **71.7%**, measured on
   T4 alone. That does not depend on the local comparison, on gVisor, or on the CPU model. **BUILD
   remains the right call.**
2. **The bar does not transfer.** `2.03× / 2.77×` is a property of *this container*. By cd1 §9.4's
   own sentence — "the corrector's cost is a property of the vCPU it lands on" — the arm's law
   forbids carrying it to the contest runner. It is quoted in §9.3 as a hard pass/fail gate anyway.
3. **A free lever is unpriced.** Before a multi-day C port with a 2.03× break-even, the cheapest
   untested question is whether the corrector's numpy touches BLAS at all, and what
   `OMP_NUM_THREADS > 1` does to those 917.929 s. Given that this codebase has already **measured**
   an 1.85× thread-borne effect elsewhere, a $0 local A/B (threads pinned vs unpinned, same tree,
   same archive) should precede the port. It may find nothing — elementwise numpy is single-threaded
   — but "may find nothing" is a prediction, and the arm's own standard is to measure instead.

**Verdict on the axis: the assumption is load-bearing, unexamined, and its violation changes the
BAR but not the DIRECTION.** I record this as INDETERMINATE-with-named-resolving-measurement rather
than a refutation, because the resolving measurement is cheap and has not been run.

---

## 4. Counter adjudication

**Round 2 is a FINDING round. The 3-clean-pass counter resets to 0.**

F1 alone forces it: a landed measurement from this same wave supersedes a figure that four
packet surfaces — including the public PR body and report — still publish, and the replacement
text was written and routed and then dropped. That is a defect on the submission path, not a
presentational preference. F2, F3, F4, F5, F6 and F7 are independently sufficient.

Two rounds of clean passes are now required after the fix batch lands. Per the protocol, this round
also discharged the assumption-challenge axis (§3), so it counts as a complete round.

---

## 5. Ranked MAIN-adjudication queue

| Rank | Item | Why now | Cost |
|---|---|---|---|
| **1** | **Apply cpu1 §5's verbatim replacement text** to `README_PUBLIC.md`, `FREEZE_CHECKLIST.md`, `ddm_pq3:150-153` and close the `nv1` claim-10 caveat; sweep the other 8 stale surfaces (`PACKET_TARGET.json:59,137,304`, `PR_BODY_DRAFT.md:185,263`, `REPORT_PUBLIC.txt:144`, `COMPLIANCE_RUNBOOK.md`, `CONTRIBUTION_ETIQUETTE.md:21`, `SWAP_PROCEDURE.md`, `GPU_ROUTING_VARIANTS.md`, `CPU_AXIS_SEALED_FIRE_ORDER.json`). **Leave the `REVIEW_PASS*_FRESH_EYES.md` files alone** — they are APPEND-ONLY historical provenance and correctly frozen. | F1. Public-facing, understates a disclosed cost by 27.7%, and the text is already written. | $0 |
| **2** | **Resolve sd1.** Re-run the guard; settle stale-vs-live on `ssd_only_code=9`; quote the exclusion patterns and re-run with exclusions removed to prove the detector can read non-zero. | The 814 → 9 → 0 closure is unverified. A vacuous zero is worse than a known debt. | $0 |
| **3** | **Re-state the port bar with its band.** Publish `2.03–2.22× (frame B)` and `2.77–3.08× (frame A)`, or re-anchor both on one inflate and say which. Honour §6.1's own promise in §6.4 and §9.3. | F2. The next arm is instructed to gate a multi-day build on a bare number. | $0 |
| **4** | **Fix `take_census`'s silent `OSError`** in `tools/vertigo_certify_move.py:100-103` — fail closed, and add it to the HARDENING OWED list. | F7. A silent data-loss path inside a tool that deletes. | $0 |
| **5** | **Re-route #1162** from "optional polish" to "measure before choosing k". Record that `d2h_sync` removal alone moves the break-even 2.03× → 1.75×. | F3. It decides the k=2 marginal case. | $0 |
| **6** | **Price the free thread lever** before the port: local A/B, threads pinned vs unpinned, same tree and archive. | §3.3. A $0 measurement that could move the same seconds as a multi-day build. | $0 |
| **7** | **Land the `decode_path` one-liner** in `decode_production_tokens` **as a precondition of the port**, not after it. Consider making `unreported` set `margin_depends_on_unverified_fast_path=True` when any native rung is compiled in. | F6. Becomes shipping-critical the moment the BUILD verdict is executed. | $0 |
| **8** | **Publish a tree-sha crosswalk.** One table: which convention produced `67dc3e32…`, `1728c2a9…`, `2103073d…`, `4c08d20d…`, `a5d23cee…`, over which file set. Stop reusing the field name `runtime_tree_sha256` for three different conventions. | F5. Currently reads as a custody mismatch. | $0 |
| **9** | **Correct the vr1 memo** (F10 a–e) and record `referenced_by` on future moves. | Unsourced "728" in an evidence table. | $0 |
| **5b** | **Fix `fire_local_advisory.py --dry-run`** — choose the shim root (`tempfile` scratch under `--dry-run`, `attempt` otherwise) inside a `try/finally`, and move the `ADVISORY_LAUNCH.json` write below the `:228` return. cd1 wrote the cure; only its blocker stopped it, and that blocker has expired. | F14. Taxes every arm on the only sanctioned local-advisory path. | $0 |
| **10** | **Soften "the round trips were never the problem" to "~10% of the gap"** (F8); band the headline `S` (F9); cite routing content rather than bare ids (F11); reconcile the double poller markers (F12); name the 0.98/2.94 s residual (F13). | Honesty polish; each is one sentence. | $0 |

---

## 6. What I could not break

Stated plainly, because a review that only lists defects misrepresents the wave.

* The **score is real**: exact component identity, T4-matched, n600, rc=0, zero blockers.
* The **submission identity chain is closed**: packet pin = evaluated tree, archive byte-identical,
  nothing touched post-seal, 33/33 re-hashed.
* The **instrument cannot move a byte**, and the reason is structural, not rhetorical: the two
  hard `.cpu()` sync barriers bracket the corrector, so the port scope can neither hide GPU work
  nor be inflated by it.
* **Custody is honest**: 15/15 retained files re-hash exactly; the 3.66 GB bound payload exists at
  full size; the vr1 reclaim moved 75.18 GiB with a 308 KiB residual and deleted nothing.
* The **MERGE_HEAD positive control** demonstrates its hazard before testing its cure — the shape
  every guard in this repo should copy.
* `cpu1`'s **label discipline is exemplary**: a derived number, stated twice, tagged both times,
  quarantined to one file, with an explicit prohibition on ever wearing a contest axis tag.

---

**Own-vehicle frontier: `S = 0.14839100138338618` @ 180,625 B `[contest-CUDA T4 n600]`, UNMOVED.**
This arm is a review; it cannot move the pointer and did not try. It re-derived the pointer from
its own components and found it exact.
