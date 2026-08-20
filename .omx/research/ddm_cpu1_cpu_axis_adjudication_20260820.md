# ddm_cpu1 — the CPU axis is not a slower CUDA axis, it is a different ground truth

**Task:** Part 1 of the operator-approved CPU-vs-CUDA declaration plan.
**Bytes:** archive `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`
(180,625 B), shipped runtime tree `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`.
**Payloads:** `/Volumes/APDataStore/pact/ddm_cpu1/retained/`.
**Effective frontier: UNMOVED** — `0.14839100138338618` `[contest-CUDA T4]`. No byte changed.

---

## ANSWER FIRST

**The whole CPU-vs-CUDA penalty on the jg5 bytes is the ground-truth DECODER, not the
CPU.** `upstream/evaluate.py:39-42` selects the GT dataset class **by device**:

```python
if device.type == "cuda":  DefaultDatasetClass = DaliVideoDataset   # NVDEC
else:                      DefaultDatasetClass = AVVideoDataset     # PyAV
```

So the two axes do not merely run different kernels over the same problem — they score
against a **different decode of the ground-truth video**. Our vehicle's pose was solved
against the DALI lineage, so the CPU axis pays the full distance between the two GT tables.

Measured here, n600, on the retained jg5 raws: **one** macOS-CPU scorer forward pass over the
candidate, with only the GT table swapped, reproduces **both** official rows.

| GT lineage swapped in | d_seg | d_pose | S |
|---|---:|---:|---:|
| PyAV (`AVVideoDataset`, what CPU uses) | `3.4740024143e-04` | `1.4701090981e-04` | **0.19335280** |
| DALI (`DaliVideoDataset`, what CUDA uses) | `2.0134819878e-04` | `6.3658738313e-06` | **0.14838424** |

- reproduces the jg5 **macOS-CPU advisory** row `0.19335266` (seg `0.00034740`, pose `0.00014701`) — **exact to 8 dp on both components**;
- reproduces the jg5 **`[contest-CUDA T4]`** row `0.14839100` (seg `0.00020139`, pose `6.37e-06`) to **−4.2e-08 seg / −4.1e-09 pose**.

**Attribution: `+0.04496856` of the observed `+0.04496166` gap = 100.02%. Residual for
kernels, device and inflate together: `−6.9e-06`,** which is 0.015% of the gap and inside the
8-dp reporting bound the two receipts carry. There is no measurable CPU-numerics penalty on
this vehicle.

---

## 1. Why this row had to be bought

`ddm_rr7` measured, on this exact archive, a local→T4 transfer whose speedup did not merely
fall short but **changed sign**. Every standing CPU input to the declaration decision was a
transfer of that same shape:

| source | number | what it actually is |
|---|---|---|
| packet `PACKET_TARGET.json:137` | contest-CPU inflate `3,422.711146813 s` | **inherited from gen-3**, labelled `inherited, not measured` (nv1 claim-10) |
| MC36 lineage read (2026-08-14) | CPU `0.20513189` vs CUDA `0.16193`, decode `831.5 s` | different bytes, different vehicle generation |
| jg5 macOS-CPU advisory | `0.19335266`, inflate `978.9 s` + evaluate `419.0 s` | `[env-mismatch advisory]`, Apple Silicon, **never a contest axis** |

None is a contest-CPU row on the shipped bytes. The two CPU wall figures in play differ by
**3.5x** and neither was measured on this archive.

---

## 2. The contest-CPU row: the WALL is measured, the SCORE is not

Call `fc-01M0FGBV7547NWJVJWQ8W3YX76`, Modal contest-CPU worker, Linux x86_64, torch `2.5.1+cpu`,
**4 torch threads** (the contest runner's CPU count; 24 logical were available and deliberately
not used), `rc=1`, `passed=false`.

**`upstream/evaluate.py` never ran.** The harness's own `--inflate-timeout` fired at **1,800.0 s**
— the contest budget — and raised. So there is **no contest-CPU score**, and none is claimed.

**But the wall IS measured, and precisely.** `subprocess.run` killed the `bash` parent; it did not
reap the decoder underneath, which ran to completion and wrote its own report: the full
**3,662,409,600 B** field, `pair_count 600`, at **4,369.600210089 s**. The wall below is that
receipt's own arithmetic, not an extrapolation from a truncated run.

| stage | seconds | share |
|---|---:|---:|
| `token_decode_or_checkpoint_load` | **3,966.804** | **90.8%** |
| `neural_render_and_resize` | 395.738 | 9.1% |
| `frame0_selector_and_io` | 3.300 | 0.1% |
| `archive_setup` | 0.230 | 0.0% |
| **inflate total** | **4,369.600** | 100% |
| *(harness timeout fired at)* | *1,800.000* | — |
| *(Modal container elapsed)* | *4,376.646* | — |

### 2.1 Against the CUDA row on the SAME bytes

| | contest-CUDA (jg5) | **contest-CPU (this row)** | ratio |
|---|---:|---:|---:|
| inflate | 1,419.904 s | **4,369.600 s** | **3.077x** (+2,949.7 s) |
| token decode | 1,341.540 s | **3,966.804 s** | **2.957x** (+2,625.3 s) |

### 2.2 The budget verdict

| frame | band | value | verdict |
|---|---|---:|---|
| **A.** canonical `tac.contest_budget` contest-CPU residual | `[1,044, 1,332] s` | 4,369.6 s | **REFUSE by 3,037.6 s** |
| **C.** absolute CI job wall | `1,800 s` **total** | 4,369.6 s | **REFUSE — inflate ALONE is 2.428x the entire wall** |

Frame B is not quoted: it corrects a residual band by the *measured* evaluate, and evaluate never
ran. Quoting it would be inventing the number the run refused to produce.

**This is not a marginal miss.** The CUDA row fits the job wall with 328.7 s to spare. The CPU
inflate overruns the whole wall by a factor of 2.4 before the evaluator starts.

### 2.3 It supersedes the inherited figure — in the harder direction

`nv1` claim-10's `3,422.711146813 s` was `inherited, not measured` from gen-3. Measured on the
shipped bytes: **4,369.600 s**. The inherited figure **understated the real cost by 946.9 s
(+27.7%)**. The caveat closes, and it closes against us.

### 2.4 What DID reproduce, exactly

| | value | cross-axis |
|---|---|---|
| `decoded_token_sha256` | `cc10a7b09353c0af…` | **IDENTICAL to rr7's T4 decode** |
| `decoder_bit_position` | `910837` | **IDENTICAL** |
| `archive_sha256` | `f3bce5d2…` / 180,625 B | the pre-registered PRIMARY falsifier — **PASS** |
| `raw_sha256` | `aff13c89…` | a **THIRD** distinct raw (T4 `6bf8acf8…`, macOS-CPU `7246a4ff…`) |

**Token decode is bit-identical across axes; the neural render is not.** Three platforms produced
three different raws from one archive. That also corrects the charter's premise that the retained
raws are "byte-identical" — they are not, and §3.4 bounds what that costs the attribution.

### 2.5 The worker confirms the GT-lineage finding at source

The contest-CPU harness logged, unprompted:

```
[contest_auth_eval] gt_lineage: PYAV_YUV420_TO_RGB via AVVideoDataset (authority=False)
```

That is §ANSWER FIRST, printed by the contest instrument on the contest CPU worker.

### 2.6 The score, DERIVED — and labelled as such

Because the CPU worker's GT lineage is PyAV, and because the PyAV leg of §3 reproduces the jg5
macOS-CPU advisory row **exactly at 8 dp**, the contest-CPU score on these bytes is
**`0.19335280` (DERIVED)** — `d_seg 3.4740024143e-04`, `d_pose 1.4701090981e-04`, rate unchanged.

**This is NOT a `[contest-CPU]` row.** It is a derivation with a measured mechanism. It must never
be quoted with a contest axis tag. What makes it decision-grade is not its precision but its
direction: it is **+0.0450 above our CUDA row and +0.0314 above the PR135 bar of 0.162**, and the
whole of that gap is the GT decoder, which no engineering on our side can remove.

---

## 3. The attribution, and why it is trustworthy

### 3.1 The design

`DistortionNet.compute_distortion(gt, comp)` runs the frozen scorer over **both** sides. The
candidate side does not depend on which GT you compare against. So the candidate's scorer
outputs can be computed **once** and then scored against **both** GT lineages. That isolates
the lineage term exactly, for one forward pass, with the candidate held byte-fixed.

Tool: `experiments/ddm_cpu1_gt_lineage_attribution.py`. Every input is pinned by **content
hash, never by path** — the `ddm_dg1`/`ddm_na10` cure applied at the load site, because the
basename `gt_first6_n600.npy` exists at two shas with **opposite** lineages.

### 3.2 The positive control

The confound-gate discipline requires a known-effect signal the apparatus must register, or no
verdict is admissible. The PyAV leg is that control: it has a published answer from the
official path. It reproduced `0.00034740` / `0.00014701` **exactly at 8 dp**. The instrument is
therefore reading the objects the evaluator reads, and the DALI leg is credible for the same
reason.

### 3.3 The two legs of the gap

| leg | Δ contribution | share | mechanism |
|---|---:|---:|---|
| pose | `+0.03036066` | 67.5% | GT-decode lineage: `d_pose` moves `6.37e-06 → 1.4701e-04` |
| seg | `+0.01460100` | 32.5% | GT-decode lineage: `d_seg` moves `2.0135e-04 → 3.4740e-04`, ratio `1.7254x` |
| rate | `0.00000000` | 0% | identical bytes |

**Pose obeys the additive law, and the law's MECHANISM is measured here rather than cited.**
Writing the candidate's pose error against the DALI table as `e = P − B`, the decomposition is
exact by algebra:

```
mean‖P − A‖²  =  mean‖A − B‖²  +  mean‖P − B‖²  +  2·mean⟨e, B − A⟩
                       C                d_pose_dali          cross
1.470109098135e-04 = 1.4061509400e-04 + 6.365873831275e-06 + 2.9941985e-08
```

It closes **to the last digit** on the retained payloads. The additive form holds because the
**cross term is essentially zero (2.99e-08, 0.02%)** — the candidate's residual pose error is
orthogonal to the offset between the two GT tables. That orthogonality is *why* `C` behaves as
a floor, and it is the part `ddm_pi2` asserted and this row measures.

**The ratio form is meaningless and this row shows why.** Per pair, `d_pose_pyav / d_pose_dali`
spans **0.530 to 983,016** (median 54.1). The per-pair offset spans `−1.31e-05` to `1.57e-03`.
Only the **population mean offset** is the invariant, because `C` is a population MSE between
two tables — a property of the clip, not of the candidate. Quote `Δd_pose` absolute; never a
multiplier.

**Canonical equation:** `cw1_gt_lineage_additive_pose_offset_v1`
(`tac.canonical_equations.ddm_cw1_win_family_laws_20260819:pose_pyav_from_dali`). This row lands
as its third `EmpiricalAnchor`, `cw1_gt_lineage_offset_cpu1_frontier_mechanism_20260820`
(residual `3.50e-08`) — the first anchor taken on the sub-0.15 frontier bytes, and the first to
measure the law's **mechanism** (the orthogonality above) rather than only its value.

**Seg is bounded by the same object.** The two GT argmax fields disagree on `1.7523e-04` of
pixels; the candidate's seg lineage term is `1.4605e-04`. The triangle inequality holds with
room to spare, so the seg gap is likewise a property of the GT decode, not of the candidate.

### 3.4 What this does NOT license

- It does not say CPU and CUDA kernels are bit-identical. It says their contribution to **this
  vehicle's score, at this operating point, at 8-dp reporting precision** is `≤ 7e-06` and is
  not separable from rounding. A vehicle with a larger `d_pose` would show a different balance.
- The candidate raws used are the **macOS-CPU inflate** output (`7246a4ff…`), not the T4 inflate
  output (`6bf8acf8…`). Those raws are **not** byte-identical; the brief's "byte-identical raws"
  premise is wrong and is corrected here. `ddm_pi2` bounds the inflate-device contribution to
  `d_pose` at `≤ 5.576e-09`, and the residual measured above is consistent with that.
- The DALI reference field is `gt_argmax.npy` `91d3ff11…`, classified `DALI_NVDEC` by `ddm_gl1`
  on **nearest-ruler evidence**, not a producer receipt. `gl1` also measured **1,644 differing
  sites between two DALI builds** (`1.39e-05` of the field) — within-family drift is nonzero
  and is roughly 300x the `−4.2e-08` seg residual above, so it comfortably bounds it. The
  residual is therefore not evidence of a kernel effect; it is smaller than the reference
  uncertainty.
- It is `[macOS-CPU advisory]`, a decomposition of an existing advisory row. It is not a score,
  not promotable, and does not move any pointer.

---

## 4. The decision

**Declare `linux-nvidia-t4`. The CPU axis is not a fallback on this vehicle — it is worse on
both axes, and the wall half is now measured rather than inherited.**

The plan's decision rule had two branches. Neither survives contact in the form it was written:

| plan branch | outcome |
|---|---|
| "CPU score ≥ ~0.162 → declare `linux-nvidia-t4`, **record CPU as the wall-comfortable fallback with its measured price**" | First half HOLDS (CPU ≈ 0.1934 DERIVED, +0.0314 over the bar). Second half is **REFUTED**: CPU is not wall-comfortable. Its measured price is 2.428x the entire CI job wall. |
| "CPU surprisingly close to CUDA → CPU declaration becomes viable per operator preference" | **Does not fire.** CPU is 0.0450 worse, and the reason is structural (a different GT decode), not tunable. |

**Why the operator's "CPU is preferable" does not survive on these bytes**, stated plainly:

1. **Score.** The CPU axis scores against a PyAV decode of the ground truth; our pose was solved
   against the DALI decode. The `1.4062e-04` distance between those two GT tables is a property of
   the clip and the decoders. We cannot engineer it away without re-solving the carrier against
   the PyAV GT — which would then lose the same amount on CUDA.
2. **Wall.** The token decode is 90.8% of CPU inflate and runs **2.957x** slower than the CUDA
   path. There is no headroom to buy: the stage that dominates is the stage that loses.
3. **And the two cannot be traded.** Declaring CPU costs +0.0450 *and* overruns the wall. There is
   no operating point on this vehicle where CPU is the better declaration.

**What would change this verdict.** A carrier re-solved against the PyAV GT lineage would move the
score half; only that. The wall half needs the token stage to get roughly **3.3x** faster on 4 x86
vCPUs — and `ddm_rr7` already measured the native port going the WRONG way on contest-class vCPUs:
**0.867x on the TOKEN stage** (1,341.5 s CUDA-python vs 1,546.6 s native-scalar-CPU, i.e. 15.3%
slower on that stage; the whole-inflate ratio is 13.6% — both denominators stated per `ddm_pq8`).
Both would have to land. Neither is close.

**Not owed, and deliberately not fired:** a re-run with `--inflate-timeout 7200` would buy the
CPU score row for another ~$0.15-0.30. It is **not worth it for the declaration**, because the
wall REFUSE already decides it and the score's direction is not in doubt. If MAIN or the operator
wants the row anyway — to close `nv1` claim-10's score half as well as its wall half — the command
is one flag off the sealed fire order:

```
.venv/bin/python tools/fire_modal_auth_eval.py \
  --seal /Volumes/APDataStore/pact/ddm_cpu1/CANDIDATE_SEAL_cpu1.json \
  --output-dir <new dir> --lane-id <new lane> --instance-job-id <new job> \
  --claim-agent <arm> --single-axis-waiver-reason "<reason>"
```
plus `--inflate-timeout 7200` threaded to the worker. The seal is still valid and its admit bar is
**not stale** (the `contest_cpu` pointer is unchanged at `0.1880443979880752` / `196acd18…`).

---

## 5. Packet handoff — exact replacement text for `ddm_pq8`

`ddm_pq3` owns the packet files; this arm did not edit them. The sentences below change, and
the replacement text is given verbatim so the handoff carries no paraphrase. Three sentences
change and one caveat closes.

### 5.1 `README_PUBLIC.md:25-27` — the "What is NOT measured" bullet

CURRENT:

> - **What is NOT measured.** There is **no `[contest-CPU]` row on these bytes**,
>   and none is claimed. This submission is GPU-required for evaluation; the
>   requested runner is `linux-nvidia-t4`.

REPLACE WITH:

> - **What is NOT measured.** There is **no `[contest-CPU]` score row on these bytes**, and none
>   is claimed. The CPU axis was measured on 2026-08-20 (Modal, Linux x86_64, 4 threads) and the
>   evaluator never ran: inflation alone took **4,369.6 s**, **2.43x the entire 1,800 s job wall**,
>   so the harness's 30-minute inflate budget refused first. Token decode is 90.8% of that and runs
>   **2.957x** slower than the CUDA path (3,966.8 s vs 1,341.5 s) on the same archive, which decodes
>   to a bit-identical token stream (`cc10a7b0…`) on both axes. This submission is therefore
>   GPU-required for evaluation, by measurement rather than by projection; the requested runner is
>   `linux-nvidia-t4`.

### 5.2 `FREEZE_CHECKLIST.md:112` — owed item 3

CURRENT:

> 3. **No contest-CPU row on these bytes**, and the axis is expected to be infeasible.

REPLACE WITH:

> 3. **No contest-CPU SCORE row on these bytes** — and the axis is now measured infeasible, not
>    merely "expected to be". `ddm_cpu1` (call `fc-01M0FGBV7547NWJVJWQ8W3YX76`, 2026-08-20) measured
>    contest-CPU inflation at **4,369.6 s** against a **1,800 s** total job wall; `upstream/evaluate.py`
>    never ran. The archive sha (`f3bce5d2…`, 180,625 B) and the decoded token stream (`cc10a7b0…`,
>    bit position 910837) both reproduce, so this is a wall result, not a decode failure.

### 5.3 `ddm_pq3:150-153` — the CPU-path paragraph

CURRENT:

> On the CPU path the projection is **1,414–1,913 s against a `[1,044, 1,332] s` residual** —
> over budget in every corner — and the prior lineage measured contest-CPU inflation at
> 3,422.7 s. No CPU row exists on these bytes and none is claimed.

REPLACE WITH:

> On the CPU path the projection is superseded by measurement. `ddm_cpu1` measured contest-CPU
> inflation at **4,369.6 s** on the shipped bytes (Modal, Linux x86_64, 4 torch threads,
> 2026-08-20) — **REFUSE against the `[1,044, 1,332] s` residual by 3,037.6 s**, and 2.43x the
> entire 1,800 s job wall on its own. The prior lineage's 3,422.7 s figure was `inherited, not
> measured` and **understated the real cost by 946.9 s (+27.7%)**. `upstream/evaluate.py` never
> ran, so **no contest-CPU score row exists and none is claimed**; the score's direction is
> nonetheless settled — the CPU axis scores against a PyAV ground-truth decode while ours was
> solved against DALI, a fixed `1.4062e-04` pose-table distance worth **+0.0450** on this vehicle.

### 5.4 `nv1` claim-10 — the caveat CLOSES

CURRENT (`ddm_nv1_numeric_claims_verification_20260820.md:40-42`):

> 4. **Claim-10 caveat owed at freeze:** the contest-CPU 3,422.711146813 s figure is labeled
>    `inherited, not measured` in `PACKET_TARGET.json:137` — no contest-CPU row exists on the
>    jg5 bytes. Any CPU-axis sentence must carry the inherited label.

REPLACE WITH:

> 4. **Claim-10 — WALL half CLOSED 2026-08-20, SCORE half still open.** `ddm_cpu1` measured
>    contest-CPU inflation on the jg5 bytes at **4,369.600210089 s** (call
>    `fc-01M0FGBV7547NWJVJWQ8W3YX76`, memo `.omx/research/ddm_cpu1_cpu_axis_adjudication_20260820.md`).
>    The inherited `3,422.711146813 s` is **superseded** and understated the cost by 946.9 s. Any
>    CPU-axis WALL sentence should now cite the measured figure and drop the inherited label. A
>    contest-CPU SCORE row still does not exist — the evaluator never ran — so any CPU SCORE
>    sentence must stay labelled DERIVED, never `[contest-CPU]`.

---

## 6. Apparatus: the post-spawn claim transition was structurally unreachable

The fire emitted a `ModalSingleFlightRefusal` traceback while the dispatch itself succeeded.
It is not a false alarm and not specific to the CPU path.

A detached Modal auth-eval dispatch runs: pre-claim `..._spawning` → `assert_modal_single_flight`
→ `.spawn()` → `register_dispatched_call_id_fail_closed` → claim transition to `..._spawned`.
The last step re-ran the guard, whose LEDGER leg excludes nothing by design — *"Live LEDGER
rows are never excluded: a non-terminal call_id on the same lane is exactly the un-harvested
duplicate-breeder this guard exists for."* Correct before a spawn; fatal after one, because by
then the dispatcher has registered **its own** call_id. **The transition therefore always
raised, on every detached auth-eval dispatch, on both axes.**

Confirmed pre-existing, not introduced here: `ddm_rr7`'s own CUDA fire log
(`/Volumes/APDataStore/pact/ddm_rr7/t4_row_r1/dispatch_stderr.log`) carries the identical
traceback, unreported. The dispatch always survived, which is exactly why it stayed silent —
the cost is a claim stranded at `..._spawning` and the suppressed operator-facing
`DISPATCHED DETACHED … / Recover: …` banner.

**Why the suite missed it.** `test_modal_auth_eval.py`'s autouse `_hermetic_single_flight`
fixture stubs the guard so unit tests do not couple to live session state. That isolation is
right — and it removed the only surface that could observe the self-conflict. Twenty-plus tests
also monkeypatch `claim_modal_auth_eval_dispatch` itself, so the post-spawn call was asserted
by **status string** while the real function was never run.

**Fix:** `claim_modal_auth_eval_dispatch(..., pre_spawn_guard: bool = True)`; the two
post-spawn call sites pass `False`. The guard prevents a second *spawn*; after the spawn there
is none to prevent. Default behaviour is unchanged, so no existing mock breaks.

**Observed in passing, for whoever owns it:** at 12:44Z `modal container list` showed TWO live
containers — this arm's `ta-01M0FGBW47HTR6DPBWTVCYPKTR` and a sister's
`ta-01M0FK6220QJZ78YNR18WBC5CR` (app `comma-ddm-sa1-…`). The `01M0FK` prefix orders AFTER
`01M0FGBW`, so the sister fired while this row was live. That is the exact condition the
single-flight guard exists to prevent. Recorded as a fact, not adjudicated here.

**Self-protection** (`src/tac/tests/test_modal_post_spawn_claim_transition.py`, 6 tests):
exercises the **real** guard against a **synthetic** on-disk ledger holding the dispatcher's own
live row — hermetic and real at once — plus an AST assertion that every `..._spawned` transition
in both dispatchers carries `pre_spawn_guard=False`, and a floor test that the pre-spawn guard
call site still exists. The AST test caught an incomplete refactor of mine mid-landing.

---

## 7. Custody

`/Volumes/APDataStore/pact/ddm_cpu1/retained/CPU1_RETENTION_MANIFEST.json` — **16 files,
118,095,745 B**, every sha256 measured from the bytes: the seal, the fire manifest, the spawn
record, the poller arming record, the outer Modal result, all five harvested remote artifacts, the
mirror-unwritten blocker, the attribution result, the graded row, and — the part that matters —
the **PAYLOADS**, not just the scalars: the candidate's pose predictions
(`cpu1_pose_pred_n600.npy`, `81c6b7ee…`), the full seg argmax field
(`cpu1_seg_argmax_n600.npy`, `68f5ad96…`, 117,964,928 B) and the per-pair vectors for all four
legs (`cpu1_per_pair_n600.npz`, `f00afa49…`). Anyone can re-run the GT swap against any other GT
table without repeating the forward pass.

**Two harvest defects recorded, neither load-bearing:**
- `MIRROR_UNWRITTEN.json` — the anchor mirror was correctly NOT written (`score_recomputed_from_components absent`). That is the right behaviour: there is no score, so the row must stay invisible to `tac.frontier_scan`. Recorded because the blocker text reads like a failure and is actually the guard working.
- The remote logs arrive as a Python `bytes`-repr string inside `MODAL_REMOTE_RESULT.json` — the same defect `ddm_rr7` listed as owed item #4, now seen a second time and therefore **not** a one-off. Materialising them needed an `eval` of the bytes literal. The harvester should decode before storing.

`/Volumes/VertigoDataTier` is at 100% (890 MiB free); retention went to APDataStore per the
storage waterfall. **Nothing was measured and discarded.**

---

## 8. Pointer

**UNMOVED.** `0.14839100138338618` `[contest-CUDA T4]`, archive `f3bce5d2…`, 180,625 B. This unit
could not move it and did not try: it bought an axis verdict, not a score. The `contest_cpu`
pointer is likewise unchanged at `0.1880443979880752` — correctly, since no CPU score was produced.

`/Volumes/VertigoDataTier` is at 100% (890 MiB free); retention went to APDataStore per the
storage waterfall. Nothing was measured and discarded: the candidate's pose predictions, the
full seg argmax field and the per-pair vectors for all four legs are retained, not just the
scalars.
