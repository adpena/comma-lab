# ddm_pr11 — adjudication of timing admission on macOS daemon bursts

Date: 2026-09-10
Axis: `[review; retained-receipt re-derivation; macOS-CPU advisory timing; contest-CUDA T4 timing; scorer-free]`
`score_claim=false`
Tokens: `[no-triality] [p0-ledger-ok]`

## Verdict

**Local rule verdict: AMEND. Direct-leg verdict: T4-DIRECT.** The frozen quarter-core rule is correct for ordinary work, but it has become a
gate with no door on this host when it treats the recurring PID-1 Apple maintenance pair
`/usr/libexec/dasd` and `/usr/libexec/syspolicyd` as ordinary competitors. Preserve the 25.0% individual
and ancestor thresholds, the 200.0% aggregate cap, the whole-window scope, the three-sample settle,
the frozen hash, and paused-process custody. Add only the exact, bounded host-baseline burst class in
this memo. Add `mode: "t4_direct"` as an independent authority path for an exact completed cold
public-entrypoint T4 decode.

This does **not** retroactively admit any of the four receipts. A rule fixed after a run cannot confer
authority on that run. Applied counterfactually to the complete retained samples, the amendment keeps
attempts 1–2 refused and would admit attempts 3–4. MAIN must freeze and hash the amended rule before a
new local run. Move 40 needs no new local run for its own timing fact: its retained exact T4 receipt
already supplies all `t4_direct` bindings and records 990.053829427 seconds `[contest-CUDA T4 n600]`,
below the 1,260-second policy limit.

`verdict_scope: FORMULATION` — local timing admission on this 18-logical/6-P-core macOS host and the
named Apple daemon burst class. This is not a claim that these daemons are harmless to other workloads,
hosts, receiver families, burst shapes, or resource axes.
`verdict_scope: INSTANCE` — the four move-40 cold runs and the one retained exact move-40 T4 receipt.

## Answers to the charter questions

| Question | Adjudication | Verdict scope |
|---|---|---|
| Must `dasd`/`syspolicyd` count as ordinary quarter-core contention? | **No, only inside the exact bounded class below.** A3/A4 contain no non-daemon row at or above 25.0%. Their bursts moved from about 6.0 to 8.7 minutes into the run, yet their walls are 783.056806/784.771637 s `[macOS-CPU advisory]`, only 1.714831 s or 0.2190% apart. Stage excess is 0.3938%/0.1535% of wall; no stage exceeds 5%. Across all four runs the wall range is 1.799% and every raw is byte-identical. | `INSTANCE`: these traces. `FORMULATION`: exact-path bounded host-baseline class, not all daemons. |
| If the strict rule stands, how many attempts are expected? | **No finite number under the observed-cycle model.** With period 16 min and burst duration at least 3 min, the longest burst-free gap is at most 13 min = 780 s. The shortest observed decode is 783.056806 s before settle. Thus admissible start-phase measure is zero, `p=0`, and `E[attempts]=1/p=+infinity`. Three 20 s quiet settle samples only worsen the geometry. Because the period is described as approximate, this is a derived result under the observed envelope, not a population theorem; empirically the strict rule is 0-for-4. | `DERIVED` cycle-envelope result; `INSTANCE` four-run empirical result. |
| Is amendment warranted? | **Yes.** Use `decode_wall_clock.admission_rule.v3` exactly as specified below. It separates unavoidable bounded baseline maintenance from operator/agent work without restoring the post-hoc own-median rule. | `FORMULATION`. |
| Should `t4_direct` exist? | **Yes.** Local timing is a proxy for the exact contest hardware. A hash-bound completed cold public-entrypoint T4 decode of the exact archive/runtime is strictly more direct and must not require an unrelated local denominator. | `FORMULATION`, exact contract below. |
| What is rp1's admissible authority? | **(a), inheritance from a valid move-40 `t4_direct` leg, is the selected authority.** The normalized receiver is byte-identical; only `archive.zip` and the two literal pins differ. A future valid calibrated local leg or rp1's own T4 fire would also be admissible, but the refused local ratio is not. | `INSTANCE`: rp1 as chartered. |
| What is RLC1's admissible authority? | **Its own `t4_direct` fire, or a new receiver-matched local/T4 calibrated leg under v3.** It cannot inherit move 40: current receiver digests are `b06e59a6…` versus `6726fd77…`. Its 829.032/831.503 s g3/g4 local runs `[macOS-CPU advisory]` remain refused under the old rule. The shortest present route is its own exact T4 fire. | `INSTANCE`: current RLC1 tree and receipts. |

## Provenance and receipt custody

`A1..A4` are the concurrency receipts, `L1..L4` their assembled local receipts, and `C1..C4`
their frame-checkpoint stores. The checkpoint digest is
`sha256(JSON(sorted[(relative_name, bytes, sha256)], separators=(",", ":")))`; timing claims use the
SHA-bound assembled local receipt because file mtimes are metadata rather than content bytes.
Every `A`, `L`, and `C` path abbreviated below is rooted at
`/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/`; the suffix shown in the table is exact.

| Object | Bytes / entries | SHA-256 / identity |
|---|---:|---|
| `A1` `…/receipts/move40_quiesced/CONCURRENCY.json` | 118,995 B | `f19efa1a1233e477643d92dcb91d37ec09bbeca5878d936b2fdae1ce126a86bb` |
| `A2` `…/receipts/move40_quiesced2/CONCURRENCY.json` | 88,685 B | `040fe2c19399dfe7bd559a7683828273b7d362cba3cbd8e278e3a6af023fd611` |
| `A3` `…/receipts/move40_quiesced3/CONCURRENCY_attempt3_refused.json` | 69,833 B | `63d55d1d3b05783c5cdbc0b0ece7c0562b0fd04541d40704036dc3a594b74202` |
| `A4` `…/receipts/move40_quiesced4/CONCURRENCY.json` | 50,430 B | `f72baecc4d72fcf76aaf3807121f181824d5c4bc74a7f1a36c20510100b9728c` |
| `L1` `…/receipts/move40_quiesced_local.json` | 230,286 B | `9cefbac9f1a003a380d817308f070b6122b99e94303bb78eeac7304fc3bc43fd` |
| `L2` `…/receipts/move40_quiesced2_local.json` | 230,802 B | `edc60626329e7175c86332a3258f60b8cb4aad111dda7671f9e3a0dcf455f02a` |
| `L3` `…/receipts/move40_quiesced3_local_refused.json` | 231,645 B | `6350942c113f4271cbb72768b5c9130f58ea8e137a8807c98dc650e029a9229f` |
| `L4` `…/receipts/move40_quiesced4_local.json` | 231,360 B | `7a446f0db0201ee90caedc86cb10f10a8214611a73167778829e10a5eff77218` |
| `C1` `…/move40_quiesced/frame_checkpoints/` | 49 files / 26,176,953 B | tree digest `80d790b31d6b454297c6f40275aa7f2a5780705529249c711e5b6e35b694b4e3` |
| `C2` `…/move40_quiesced2/frame_checkpoints/` | 49 files / 26,178,028 B | tree digest `531f11274eb7a85fcb2a3be0eb3de9ddbe8890a872a2ba9c4682f85cc9440b6b` |
| `C3` `…/move40_quiesced3_refused/frame_checkpoints/` | 49 files / 26,178,028 B | tree digest `c52066dd021a5538d3f4da986206e489f1397bfd4fa9a4b36caac2acfcc7e6a4` |
| `C4` `…/move40_quiesced4/frame_checkpoints/` | 49 files / 26,178,028 B | tree digest `6ac7e19d5bd303d25dec9d2dbdc5f044a6113f211a7e62c5f7e6bbccd248fc29` |
| `.omx/research/ddm_dwc1_20260910/MOVE40_QUIESCED_LEG.json` | 7,293 B | `97376fa420885563d8ebddacfae91d5d562862e7029663f08e9c3786648a50f8` |
| `.omx/research/ddm_rlc1_20260910/QUIESCED_TIMING_RECORD.json` | 3,064 B | `a8d1e1a781a0c2f80962591288dbcc4ed023113e31766baaec368db8ef75c4ed` |
| `/Volumes/APDataStore/pact/ddm_sj1_t4_compose39_rp1_union_20260910/MODAL_REMOTE_RESULT.json` | 159,313 B | `cd6d5ef5243e26fa1868100a2bbb34efe0d6446b4a7fb9bd1eab7f309f5d3d00` |
| frozen v2 source | commit | `52c4962ccf57d94604fb29080e5f9eab21743640` |
| `--assert-user-activity` follow-up | commit | `857f501a3162979304838391312ab66934aa4a51` |
| v2 frozen rule | canonical JSON SHA-256 | `b94474c6167ef51bd1e721b1928ab2a627b1a1821a448d899dd2c362a4eacdf1` |

All 78 + 64 + 59 + 47 = **248** retained concurrency samples were traversed. No sample-list
subsampling was used.

## Full-sample evidence

| Attempt | Wall `[macOS-CPU advisory]` | Whole-window evidence | Stage diagnostic | Actual frozen verdict |
|---|---:|---|---|---|
| A1 | 797.145979125 s | two venv Pythons at 66.9%; ordinary aggregate 220.7%; one `dasd` burst, 14 active samples, inclusive span 280.416 s, max 96.8% | max stage +22.2545%; total excess 5.658484 s = 0.709843% | REFUSE under raw v1; later post-hoc admission remains invalid |
| A2 | 785.138280209 s | ChatGPT 124.2%; tar 98.3%; two Pythons 100%; git 98.2%; other ordinary work; one `dasd` burst, 14 samples, span 280.428 s, max 97.0% | max stage +9.8866%; total excess 3.250620 s = 0.414019% | REFUSE |
| A3 | 783.056805625 s | no non-daemon row >=25%; one exact PID-1 burst, 10 samples, span 200.292 s; `dasd` max 96.7%, `syspolicyd` max 64.0%, combined max 160.7%; unwaived aggregate max 47.5% | max stage +2.5740%; total excess 3.083534 s = 0.393782% | REFUSE under frozen v2 |
| A4 | 784.771636667 s | no non-daemon row >=25%; one exact PID-1 burst despite `caffeinate -u`, 11 samples, span 220.320 s; `dasd` max 97.1%, `syspolicyd` max 72.7%, combined max 169.1%; unwaived aggregate max 62.7% | max stage +1.1163%; total excess 1.204850 s = 0.153529% | REFUSE under frozen v2 |

The decisive evidence is not “low own-median excess proves no slowdown.” pr10 correctly closed that
claim. The evidence is the matched whole-wall comparison: same receiver, same archive, cold n600,
bit-identical raw `c5a7986c…`, and two daemon-only traces whose burst phase moved materially while wall
changed 0.2190%. That supports a narrow baseline class. It does not support a general impact waiver.

The DAG recall found a different workload where `dasd` plausibly reduced a GPU row's CPU-side feed rate.
That prevents promoting this result beyond the current CPU decode instrument; it does not contradict the
matched A3/A4 result.

## Exact prospective amendment — `decode_wall_clock.admission_rule.v3`

MAIN should implement the following fields in the frozen rule object. The canonical rule SHA covers
this complete object, including `definition`, exactly as v2 does; MAIN computes and records the new SHA
after implementation and before launch.

```json
{
  "schema": "decode_wall_clock.admission_rule.v3",
  "threshold_pcpu": 25.0,
  "visible_pcpu": 5.0,
  "ancestor_cap_pcpu": 25.0,
  "aggregate_cap_pcpu": 200.0,
  "interval_seconds": 20.0,
  "settle_quiet_samples": 3,
  "stage_tolerance": 0.05,
  "decode_threads": 4,
  "host_constraints": {
    "platform_system": "Darwin",
    "machine": "arm64",
    "logical_cpu_count": 18,
    "performance_core_count": 6
  },
  "host_baseline_burst": {
    "schema": "decode_wall_clock.host_baseline_burst.v1",
    "activation_pcpu": 25.0,
    "members": [
      {"comm": "/usr/libexec/dasd", "ppid": 1, "pcpu_max_inclusive": 100.0},
      {"comm": "/usr/libexec/syspolicyd", "ppid": 1, "pcpu_max_inclusive": 75.0}
    ],
    "combined_pcpu_max_inclusive": 175.0,
    "max_bursts_in_window": 1,
    "max_active_samples_per_burst": 15,
    "max_inclusive_span_seconds": 300.0
  },
  "definition": "ordinary competitor = pcpu >= threshold_pcpu for every process outside producer tree, monitor tree, and pid 0; monitor ancestors compete at pcpu >= ancestor_cap_pcpu. The host-baseline allowance is disabled unless platform.system(), platform.machine(), os.cpu_count(), and p_core_count() exactly equal host_constraints. A host-baseline row matches only exact comm and ppid; a burst-active sample has at least one matching row at pcpu >= activation_pcpu. The allowance is valid only when matching rows stay pcpu <= their member caps, matching-row sum is <= combined cap in every sample, active samples form at most one consecutive run, active-sample count is <= max_active_samples_per_burst, and last.monotonic - first.monotonic + interval_seconds <= max_inclusive_span_seconds. If valid, matching rows do not compete and are subtracted from the aggregate; if any condition fails, no matching row is waived. Unwaived summed pcpu competes at >= aggregate_cap_pcpu. Apply to every non-settle sample; count = max over samples; admitted iff count == 0. Start after settle_quiet_samples consecutive samples with zero unwaived competitors; a later allowance failure refuses the final receipt. Stage rates are diagnostic only."
}
```

Normative comparisons and storage semantics:

1. Match all four host constraints exactly before considering the allowance. Then match `comm` by exact
   string and `ppid` by exact integer. Basename, substring, signer, vendor, and “system daemon” matches
   are forbidden. A different host shape, renamed process, or reparented process is ordinary.
2. `>=` is competing for ordinary individual, ancestor, activation, and aggregate thresholds. `<=` is
   allowed for each named cap, the combined cap, burst count, active-sample count, and inclusive span.
3. A burst is a maximal run of consecutive non-settle sample indices having at least one active matching
   row. A one-sample inactive gap creates a second burst. Zero bursts is valid.
4. Validate the burst globally after the window. Until then settle can provisionally ignore a matching
   row only while the observed prefix still satisfies every bound. Any later violation refuses the run.
5. If the global allowance is valid, compute
   `unwaived_other_pcpu_sum = other_pcpu_sum - sum(pcpu of exact matching rows)` per sample and apply
   `unwaived_other_pcpu_sum >= 200.0`. If invalid, subtract zero.
6. Store `host_baseline_bursts`, every matched row, each bound/result, and
   `unwaived_other_pcpu_sum` per sample. Preserve the full visible lists. `competing_process_count`
   remains the maximum unwaived count.
7. Write `margin_time_basis="quiesced"` only when no burst-active sample exists. A passing receipt with
   an allowed burst writes `margin_time_basis="bounded_host_baseline"`; `decode_wall_clock` accepts that
   value only with v3, a matching frozen SHA, `competing_process_count == 0`, and every allowance check
   true.
8. The assembler copies rule, hash, freeze time, start time, and burst adjudication from the producer.
   It accepts no threshold or class override. A mismatch or recount difference is a typed refusal.
9. Stage diagnostics remain non-authority. They must be retained, but neither low stage excess nor an
   own-run median may excuse an ordinary competitor or an out-of-bounds named burst.

### Counterfactual result over the four complete traces

| Attempt | v3 result if v3 had been frozen before launch | Why |
|---|---|---|
| A1 | **REFUSE**, count 3 at `during_0014` | two 66.9% Pythons plus unwaived aggregate 220.7%; the valid named burst cannot excuse them |
| A2 | **REFUSE**, count 5 at `during_0003` | two 100% Pythons, git 98.2%, `sysmond` 71.2%, plus unwaived aggregate 425.9% |
| A3 | **ADMIT**, count 0 | exact PID-1 members; 10 samples/200.292 s; member/combined caps pass; unwaived aggregate max 47.5% |
| A4 | **ADMIT**, count 0 | exact PID-1 members; 11 samples/220.320 s; member/combined caps pass; unwaived aggregate max 62.7% |

These are counterfactual classifications, not authority upgrades. A3/A4 keep their actual REFUSE
status because their frozen SHA is v2.

## Exact `mode: "t4_direct"` leg contract

Use schema `candidate_decode_wall_clock.v2`. The exact move-40 instantiation is below. Future direct
legs have the same fields and substitute only their own re-hashed receipt/runtime/archive identities
and their receipt-derived seconds.

```json
{
  "schema": "candidate_decode_wall_clock.v2",
  "mode": "t4_direct",
  "score_claim": false,
  "candidate_t4_receipt": {
    "path": "/Volumes/APDataStore/pact/ddm_sj1_t4_compose39_rp1_union_20260910/MODAL_REMOTE_RESULT.json",
    "sha256": "cd6d5ef5243e26fa1868100a2bbb34efe0d6446b4a7fb9bd1eab7f309f5d3d00"
  },
  "runtime_dir": "/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/candidate_runtime",
  "archive_path": "/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/candidate_runtime/archive.zip",
  "archive_sha256": "986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857",
  "receiver_sha256": "6726fd77a7c4fa80b70ce37accb30595c1004cdd91b9ef990bf9c12f21c296bf",
  "t4_runtime_sha256": "27fc92e8ee156d01ffe65ceb2e141ac0c98a0ccd7b9096d12aa18919fcb28a39",
  "t4_runtime_digest_definition": "tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest",
  "t4_seconds_field": ["artifacts", "contest_auth_eval.json", "inflate_elapsed_seconds"],
  "t4_archive_sha256_field": ["expected_archive_sha256"],
  "t4_runtime_sha256_field": ["expected_runtime_tree_sha256"],
  "t4_hardware_field": ["artifacts", "modal_cuda_preflight.json", "torch_cuda_device_name"],
  "t4_cuda_available_field": ["artifacts", "modal_cuda_preflight.json", "torch_cuda_available"],
  "t4_timing_scope": "cold_public_entrypoint_decode",
  "measured_t4_decode_seconds": 990.053829427,
  "projected_t4_decode_seconds": 990.053829427,
  "limit_seconds": 1260.0
}
```

Validation is fail-closed and performs all of the following:

1. Re-hash the receipt, current `archive.zip`, normalized receiver, and T4 upload projection. Require
   equality with all four stored hashes/identities. `archive_sha256` must equal the receipt's canonical
   `expected_archive_sha256`; `t4_runtime_sha256` must equal its canonical
   `expected_runtime_tree_sha256`.
2. Require top-level `passed is true`, integer `returncode == 0`,
   `canonical_path == "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda"`, and
   `inflate_sh_rel == "inflate.sh"`.
3. Parse JSON-string artifacts exactly as `_field` already does. Require T4 hardware, case-insensitive,
   in `{nvidia-t4, tesla t4, nvidia tesla t4, nvidia t4}` and require CUDA availability true.
4. Require `artifacts.contest_auth_eval.json.n_samples == 600`. Read only
   `inflate_elapsed_seconds`; evaluation or whole-job seconds cannot substitute.
5. Require a cold receiver report in `artifacts.contest_auth_eval.stdout.log`: pair count 600,
   `checkpoint_resume is false`, `token_decoder.checkpoint_resumed_from_frame == 0`, and
   `token_cache.status == "DISABLED"`. Missing or unparsable cold proof refuses. A future producer may
   expose the same facts as canonical structured fields, but the values and comparisons do not change.
6. Require `measured_t4_decode_seconds == projected_t4_decode_seconds ==` the canonical receipt field,
   finite and `> 0.0`; admit iff it is `<= 1260.0`. A timeout is useful refusal evidence but cannot
   create `t4_direct`, which requires completion.
7. `t4_direct` has no `local_receipt`, `calibration_receipt`, or CPU/T4 ratio. Supplying any is a schema
   error. Timing is not score authority; `score_claim` must remain false.

Move 40 satisfies this contract from the retained receipt: archive
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`, receiver
`6726fd77a7c4fa80b70ce37accb30595c1004cdd91b9ef990bf9c12f21c296bf`, T4 runtime
`27fc92e8ee156d01ffe65ceb2e141ac0c98a0ccd7b9096d12aa18919fcb28a39`, hardware `Tesla T4`, cold
n600 public path, and 990.053829427 s.

### Inheritance from `t4_direct`

The existing `mode: "inherited"` gains these exact semantics:

- `source_leg` must point directly to a fully valid `mode in {"measured", "t4_direct"}` leg; inheritance
  from inheritance is refused.
- The source archive must equal `pointer_archive_sha256`, and that pointer must be current at seal time.
- The candidate archive and candidate runtime are re-hashed. Candidate
  `measure_receiver_digest(runtime_dir)` must equal the source `receiver_sha256`. That digest excludes
  `archive.zip` and normalizes only the two top-level archive-pin literal values; every other code or
  manifest change refuses.
- Store both `source_t4_runtime_sha256` and the freshly computed `candidate_t4_runtime_sha256`. They are
  identity records, not an equality requirement, because archive pins and archive bytes differ.
- `projected_t4_decode_seconds` equals the source direct measured value (or a valid measured-leg
  projection), `limit_seconds` remains 1,260.0, and the source must validate at `<= 1260.0`.
- Store `inheritance_scope="identical normalized receiver code; candidate payload-dependent time not remeasured"`.
  An exact candidate T4 result, when present, overrides inheritance and refuses if over limit.

For rp1, option **(a)** is therefore admissible and selected. Option **(b)** is not currently admissible:
the old move-40 local ratio was withdrawn and A3/A4 cannot be reclassified. Option **(c)** would be
admissible but duplicates an exact source measurement for unchanged receiver code.

RLC1 cannot use (a): its current normalized receiver digest is
`b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d`, not move 40's `6726fd77…`.
Its admissible routes are a new v3 local base plus receiver-matched T4 calibration, or its own
`t4_direct` receipt. The g3/g4 old-rule runs remain diagnostics only.

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus by content for `decode_wall_clock`, `quiesced`, `timing
admission`, `competing_process_count`, `margin_time_basis`, `t4_direct`, `candidate_t4_receipt`, `dasd`,
`syspolicyd`, `daemon burst`, `whole job wall`, and `cold public-entrypoint`; searched
`CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED blocks, design/charter/SPEC files, and task-ledger
rows; and generated/searched the canonical-equation registry.

Beyond the charter's seeds:

1. `ddm_wc2_wall_clock_pass_20260820.md` confirms the 1,800 s public limit covers the whole CI job and
   the internal 1,260 s inflate policy preserves setup/evaluate reserve. This kept the policy limit
   separate from contention classification.
2. `ddm_dwc1_decode_wall_clock_seal_leg_20260910.md` shows the seal already normalizes only archive pins
   for receiver identity and treats payload-dependent timing as unmeasured under inheritance. This set
   the exact direct-inheritance boundary rather than inventing a second receiver digest.
3. `ddm_pr9_second_family_check_rlc1_cure_20260910.md` and the current RLC1 runtime show a receiver
   change. This forbids move-40 inheritance and routes RLC1 to its own T4 fire or receiver-matched
   calibration.
4. The DAG contains a prior different-workload observation that `dasd` plausibly slowed CPU feed into a
   GPU row. This narrowed the amendment to the exact host/comm/ppid/burst envelope; no general Apple
   daemon exemption is allowed.
5. The canonical-equation registry contains general wall-clock/EV and concurrency equations but no
   calibrated process-CPU or daemon-burst admission law for this receiver. No equation supplied a
   threshold or promoted the four local receipts.

I did not find beyond the named move-40 T4 receipt a second cold exact T4 replay of the same object, nor
an admissible direct T4 receipt for RLC1, in the searched scope.

## Boundaries

- Adjudication only: no source, test, receipt, checkpoint, payload, sidecar, pointer, upstream file, or
  SSD object was edited, moved, deleted, or rerun. The only deliverable edit is this memo; the mandated
  checkpoint writer appended the shared progress store.
- No decoder, timing producer, scorer, exact evaluator, local rerun, or paid dispatch was launched.
- The A1/A2 stage numbers come from their SHA-bound assembled receipts. Current checkpoint mtimes are
  mutable filesystem metadata and are not substituted for those retained values.
- `%CPU` is the retained `ps` decaying average. The rule does not observe instantaneous residency,
  frequency, memory bandwidth, I/O wait, or thermals.
- Four runs on one host/day do not establish population causality. The amendment is a conservative
  host-baseline classification with caps and a prospective freeze, not a proof of zero daemon impact.
- A3/A4 are not salvaged. Their v2 rule SHA controls their actual REFUSE status.
- `t4_direct` proves decode timing only. It does not assert score, contest-CPU behavior, or whole-CI
  completion.
- RLC1's 829.032/831.503 s rows are old-rule refused diagnostics and do not clear timing.
- The exact frontier did not move. This memo is apparatus adjudication, not goal progress.

<!-- # FORMALIZATION_PENDING: review-only timing-admission adjudication; the prospective v3 producer
and t4_direct validator are queued, and this memo creates no score row or canonical equation. -->

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / ddm_dwc1`; consumer store:
  `src/tac/decode_timing_concurrency.py`, `src/tac/decode_wall_clock.py`, their tests, and a new frozen
  v3 rule receipt; fire trigger: harvest of this committed pr11 memo before any next local timing.
  Implement the exact v3 class and `t4_direct` contract above, freeze/hash v3 before launch, and do not
  edit or reassemble A1–A4.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / rp1`; consumer store: rp1 candidate seal plus
  `.omx/research/ddm_dwc1_20260910/MOVE40_T4_EVIDENCE.json`; fire trigger: v2 seal validator accepts a
  direct source leg and rp1's staged normalized receiver digest equals move 40's. Build move 40's
  `t4_direct` leg from the retained receipt and inherit it directly into rp1; fire rp1 only after the
  remaining candidate gates pass.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / rlc1`; consumer store:
  `.omx/research/ddm_rlc1_20260910/TIMING_FIRE_ORDER.json` and the RLC1 candidate seal; fire trigger:
  exact RLC1 runtime/archive identity is frozen and the T4 lane is available. Prefer RLC1's own cold
  public-entrypoint `t4_direct` fire; alternatively produce a new v3-admitted receiver-matched local/T4
  calibration. Do not inherit move 40 or reuse g3/g4 as authority.

Frontier unchanged: `composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)`.

## LIVE-HYPOTHESES

- The exact v3 class will produce repeatable local calibration receipts on this host because A3/A4 had
  zero unwaived competitors and matched within 0.2190% despite different burst phase. This is plausible
  but untested prospectively under a frozen v3 SHA.
- RLC1 will clear 1,260 seconds on its own T4 direct fire. Its two refused local runs
  `[macOS-CPU advisory]` agree within 0.3% around 830 seconds and differ from move 40 chiefly by counted
  receiver work; no RLC1 T4 receipt exists.
- rp1 inheritance will remain valid after staging because the chartered delta is only archive bytes and
  two normalized pin literals. The final staged tree/digest equality check has not yet occurred.

## DEAD-ENDS

- Leaving v2 unchanged is closed on this host-cycle model: a 783.057-second decode
  `[macOS-CPU advisory]` cannot fit inside a maximum 780-second clean gap, so the expected strict-rule
  attempt count is infinite.
- Reclassifying A3/A4 after seeing them is closed: their frozen v2 SHA says REFUSE, and pr10 established
  that post-run rule changes cannot create authority.
- Restoring the own-median impact rule is closed: uniform slowdown can move the median, and 37–38% of the
  wall is outside checkpoint-rate coverage.
- A broad “system daemon” exemption is closed: A2 includes `fseventsd`, `sysmond`,
  `knowledgeconstructiond`, and `mds_stores` above 25%, and prior DAG evidence shows daemon impact is
  workload-dependent.
- Treating `caffeinate -u` as prevention is closed: A4 retained the same `dasd`/`syspolicyd` class.
- Reusing move 40's withdrawn local CPU/T4 ratio is closed. The exact T4 fact survives independently;
  the local calibration lineage does not.
- Requiring a local calibration beside an exact completed same-object T4 decode is closed: it adds a
  noisier denominator without strengthening the hardware/runtime/archive binding.
- Letting RLC1 inherit move 40 is closed by receiver-digest inequality. Its own direct T4 receipt or a
  receiver-matched valid calibration is required.
