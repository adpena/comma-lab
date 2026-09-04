# ddm_gv1 — governor / memory guard / cell queue / costate organ / ANE

**Arm:** ddm_gv1 (Opus) · **Charter:** `.omx/research/charters/ddm_gv1_governor_memory_guard_controller_polish_20260904.md` (331f89039)
**Operator source:** 2026-09-04, verbatim *"Remember we can fully saturate cpu, gpu, and ane; we can continue polishing the memory guard and governor and costate organ and controller as well"*
**Cost:** $0 · **Pointer:** UNMOVED (apparatus work; it cannot move it)
**Constraints honored:** ng2 + ng3 were LIVE on the Metal throughout; I read their manifests, configs, and `history.jsonl` row counts and never touched their custody, claims, PIDs, or the Metal. `upstream/` and `submissions/semantic_joint_ctxmix/` untouched. No `/tmp` paths.

---

## The headline

The hand-written fire scripts computed admission with a THIRD, uncalibrated memory basis, and they
charged nothing for the growth the live cells still owed. Both are now fixed in one governed
function that the queue driver and the digest share, and the Metal-contention question ("does a
second cell actually make the machine faster?") is now MEASURED rather than assumed.

**The measured law:** at N=2 on this box, concurrency **paid by 11.7%** — 31.2854 steps/min vs a
28.0 steps/min serial baseline (speedup 1.1173, per-cell efficiency 0.5586). One anchor. Not a
scaling law. N=3 is unmeasured and must not be extrapolated.

---

## 1. The memory-basis defect (MEASURED)

MAIN's fire scripts admitted a concurrent cell with inline shell arithmetic
(`/Volumes/APDataStore/pact/ddm_ng3_tau_band/fire/fire_ng3_tau_cell.sh`):

```bash
FREE_NOW=$(vm_stat | awk '/Pages free/…'); INACT=$(vm_stat | awk '/Pages inactive/…')
RECLAIM=$((FREE_NOW+INACT))
[ "$RECLAIM" -ge $((NG3_PEAK_GIB+16)) ] || exit 2
```

That expression is `psutil.virtual_memory().available` recomputed by hand. It is the
reclaimable-BLIND basis that this repo's own STRICT gate
`check_no_raw_virtual_memory_safety_basis` (`src/tac/confound_gates.py:2352`, helper at `:2335`)
refuses in Python — on macOS it counts dirty anonymous pages parked in the inactive queue as
available even though evicting them needs swap. **The shell spelling escaped the gate because the
gate parses Python ASTs.** (I verified the gate: 0 violations repo-wide, and it does not see shell.)

Measured live, twice, minutes apart, with both cells running:

| basis | reading #1 | reading #2 |
|---|---|---|
| shell `free + inactive` (what the fire scripts used) | 13 GiB | 23.06 GiB |
| `tools.mem_basis.conservative_free_gib` (canonical) | 18.12 GiB | 19.76 GiB |
| raw `psutil.virtual_memory().available` | 22.53 GiB | 22.53 GiB |

Three bases, three numbers, and **the shell basis errs in BOTH directions** — 5 GiB *under* the
canonical figure in reading #1, 3.3 GiB *over* it in reading #2. The over-read is the dangerous
direction: it admits a cell the box cannot hold. This is `[[m123]]` (two validators disagree ⇒ the
disagreement is the finding) at the admission surface.

### The second, larger defect: unrealized growth was free

The shell rule charged only the candidate. It charged **nothing** for what the live cells still owed
toward their own declared peaks. Measured at the same moment:

| live cell | declared peak | live tree RSS | **unrealized growth** |
|---|---|---|---|
| ng3 tau_band | 41.50 GiB | 0.60 GiB | **40.90 GiB** |
| ng2 area_cap | 2.40 GiB | 0.51 GiB | 1.89 GiB |

40.9 GiB of committed-but-unallocated memory was invisible to the admission rule. A second cell
admitted against raw reclaimable would have been racing ng3's ramp.

### The cure — `tools/cell_admission.py`

One function, one CLI, two legs, both fail-closed:

```
required   = candidate_peak + Σ(live cells' unrealized growth) + margin
headroom   = reclaimable_canonical − required                      >= 0   AND
ceiling    = operator_ceiling − (true_committed + candidate + Σ unrealized)  >= 0
```

`admit` exits **0 / 2 / 3** (ADMIT / REFUSE / unmeasurable) — rc=2 is the code the fire scripts
already branch on, so they can drop the inline arithmetic and call the tool. The default margin is
16.0 GiB, matching MAIN's hand rule exactly: the apparatus reproduces the operator-blessed decision
rather than silently loosening it. The naive shell figure is still *reported* alongside the
canonical one, so the disagreement stays visible instead of silently differing.

**A third defect found while building it (MEASURED).** Reading the manifest PID's RSS reads the
*supervisor*, not the trainer: the canonical launcher detaches as `_supervise → safe_run → trainer`.
On live ng3 the supervisor alone read 0.023 GiB while the process tree read 0.608 GiB — a **26×
under-read** that would have made every live cell look free and inflated its unrealized growth.
`process_tree_rss_gib` sums the tree.

**Scope note, stated honestly.** Discovery is deliberately broader than "training cell": every live
job launched through the canonical launcher holds real memory and is charged. `is_cell` marks the
subset with a sealed run-config and a step budget. Charging only "cells" would under-protect against
a 12 GiB non-cell job — and there was one alive on this box while I worked.

---

## 2. Metal contention — now measured, not assumed

`cell_admission.py sample` reads each live cell's `history.jsonl` row count twice across a window
(purely observational; the cells are never touched) and appends a row to
`.omx/state/metal_contention_ledger.jsonl` under an exclusive lock.

**MEASURED, 2026-09-04T15:55:37Z, 420.004 s window, launched through the canonical detached launcher:**

| cell | steps in window | steps/min |
|---|---:|---:|
| ng2 area_cap | 108 | 15.4284 |
| ng3 tau_band | 111 | 15.8570 |
| **total, concurrency 2** | 219 | **31.2854** |

Against the serial baseline of 28.0 steps/min: **speedup 1.1173, per-cell efficiency 0.5586.**

`throughput_verdict` admits a further cell only while measured concurrent total ≥ the serial
baseline, and labels its evidence grade honestly: `MEASURED` / `NO_SERIAL_BASELINE` /
`NO_CONCURRENT_OBSERVATION`. The not-yet-measured state ADMITS — a governor that refuses everything
until it has data can never collect any — and the memory leg still fail-closes, so the unmeasured
throughput leg never admits blind on the dimension that can hurt the machine.

### Two honesty caveats, load-bearing

1. **The serial baseline is SECOND-HAND.** 28.0 steps/min was measured by MAIN and stated in the
   charter and memory; I did not measure it and its window length is unrecorded. The ledger row
   carries `provenance: SECOND_HAND…` and `measured_by: claude-main`.
2. **MAIN's earlier concurrent reading was 20 + 15 = 35 steps/min; mine is 15.43 + 15.86 = 31.29.**
   Both are honest. Mine has a longer, recorded window, and the two cells had converged to
   near-symmetric rates by then, which suggests MAIN's asymmetry was ng3 still ramping. The
   difference (35 vs 31.29, ~12%) is itself **larger than any measured noise floor, because no noise
   floor exists** — one 420 s window gives no variance estimate. Any future comparison of two
   speedups closer than that unmeasured floor is unresolvable.

**EQUATIONS LEG:** registered as `metal_concurrency_speedup_gv1_v1`
(`tools/register_metal_concurrency_speedup_equation.py`), one anchor, null hypothesis = "concurrency
buys nothing", residual +3.2854 steps/min, `concurrency_extrapolation_permitted: False`,
`RECALIBRATE_ON_NEW_ANCHORS`. Per-cell efficiency at N=2 is already 0.5586, so the marginal third
cell may be net-negative and **nothing in this anchor predicts it.**

---

## 3. The cell queue driver — and it re-found the ng2 defect on its own

`tools/cell_queue_driver.py` generalizes `experiments/ddm_qbr1_cell_chain.py` from one sealed
six-cell fire order to an arbitrary ordered queue with governed concurrency. It **imports** the
chain driver's claim/authorize/verify primitives, the admission governor, and the canonical re-root
helper — it does not fork a parallel launcher (pinned by a test that greps for `start_new_session` /
`os.fork` / `nohup` and asserts their absence). The queue spec owns every scientific value and every
child argv; the driver only verifies, admits, claims, authorizes, launches, and reads.

Per-cell it checks, **before spending a second of Metal**:

- **THE SEAL LAW** — runs `verify_inputs()` inside the sealed tree's own interpreter and refuses on
  content drift (`PIN_CONTENT_DRIFT`) *or* on paths rooted outside the firing tree
  (`PIN_PATHS_NOT_REROOTED`), naming the cure tool in the refusal message.
- **duplicate done-receipt names across the queue** (`DUPLICATE_DONE_RECEIPT`) — the ng1/ng2
  collision; the launcher refuses to overwrite an existing receipt.
- launcher argv binds this cell's authorized config and its declared receipt.
- storage reserve (via the chain driver's own `storage_preflight`).
- **admission** (memory + throughput) from §1–2.

Blocked cells degrade into plan entries with a `blockers` list; a broken cell never crashes the
queue. `--dry-run` performs every verification and is pinned by a test that makes `fire_cell`,
`place_claims`, and `subprocess.run` raise if touched.

**Phantom-claim discipline.** The claims must go in *before* the authorize and the launch (a launch
placed first would be unclaimed), so any failure after that point would leave two ACTIVE claims for
a cell that never ran. Every downstream refusal therefore carries `placed_claims` and
`claims_need_terminal_row: True` with the exact lane ids — the orphan is **reported, never silent**,
and the caller has what it needs to close them with
`claim_lane_dispatch.py claim --force --status refused_dispatch…`.

### The dry-run against the real ng-cell store (the charter's specified test)

Queue spec reconstructed verbatim from MAIN's two fire scripts →
`/Volumes/APDataStore/pact/ddm_gv1_governor/ng_queue_dryrun_spec.json`; plan →
`…/dryrun_plan_1.json`.

| cell | seal check | admission | ready |
|---|---|---|---|
| ng3 tau_band (**re-rooted** config) | **PASS** — 20 pins, content-identical, paths rooted in the firing tree | REFUSE (headroom −81.93 GiB) | no |
| ng2 area_cap (**original** sealed config) | **REFUSE — `PIN_PATHS_NOT_REROOTED`** | REFUSE (headroom −45.59 GiB) | no |

**The driver independently re-discovered the exact defect that killed ng2's launch in 4 seconds on
2026-09-04** — and it did so in the plan, at zero cost, from the same config. Passing ng3's cured
config while refusing ng2's uncured one is a positive and a negative control in one run: the check
discriminates, it does not just refuse everything.

Both refusals on admission are correct — the box genuinely had ~19 GiB reclaimable and 108 GiB
committed with both cells live.

---

## 4. Costate organ SENSE for live cells

`costate_digest.section_live_cells()`, wired into `build_digest` beside the DDM campaign section.
The digest previously spoke only about TR1-era witness runs and the DDM receipt fleet, so a machine
running two concurrent Metal cells looked idle at SessionStart. Live output:

```
live cells: 2 training cell(s) of 5 governed job(s)
  job  measures_n600 pid=95581 declared_peak=3.0 GiB (not a training cell)
  cell seed_20260902_area_cap_control_native100 [control] 2422/5000 (15.4/min, ETA 2.8 h) peak 2.4 GiB declared / 0.5 GiB live
  cell seed_20260902_tau_band_control_native100 [control] 402/5000 (15.9/min, ETA 4.8 h) peak 41.5 GiB declared / 0.6 GiB live
  admission: another 41.5 GiB cell -> REFUSE (headroom -85.8 GiB, reclaimable 15.1 GiB)
  contention: 31.3 steps/min at concurrency 2 vs 28.0 serial (ratio 1.12; concurrency PAYS)
```

It is SENSE-only (`actuation: SENSE_ONLY`, `score_claim: False`), **never sleeps** (it reads the last
recorded contention row rather than sampling a window — pinned by a test that makes `time.sleep`
raise), and fails open to one honest line rather than blanking the digest.

---

## 5. ANE — the honest answer was RECALL, not a build

Charter item 5 asked whether a CoreML export of the frozen scorers exists, and for the smallest
honest receipt if not. Delegated as a read-only audit. The finding is decisive and it is **better
than a build**: the conversion has already been done and measured, twice, and the lane closed.

**Does a CoreML export exist in-tree? NO** — the only `.mlmodel*` on disk is coremltools' own test
fixture inside `.venv_executorch_spike`. The 2026-07-13 work built its packages in
`/private/tmp` scratch that auto-cleaned (`experiments/results/ane_unlock_followup_20260713/storage_preflight.json`).

**But the conversion ran cleanly and the fidelity was measured:**

- `experiments/.scratch/ane_venv_20260713/a2_bench.log` records SegNet → CoreML at **952/952 MIL ops**,
  all passes green. **`smp.Unet('tu-efficientnet_b2')` has no CoreML op-support blocker — measured, not inferred.**
- Fidelity vs CPU-torch fp32, real weights, **n600**: argmax flip rate **0.08834**
  (`ane_unlock_followup_20260713/measurement_receipt.json`, `label_grade_eligible: false`). That is
  **~90× the entire d_seg budget (~1e-3)**. The ANE is fp16; **fp16 can never be a d_seg authority.**
- The damage is precision, not op substitution: CoreML **fp32** flips 66 / 4,718,592 px (1.4e-5), and
  `precision_only_share_of_torch_vs_fp16_flips: 0.99977` (`r0_exact_three_way_split.json`).
- Speedup vs a matched 1-thread CPU-torch baseline: **3.393×** (CPU_AND_NE), 3.609× (best variant) —
  against a pre-registered **10×** bar. Lane verdict `complete_no_joint_bar`, 0 candidates
  (`ane_unlock_correction_20260713/r5_composition.json`).
- ANE *placement* was never proved (`"CPU_AND_NE requested; ANE placement NOT proved"`), and there is
  no backward: `backward_vjp_reachable_on_ane: false`, 0 selectors.
- **Do not cite the "38.03×" figure** that appears in the bench log — it is `RANDOM_INIT` against an
  unmatched baseline.

**Verdict: the ANE is not a substrate for advisory scorer reads on this problem, and I did not build
one.** Re-running the conversion would cost 30–60 min and yield near-zero new information. The one
live residue is not ANE at all: `coreml_fp32_cpu_and_gpu` is a real, fidelity-passing local
accelerator (heldout flip 1.187e-5, 3.61× forward, 2.29× teacher wall-clock at p=0.78) that fails
only the 10× bar its own lane pre-registered. That is a reusable advisory throughput lever, banked
here rather than rediscovered.

This is `[[m53]]` (negative-existence is the #1 false-claim class) avoided in the useful direction:
the honest receipt was that the measurement already existed.

---

## What landed

| surface | file | tests |
|---|---|---|
| memory-guard admission + contention telemetry | `tools/cell_admission.py` | `src/tac/tests/test_cell_admission.py` (57) |
| ordered cell queue driver | `tools/cell_queue_driver.py` | `src/tac/tests/test_cell_queue_driver.py` (57) |
| costate organ live-cell SENSE | `tools/costate_digest.py::section_live_cells` | `src/tac/tests/test_costate_digest_live_cells.py` (12) |
| equations leg | `tools/register_metal_concurrency_speedup_equation.py` | registry read-back verified |

**131 new tests, all passing; ruff clean on every file.**

Regression check (207 passed / 1 failed): the single failure,
`test_costate_digest_ncde.py::test_section_omitted_on_short_telemetry`, is **PRE-EXISTING** — proved
by measurement, not inspection: it reproduces identically after `git checkout` of my
`costate_digest.py` change, and my diff there is `+131/-0` with zero lines touching `ncde`.

### What the second review pass actually found

Three real defects, each fixed and pinned by a test:

1. **The throughput leg was silently vacuous whenever an unrelated job was alive.** It counted every
   live governed job, so with 2 cells + 1 unrelated job it demanded a concurrency-4 ledger row that
   would never exist, and the leg always admitted. Memory charges every job (they all hold RAM);
   throughput now counts only training cells contending for the Metal. This is `VACUITY==PASS`
   caught in my own code.
2. **The queue driver walked the SSD tiers once per queued cell.** Beyond being slow, it judged cell
   1 against a different machine snapshot than cell N. One snapshot per plan now.
3. **The fire path could orphan lane claims silently** (see the phantom-claim discipline above).

Durable artifacts (SSD tier, not `/tmp`): `/Volumes/APDataStore/pact/ddm_gv1_governor/`
(`ng_queue_dryrun_spec.json`, `dryrun_plan_1.json`, `dryrun_plan_2.json`, `rate_sample_1/`,
`regression_1/`).

`.omx/state/metal_contention_ledger.jsonl` (2 rows) is **gitignored LIVE_STATE** — the serializer
correctly refused to commit it. The durable, committed record of the measurement is this memo's
verbatim numbers plus the anchor inside `metal_concurrency_speedup_gv1_v1` in the canonical
equations registry. The ledger itself rebuilds from `cell_admission.py sample`.

## What did NOT land, and why

- **No cell was fired.** The box was genuinely full for the whole unit (headroom −45 to −86 GiB with
  ng2 + ng3 live), so every admission legitimately REFUSED. The driver's fire path is built and
  structurally tested but has never fired a real cell — that is an **honest untested-in-anger path**,
  not a verified one. The first real fire should be watched.
- **No N=3 contention row**, so the throughput leg has never actually *refused* on measured evidence.
- **No first-party N=1 baseline.** The serial anchor stays second-hand until someone measures a solo
  cell with a recorded window.
- **The shell fire scripts still contain their inline arithmetic.** They live in the SSD cell stores,
  not in git, and rewriting a live cell's fire script was out of scope while it was running. The
  replacement is one line: `.venv/bin/python tools/cell_admission.py admit --candidate-peak-gib <P>`
  (rc=2 on refuse, the code they already branch on).
- **No STRICT gate for the shell spelling of the raw-vm basis.** The Python gate exists and is clean;
  the shell surface is unguarded. That is the natural next self-protection landing and it is named
  here rather than silently left.

## Cross-references

`[[saturate_cpu_gpu_ane_governor_is_governance_not_hardware_20260904]]` (the directive) ·
`[[seal_validates_only_inside_the_tree_that_fires_it_20260904]]` (the seal law the driver enforces) ·
`[[m78]]` reclaimable-aware admission · `[[m79]]` ceiling 116 GiB · `[[m123]]` two validators disagree ·
`[[m101]]` governed-only (count-half superseded) · `[[m53]]` negative-existence ·
CLAUDE.md "Results must become system intelligence", "Off is a tracked queue", the resumability and
containment laws.
