# pu1 — the assumed-zero placeholder gate: 1 defect in 9, gate landed, both controls executed

`verdict: CURED` · `verdict_scope: the ddm_qs1 dual-axis paid-dispatch transport`
· axis `[apparatus — no score, no dispatch, no spend]`

## The genus, named precisely

**A paid admission row whose every distortion axis is an assertion.**

Not "`*_unmeasured` flags are bad" — most of those are honest status labels and I
changed none of them. The defect needs four legs at once:

1. a numeric **placeholder** (`0.0`) on a distortion axis,
2. a flag beside it saying the value was never measured,
3. a **consumer that reads the value** while nothing reads the flag
   (`ddm_re1t_modal_t4_sign_gate.py:664-676` folds
   `pose_delta_s_placeholder = 0.0` into its mixed sum), and
4. the row is buying an **admission**, not a measurement — so the placeholder
   sits on the axis the candidate existed to move.

ps1u r2 had all four. It spent ~$0.16 and came back REFUSED at **+1.686e-02 S**
because the unscreened pose leg was **8.93× worse**, while the rate leg it paid
(+588 B) was known all along. Only the cost was real.

## The true population: 1

**Method (MEASURED, 2026-08-16).** I enumerated every retained
`SEALED_REQUEST.json` with `schema == ddm_qs1_t4_dual_axis_request.v1` on both
SSD tiers, paired each with its retained `POSE_SCREEN_RESULT.json`, and ran the
predicate over the real bytes. **9 sealed requests. 1 refused.**

| request | pose screen | seg leg | gate |
|---|---|---|---|
| `ddm_qs2_dual_axis_20260813_r1` | `local_pose_delta = 1.126177e-07`, `pose_unmeasured false` | — | ALLOW |
| `ddm_qs5_dual_axis_20260813_r1` | `local_pose_delta = -6.657906473377261e-09` (flag stale) | — | ALLOW |
| `ddm_qs1_dual_axis_20260813_r2` | payload `conservative_residual_pose_bound_s = 1.7746229678414843e-05` | — | ALLOW |
| `ddm_qs4_dual_axis_20260813_r1` | payload `pose_delta_s = 1.378369737898914e-05` | — | ALLOW |
| `ddm_pk3_dual_axis_20260813_r1` | payload `base_sample_dpose = 1.6211002068381195e-04` | — | ALLOW |
| `ddm_mc36_dual_axis_t4_r1` | payload `local_advisory.delta_dpose = -1.4632967835484165e-10` | — | ALLOW |
| `ddm_re1_dual_axis_pose_20260814_r1` | placeholder | **MEASURED** by `ddm_re1_round1_t4_gate_20260813r2`, `seg_delta_s = -1.6954210069444444e-06` | ALLOW |
| `ddm_qs1_dual_axis_20260813_r1` (superseded) | inputs never retained | — | unreachable — fails the pre-existing `require_record` check first |
| **`ddm_ps1u_dual_axis_pose_20260816_r2`** | **placeholder only** | **`re1t_run_id = "NONE_ps1u_seg_asserted_decode_identical"`, `seg_leg_measured false`** | **REFUSE** |

So: **1 of 9**. Six arms screened pose locally and simply parked the number in
the evidence payload instead of the request; one (re1) legitimately buys an
unknown pose leg backed by a real measured seg leg. Only ps1u shipped with
nothing measured at all.

MAIN's prior read is **CONFIRMED at source**: no dispatch gate anywhere reads
any `*_unmeasured` flag. The five sites that touch `pose_unmeasured`
(`ddm_re1t_modal_t4_sign_gate.py:564,:664`, `ddm_re1t_t4_sign_gate_worker.py:183`,
`ddm_qs2_compensation_rate_rung.py:812`, `ddm_js1c_cuda_custody_stage0.py:372`)
all **require the placeholder** — they enforce that it stays `0.0`. Zero of
them refuse a candidate for carrying it. The class was uncured.

## The finding that decided the design

ps1u's evidence payload **does** contain a finite non-zero pose float:
`pre_registered_admission.required_cuda_dpose_after = 6.251198917870592e-06`.
That is a **target**, not a measurement. A naive "scan for a pose-shaped number"
gate FALSE-ALLOWS the one request that matters — because *a target and a
measurement are indistinguishable by looking at the number*. That is the same
genus one level up, and it is why the gate recognises a screen only through an
explicit declaration or an explicit, corpus-measured key vocabulary.

## Landing 1 — the gate

`src/tac/deploy/dispatch_axis_screen.py` (new) —
`assert_distortion_axis_locally_screened(request, evidence_payload)`. The rule:
**a sealed dual-axis request may not dispatch unless at least one distortion
axis carries a local measurement.** Rate never satisfies it; rate is what the
row spends. Resolution ladder, first hit wins per axis:

1. `local_axis_screen` — the canonical forward declaration new arms must use.
   Claiming `measured: true` without a finite `delta_s` **and** a substantive
   `basis` raises, so the declaration cannot be a cheaper placeholder.
2. LEGACY request-level pose (the qs2 / qs5 shapes).
3. LEGACY `seg_leg_provenance` naming a real prior seg run (the re1 shape).
   `re1t_run_id` starting `NONE` is an assertion and does not count.
4. LEGACY evidence-payload pose under `CANONICAL_POSE_SCREEN_KEYS` — nine leaf
   paths, every one read out of a retained payload on 2026-08-16.

Nothing hits ⇒ refuse, fail-closed.

Every rung answers *"what proves a measurement happened?"*, never *"what fails
to deny it"* — an absent key is never a pass.

**Call site:** `experiments/ddm_qs1_modal_t4_dual_axis.py::load_sealed_inputs`.
I read the code rather than guessing: that one function is the choke point for
this transport. All nine sealers route `modal run ...ddm_qs1_modal_t4_dual_axis.py::main`,
and most also call `load_sealed_inputs` in-process to validate their seal. So a
refusal there fires **at seal time and at fire time**, before the meter starts.
Unparseable evidence parses to `None`, which reads as UNSCREENED — the safe
direction.

**Waiver:** `unscreened_axis_dispatch_waiver = {"rationale": "..."}`, ≥24 chars,
placeholder literals (`<reason>`, `<rationale>`, `TBD`, `pending`, `n/a`, …)
rejected. It is a JSON field, not a `# ..._OK:<rationale>` source comment,
because this gate reads a hash-sealed JSON request at runtime, not source text;
the substantive-rationale semantics are the repo's (Catalog #287), the carrier
is adapted to the surface.

## Landing 2 — the executed controls

`src/tac/tests/test_dispatch_axis_screen.py` (new), collected by default
(`testpaths` includes `src/tac/tests`). **35 passed, 0 skipped, rc=0** (33 before
the second-pass bypass fix below, 35 after).

Both directions, through the **real** dispatcher loader on **real files**:

- REFUSES a ps1u-shaped sealed request (`test_real_loader_refuses_the_ps1u_shape`).
- REFUSES **the actual retained ps1u bytes** —
  `/Volumes/APDataStore/pact/ddm_ps1u_uncapped_pose_20260816/dual_axis_pose/SEALED_REQUEST.json`,
  sha `0fdffa520374f7cc1b610944303a72977559e90e2b577566566754d6e8bf78a7`, with
  its real `fire_inputs/` — so the request that bought the +1.686e-02 S REFUSE
  cannot fire again (`test_real_loader_refuses_the_retained_ps1u_bytes`, RAN, not skipped).
- ALLOWS a genuinely screened candidate through the same loader
  (`test_real_loader_allows_a_screened_candidate`).
- Corrupt evidence fails closed (`test_unparseable_evidence_fails_closed`).

**Real-corpus sweep through the real loader, rc=0:** 7 ALLOW
(re1, qs4, qs2, pk3, qs5, mc36, qs1 r2), **1 REFUSE** (ps1u). **Zero false positives.**

**Mutation controls (would the tests pass if the code were broken?):**

| mutation | result |
|---|---|
| neuter the refusal (`if False:`) | **4 refusal tests FAIL** — `test_refuses_the_exact_ps1u_shape`, `test_real_loader_refuses_the_ps1u_shape`, `test_unparseable_evidence_fails_closed`, `test_real_loader_refuses_the_retained_ps1u_bytes` |
| blind the evidence recogniser | **7 allow tests FAIL** |

Both reverted byte-identically (sha `734b0bd2de2c433410af8bd39a3def1e8858e81365650b0af267a2f7b0f4b438`,
verified before and after). Sister suites unaffected: 43/43 pass, including
`test_ddm_mc36_dual_axis_seal.py`, which calls the real `load_sealed_inputs` and
is ALLOWED via ladder rung 4 — the gate is live in an existing test and does not
break it. `ruff check`: clean.

## Second-pass review found a bypass in my own gate

My fixes are unreviewed new code. The second pass caught one: the first draft
recognised a measured pose leg with `request.get("pose_unmeasured") is not True`
— which returns True when the key is **absent**. A ps1u successor could have
sailed through by *deleting the flag*. That is the ps1u defect wearing a hat,
re-introduced by the gate meant to extinct it.

Fixed to `is False` (positive evidence only), plus two regression tests
(`test_deleting_the_pose_unmeasured_flag_is_not_a_bypass`,
`test_deleting_both_pose_fields_is_not_a_bypass`). **35 passed** after the fix;
real-corpus verdicts unchanged at 7 ALLOW / 1 REFUSE.

## What I deliberately did NOT change, and why

- **The 366 `nonadditive_stack_interactions_unmeasured` / 48
  `op3_specialist_head_bytes_unmeasured` / 7 `required_unmeasured` sites.**
  Honest status labels. Rewriting them is churn that destroys real signal.
- **The worker placeholder law** (`ddm_re1t_t4_sign_gate_worker.py:183` and the
  four sister enforcement sites). It is correct in its role: it stops the remote
  worker from adjudicating pose off a local number. My gate adds a *different*
  requirement — that a local measurement exist *somewhere* — without touching a
  single worker byte, so no remote `worker_source_sha256` drifts.
- **The re1t seg-only transport** (`ddm_re1t_modal_t4_sign_gate.py`). Its rows
  buy a *measurement* (the seg field) and its adjudication already names
  `pose_follow_up_required_before_composition`. Applying an
  "already-measured-locally" rule there would refuse every honest re1t row.
  This is a judgment about scope, and it is where I am most likely wrong: if a
  future re1t-family row starts buying an *admission* rather than a
  measurement, the gate belongs there too.
- **The nine sealers.** None needed editing. The ladder recognises what they
  already write. New arms should use `local_axis_screen`; nothing forces a
  retrofit, so I did not create eight commits of churn.

## The honest limit

The gate cannot tell a real number from a fabricated one. A successor could
write `"pose_delta_s": 1e-9` and sail through. What it **can** do is make the
*omission* impossible to make silently: an arm must now produce a number or
write a rationale, and both are auditable acts. The ps1u failure was not
fabrication — it was an inherited schema whose placeholder nobody had to think
about. That specific silence is now closed. Deliberate fabrication remains
covered by NO-FAKE, not by this gate.

Second limit: the ladder's legacy rungs 2–4 recognise shapes I measured on
2026-08-16. A tenth arm inventing a tenth key spelling gets refused. That is
fail-closed and intended, but it will read as a false positive to whoever hits
it — the fix is one line in `CANONICAL_POSE_SCREEN_KEYS` or, better,
`local_axis_screen`.

## What I measured

9 sealed dual-axis requests on disk. 1 carries no measured distortion axis. The
gate refuses that one and the other 8 pass, through the real dispatcher loader,
on the real bytes. Neuter the gate and 4 tests fail; blind its recogniser and 7
fail. The pointer did not move and this arm never intended it to — this is
apparatus, and the next ps1u-shaped row costs $0 instead of $0.16.
