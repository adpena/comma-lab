# ddm_qn1 — the QBR1 CONDITIONAL-N600-BUY is now one command

**Arm:** `ddm_qn1_qbr1_n600_realization_ticket` · **Date:** 2026-09-03 · **Commit:** `a42e0fa5f`
**Craft:** `docs/operating_manual_craft_handoff.md` (§3 risk-ranking, §4 re-derive, §5 label, §6 attack-your-own)
**Axis:** `[macOS-CPU scorer-free plumbing proof]` · `score_claim=false` · pointer **UNMOVED**

## Answer first

The ticket generator is built, tested, and proven against the live burn's cell 1. When
`ADJUDICATION_RESULT.json` lands, MAIN fires **one command** and gets a sealed, byte-bound n600
realization fire order. No scorer, Modal, or Metal ran here.

Two things the work found that change how the ticket must be read:

1. **BR2 and QXR1 cannot realize the burn's field.** Both hardcode a single archive SHA and pin their
   output root (`br2.storage_preflight` refuses any root but `/Volumes/APDataStore/pact/ddm_br2`). So
   the ticket could not simply emit BR2 argv — it would have realized the OLD born object. QN1 carries
   its own `realize` subcommand: the same BR2 protocol, with the object taken from the ticket, and the
   scorer-touching core delegated to BR2's own measured `realize_chunk`.
2. **The pre-registered falsifier does not clear 0.12, and the ticket now says so out loud.** At
   `d_seg = 0.01` the seg term alone is 1.0. DERIVED on the bound archive: S at the falsifier corner =
   **1.106353215794932**. The d_seg that 0.12 actually needs at the falsifier pose is
   **1.3646784205e-4** — the falsifier is **73.28× looser** than the target. A field named
   `falsifier_clears_0_12` was in my first draft and would have been read as "meeting it = sub-0.12";
   my own test refuted it. It is now `falsifier_alone_clears_0_12: false` plus
   `falsifier_is_a_regime_marker_not_the_target: true` plus the two derived numbers.

## Ticket schema — `ddm_qn1_n600_realization_fire_order.v1`

| Block | Contents |
|---|---|
| `adjudication` | schema, disposition, `treatment_wins`, `treatment_pose_corner_passes`, `seed_rows`, source `file_fact`, the burn's own `preregistration` (re-derived), `synthetic` flag |
| `winner` | seed, arm, `cell_id`, n32 `S_hat`/`d_seg_hat`/`d_pose_hat`/`rate_exact`, `pose_corner_pass`, endpoint step, `selection_rule`, `result_fact`, `n32_numbers_are_selection_statistics_only: true` |
| `object` | container `file_fact`; `archive` + `packet` (member, bytes, sha256); `decoded_field_digest`; per-section digests; `latent_records`; receiver packet + archive round-trip bit-identity; `scored_ancestor_sha256`; `rate_exact_recomputed` |
| `realization` | protocol + `protocol_source` fact, runner fact, `chunk_pairs` 30, `chunks` 20, full `chunk_plan` (first/last pair + payload name per chunk), output/resume roots, retention clause, `minimum_free_bytes` + its derivation, `expected_wall_seconds` + both measured sources, **verbatim `argv`**, one-line `command`, `scorer_claim_id` |
| `score_law` | `100*d_seg + sqrt(10*d_pose) + 25*B/37545489` evaluated at zero distortion on the bound archive, plus `sub_0_12_distortion_budget`, `byte_feasible`, `delta_vs_0_12`, `delta_vs_afr1` |
| `prediction` | the QXR1 falsifier re-derived on this archive (see the correction above) |
| guards | `no_distortion_transfer`, `n32_advisory_numbers_are_not_transferred_to_n600`, four `*_invocations_by_this_generator: 0`, five `boundaries` |

**Winner selection (stated, because it is stronger than the family rule).** The family outcome needs
≥2/3 seeds. A single n600 buy is ONE cell, so that cell must itself both win its seed AND pass its own
pose corner; ties break on lowest `treatment_S_hat`, then lowest seed. If the family is LIVE but no
single cell satisfies both legs, the ticket REFUSES ("no single cell is buyable").

**Refusals (all fail-closed `QN1Error`).** missing result · non-LIVE disposition (MIXED / CLOSED /
anything) · adjudication schema drift · empty seed rows · no buyable cell (pose corner) · claim
placeholder · foreign lane prefix · incomplete cell result · endpoint step ≠ 5000 · output root inside
the live burn custody · output root outside QN1 custody · realization root inside burn custody ·
pre-registration drift · RESULT.json drift since adjudication · milestone/adjudication pose-corner
disagreement · archive or packet SHA drift · `rate_exact` not the score law · receiver not
bit-identical · section set ≠ QBF1 · latents ≠ 600 · archive byte-identical to a scored ancestor ·
container member absent or non-regular · `realize` without `--launch-authorized` · `--resume-from` ≠
`--output` · dry-run ticket · ticket schema drift · claim id ≠ sealed · `--output` ≠ sealed root.

## Dry-run receipt — plumbing only, on a CONTROL cell

`/Volumes/VertigoDataTier/pact/ddm_qn1_qbr1_n600_realization_ticket/`

| File | bytes | sha256 |
|---|---:|---|
| `DRY_RUN_RECEIPT.json` | 8,247 | `8388f7392fa87f3179556dcb4b89d81d9ac0122d369b4aee89db65302e67e364` |
| `DRY_RUN_FIRE_ORDER.json` | 12,927 | `5ac87a4cd6551808425bd0637efffe74f7c32a54808ae8ba09adff5232ee1e2f` |

**Source (READ-ONLY):** `runs/seed_20260902/control_native100/milestones/step_002000/MILESTONE.json`
— the latest materialized milestone of the LIVE burn's cell 1.

> ⚠ **This is a CONTROL arm at step 2000, not a treatment at the 5000 endpoint, and the adjudication
> flags used for the binding half are SYNTHETIC.** The cell's real `pose_corner_pass` is **False** and
> is recorded verbatim in the receipt next to the synthetic flag. Nothing here is a treatment verdict,
> a family verdict, or a score. The ticket it wrote is stamped `mode: DRY_RUN_PLUMBING`,
> `disposition: DRY_RUN_NOT_FIREABLE`, and `realize` refuses it.

**Bindings (MEASURED, byte-exact against the milestone record):**

| Item | Value |
|---|---|
| container | `.../step_002000/reencoded/reencode_payloads.tar`, 2,723,840 B, sha `f3d6f6fbfd79574981720b32069561be1bd102794807462f78f31ac5665c8892` |
| archive (`archive.zip`) | **106,626 B**, sha `59fc06204d3080bb38a5da9a8cb4897e52430f01dbfb00c021e844ef3412276e` |
| packet (`packet.qbf`) | **106,518 B**, sha `b9b6a9a6f1a8d56a7e6ed3a5ca9f88db58442b22a5bee5f944ce9ec65dfb840f` |
| decoded field digest | `626337db0ce07e36e4af0cb90e494ee2fe4e43adb120bbc87d484fb75b893e83` |
| receiver | packet bit-identical ✓ · archive re-encode bit-identical ✓ · sections {1,2,3,4} ✓ · **600/600 latent records** ✓ |
| rate | `25·106626/37545489 = 0.07099787673560465` — equals the milestone's own `rate_exact` to 1e-15 |
| byte-feasible | **YES** (106,626 ≤ 137,986) — this object is already inside the rate corner |
| chunk plan | 20 chunks × 30 pairs, `[0,29] … [570,599]`, payloads `scorer_pairs_0000_0029.npz … scorer_pairs_0570_0599.npz` |
| ancestor guard | not byte-identical to BR2 / QXR1 / AFR1 archives |

**Refusals: 14 of 14 fired, each for its own reason** (`all_refusals_fired: true`). The
PRIOR-LAW PREDICTION in the charter held: **0 falsifier hits.**

**Also verified against the primary artifact:** `br2.model_from_packet` builds a real `QBFLOWTorch`
from this burn packet. The fire order is genuinely fireable, not just well-typed.

## The exact MAIN command

**Step 1 — when `ADJUDICATION_RESULT.json` lands (one line):**

```
.venv/bin/python experiments/ddm_qn1_qbr1_n600_realization_ticket.py ticket --scorer-claim-id ddm_qn1_scorer_20260904
```

Defaults: `--adjudication /Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/ADJUDICATION_RESULT.json`,
`--output /Volumes/VertigoDataTier/pact/ddm_qn1_qbr1_n600_realization_ticket`,
`--realization-output /Volumes/APDataStore/pact/ddm_qn1_qbr1_n600_realization`. It writes
`FIRE_ORDER.json` or refuses. The default `--scorer-claim-id` is the placeholder `QN1_SCORER_CLAIM_ID`,
which refuses — MAIN must bind a real `ddm_qn1_*` lane id.

**Step 2 — the realization the ticket then carries verbatim in `realization.command`:**

```
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /Users/adpena/Projects/pact/.venv/bin/python /Users/adpena/Projects/pact/experiments/ddm_qn1_qbr1_n600_realization_ticket.py realize --ticket /Volumes/VertigoDataTier/pact/ddm_qn1_qbr1_n600_realization_ticket/FIRE_ORDER.json --output /Volumes/APDataStore/pact/ddm_qn1_qbr1_n600_realization --resume-from /Volumes/APDataStore/pact/ddm_qn1_qbr1_n600_realization --scorer-claim-id ddm_qn1_scorer_20260904 --launch-authorized
```

Preconditions the runner re-checks at fire time: a fresh unique **active `local_macos_cpu`** claim for
that id with no other active scorer claim in 24 h · **AP free ≥ 1,500,000,000 B** · container, archive,
packet, and decoded-field SHAs re-matched · `--output` equal to the sealed root. Expected wall
**≈ 485 s** (BR2's own `REALIZED_RESULT.json` elapsed **479.663 s**; the burn's `REDERIVED_TIMING.json`
`br2_realization_seconds_each` **484.769 s**), so it is under the 30-min detached-launch threshold; use
`tools/launch_detached_process.py` anyway if MAIN wants the receipt.

**Storage precondition, DERIVED not assumed:** BR2 retained 20 chunks summing **1,058,094,084 B**;
1.25 × that + 100 MB = 1,422,617,605 B → rounded to **1.5 GB**, which matches QXR1's `MIN_FREE_SCORE`.

## RECALL EVIDENCE

- `experiments/ddm_br2_born_object_scorer_realization.py` (read in full) — protocol, chunk core,
  `aggregate` arithmetic, claim contract, and its frozen `ARCHIVE_SHA256`/`OUTPUT_ROOT` pins.
- `/Volumes/APDataStore/pact/ddm_br2/REALIZED_RESULT.json` — MEASURED d_seg **0.17077688429090712**,
  d_pose **115.83742417077796**, S **51.18**, DISTORTION-REFUSED, 20 chunks, 479.663 s. **Not
  transferred**; it belongs to the OLD born object.
- `/Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding/FIRE_ORDER.json` — the fire-order
  shape, the 1.5 GB precondition, the 485 s figure, the falsifier text, "identical-by-construction ⇒
  derive, don't fire".
- `experiments/ddm_qbr1_born_fairform_burn_prep.py` — `adjudicate()` emits
  `ddm_qbr1_adjudication_result.v1` with exactly `{disposition, treatment_wins,
  treatment_pose_corner_passes, seed_rows, source_results}`; `SEEDS`, `ARMS`, `MILESTONES`,
  `RESULT_SCHEMA`, `TOTAL_STEPS` read at source.
- `ADJUDICATION_SCHEMA.json` + `SEALED_MAIN_FIRE_ORDER.json` (burn custody) — the pre-registered rule
  the ticket re-derives rather than restates.
- Memory `m110` (pose absolute budget ≤ 1.25e-4), the CLAUDE.md goal banner (137,986 B ceiling,
  afr1 rate 0.11985594327989708 — reproduced by a test), `ddm_bz2d` (never inherit distortion).

## LIVE-HYPOTHESES

1. **The burn's object is already byte-feasible; only distortion is open.** 106,626 B ⇒ rate 0.0710,
   leaving a **0.0490** distortion budget. MEASURED at n32 on cell 1 step 2000: `100·d_seg = 0.322`
   alone spends 6.6× that budget. The n600 buy is a distortion measurement, not a rate one.
2. **The falsifier is 73.28× looser than the target.** A treatment cell can meet the QXR1 falsifier and
   still be ~9× above 0.12. Any ticket read must use `d_seg_required_for_0_12_at_the_falsifier_pose`
   (1.3646784205e-4), not the falsifier.
3. **n32 → n600 is untested on this vehicle.** The adjudication's HT `S_hat` is a *selection*
   statistic. Whether the n600 row lands near it is exactly what the buy measures.
4. **Prefix/selection bias is not this arm's to assert.** The burn's n32 draw is stratified HT, not a
   prefix, so `m88`/`m96` do not obviously apply — UNVERIFIED, flagged for whoever reads the row.

## DEAD-ENDS (do not repeat)

- **Emitting BR2 `realize` argv directly.** BR2's frozen `ARCHIVE_SHA256` and `OUTPUT_ROOT` pin mean
  that command realizes the OLD born object no matter which ticket you hand it. Verified at source.
- **Adding flags to BR2.** Its `realize` subparser has exactly `--output / --resume-from /
  --scorer-claim-id / --launch-authorized` (asserted by a test that reads BR2's source). QN1 adds
  exactly one flag, `--ticket`, on its own parser.
- **Reusing `br2.assert_active_scorer_claim`.** It hardcodes the `ddm_br2_` prefix and cannot admit a
  QN1 lane; QN1 carries the same contract with the prefix parameterised (documented as such).
- **Reusing `br2.aggregate` unchanged.** Its rate term reads a module constant (BR2's 106,832 B), which
  is the wrong archive. QN1's `aggregate` is byte-equal to BR2's on d_seg/d_pose/seg/pose/rate/S when
  the archive matches — asserted by a test.
- **Probing the ancestor guard by editing the milestone's recorded SHA.** The SHA-drift check fires
  first, so the probe passed for the wrong reason and the guard stayed unproven. The guard now compares
  the RECOMPUTED digest and the probe declares a stand-in ancestor.

## Boundaries and what is owed

- `realize` has **never been executed** — no scorer was permitted. Its non-scorer scaffolding (flags,
  ticket checks, custody guards, chunk plan, aggregation) is unit-tested; its scorer-touching core is
  BR2's own already-exercised `realize_chunk`. **First MAIN fire is also its first execution**; expect
  to babysit chunk 1, which is restartable by design.
- The dry run bound a **step-2000 CONTROL** milestone. The live path additionally requires
  `complete=true`, step 5000, the treatment arm, and milestone/adjudication pose-corner agreement —
  none of which the dry run could exercise positively.
- `storage_preflight` uses the module constant, not the ticket's field, deliberately: a hand-edited
  ticket must not be able to LOWER the precondition.
- `tools/review_tracker.py` does not track `tests/`; the test file got two by-hand adversarial passes
  (they caught two real code defects), the module got two tracked passes plus four adversarial rounds.
  I do not claim a 3-clean-pass SEAL — round 4 was the first clean one.

## NEXT_IF_RESUMED

1. **Wait for `ADJUDICATION_RESULT.json`** (~+13 h from 14:47 UTC-7; cell 1 was at step 2000 of 5000).
   Then run Step 1 above with a fresh `ddm_qn1_scorer_<date>` claim.
2. If it refuses with **`no single cell is buyable`** or a non-LIVE disposition, that is the honest
   answer — do not soften the gate. Report the disposition and stop.
3. If it writes `FIRE_ORDER.json`, append the active claim row, confirm AP free ≥ 1.5 GB, and fire
   Step 2. ~485 s, fully restartable, every payload retained.
4. Read the row against **`d_seg ≤ 1.3646784205e-4`**, not the falsifier. Then decide: if the realized
   d_seg is order 1e-3 or worse, the OPTIMIZATION route is measured-closed on this object at n600 and
   the next unit belongs to the RATE corner (`ddm_x012` door map), not to more burn cells.
5. Owed either way: fold the realized row into `ddm_x012` and state plainly whether the pointer moved.

---

**Own-vehicle frontier: S 0.14797617125559104 @ 180,002 B `[contest-CUDA T4 n600]` (afr1) — UNMOVED
this unit.** This arm produced a fire order, not a score; that is a MEANS, and the END is still a lower
exact row.

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `gap_decomposition_against_demonstrated_floor_v1` — `tac.canonical_equations.gap_decomposition_against_floor_20260802` (`tac.canonical_equations`). **Relation:** REFINES the law's m66 clause (a ΔS without its gap denominator is unanchored).

QN1's own correction is exactly that clause firing on a FALSIFIER: at d_seg = 0.01 the seg term alone is 1.0, S at the falsifier corner = 1.106353215794932, and the d_seg 0.12 actually needs at the falsifier pose is 1.3646784205e-4 — the falsifier is 73.28× looser than the target. Scope honesty: these are DERIVED on a bound archive, not the two matched MEASURED triples the law's `included` clause requires, so this is a refinement of the reading rule, not a new in-domain anchor.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
