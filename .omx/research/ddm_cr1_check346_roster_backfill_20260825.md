# ddm_cr1 — Catalog #346 council-roster backfill: 20 → 0, honestly

**Arm:** cr1 (#346 council-roster backfill), ledger row #1278.
**Date:** 2026-08-25. **Cost:** $0, local only.
**Gate:** `check_council_dispatch_roster_complete_per_canonical_helper`
(`src/tac/preflight.py`), demoted warn-only at its `preflight_all` callsite by
`de942ec9b6`; **re-flipped `strict=True` at live count 0 in `25bbd07055`** —
one atomic commit carrying the detector cures, the controls, the 16 waivers and
the flip, per the strict-flip atomicity rule (no warn-only purgatory).

**Verified on the landed commit:** the 16 memo diffs are **384 additions / 0
deletions** with **zero removed lines**; `council_attendees` appears in the diff
only inside the appended note's own prose, never as a changed line.

## Headline

The 20 reported violations were **not 20 under-rostered councils**. They were
**4 detector defects + 16 genuine post-expansion under-rosterings**. Nothing was
repaired by inventing attendance: the 16 memo diffs are **384 additions / 0
deletions**, and the string `council_attendees` appears in **none** of them.

| Disposition | N | Mechanism |
|---|---:|---|
| SCOPE-FP — DAG-FEED companion | 1 | gate scope fix + both-direction controls |
| DETECTOR-FP — anachronistic seat demand | 1 | `seat_available_at` era filter + controls |
| DETECTOR-FP — name normalization (fully cured) | 2 | `canonical_seat_key` + controls |
| GENUINE under-rostering — acknowledged waiver | 16 | append-only note naming the absent seats |
| **Total** | **20** | live count now **0** |

A further **6** memos carried *partial* name-normalization false positives: they
remain violations, but the seats wrongly reported absent (`Ballé`, `PR95-author`,
`MacKay_memorial`) were struck from their waiver text, so no waiver names a seat
that was actually present.

## The roster timeline (reconstructed, with provenance)

Authority = the first commit introducing each seat into
`src/tac/canonical_council_roster.py`
(`git log --reverse -S'name="<seat>"' -- src/tac/canonical_council_roster.py`),
cross-checked against the CLAUDE.md "Grand Council (advisory)" / "Council
conduct" / "Experiment design" sections.

| Date | Commit | Seats added |
|---|---|---|
| 2026-05-19 | `06feeecf17` | module landing — 12 inner (incl. `PR95Author`, `MacKay`, `Balle`, `Quantizr`, `Hotz`, `Selfcomp`, `Assumption-Adversary`) + `TimeTraveler`, `TimeTravelerProtege` |
| 2026-05-19 | `77a2d0f38f` | `Rudin`, `Daubechies` (inner co-leads) + `Rudin_Grand`, `Daubechies_Grand` — roster-maintenance-v2, 4-co-lead structure |
| 2026-06-01 | `4b7db346a6` | `HaoChen_NeRV`, `Shrivastava_INR`, `Gwilliam_RNeRV` |
| 2026-06-01 | `153f228232` | `Kang_SNeRV`, `Bull_HiNeRV` |
| 2026-07-06 | `002257c665` | `FrankNielsen` |

**The decisive consequence: era exemption cannot rescue an inner-council
omission.** All 14 inner seats existed from 2026-05-19, and the gate's cutoff is
2026-05-19. Every memo in the violation set is dated 2026-05-20 or later, so all
16 genuine cases owed the full inner council at convocation time. The era filter
only ever relaxes a **grand** seat, and only for the single 2026-05-20 memo.

## Cure 1 — attendee-name normalization (`canonical_seat_key`)

The matcher compared attendee strings to seat names by **exact equality**. Memos
legitimately record the same canonical person under a spelling variant, so the
gate reported seated people as absent — a false under-rostering report.

Live variants found: `Ballé` vs seat id `Balle` (the person's own spelling);
`PR95-author` vs `PR95Author`; `MacKay_Memorial` / `MacKay_memorial` vs `MacKay`;
`AssumptionAdversary` vs `Assumption-Adversary`; `Schmidhuber-LEAD`;
`Tishby-memorial`; `Time-Traveler`.

Folded (identity-preserving decoration only): accents, case, parenthetical
qualifiers, separated ROLE annotations (LEAD / CO-LEAD / memorial / seat),
punctuation. **Deliberately not folded (identity-bearing):** `mentor` — because
`TimeTraveler` (mentor) and `TimeTravelerProtege` are distinct seats — and the
`_Grand` sister-seat suffix, so `Rudin` never satisfies `Rudin_Grand`.

Compound names are never split: `RaoBallard` satisfies neither `Rao` nor
`Ballard`. Splitting it would manufacture two per-seat attendances out of one
token — the `[[rr11f1-replayed-reauthor-collapses-discriminator]]` genus.

**Structural guard:** `_check_seat_key_injectivity()` runs at import and raises
if any two canonical seats fold to one key. A future rule that collapses a
distinction cannot be merged silently.

Fully cured 2 memos (`council_t2_composite_nscs06_v8_..._20260528`,
`council_t2_pr101_lc_v2_clone_..._20260528`) and corrected the reported seat set
on 6 more.

## Cure 2 — seat availability era (`seat_available_at`)

A memo cannot owe attendance from a seat appended after it was convened; the
gate was making an anachronistic demand. `validate_council_dispatch_roster` and
`required_attendees_for_topic` take a keyword-only `as_of_utc_yyyymmdd`
(default `None` = current roster, i.e. unchanged behavior); the gate passes the
memo's own filename date.

Live case: `grand_council_symposium_dwt_hnerv_world_model_bind_20260520.md`,
inner council **complete**, failing only on `len(missing_grand) >= 5`. Its 6
"missing" grand seats included `HaoChen_NeRV`, `Shrivastava_INR`,
`Gwilliam_RNeRV` — all added 2026-06-01, twelve days after the memo. Removing
the three that did not exist leaves 3 missing, under the T3 advisory threshold
of 5 ("Grand Council (advisory)": missing 1–4 may be acceptable) → complete.

Because the roster is APPEND-ONLY (Catalog #110/#113) and the default is `None`,
this filter can only ever **remove** a post-dated seat from a requirement.

## Cure 3 — DAG-FEED companion scope

`council_gc5_micro_macro_bridge_DAG_FEED_20260728.md` reported all 14 inner
seats missing. It is a triality **DAG FEED** — a synthesis-only trajectory
record *of* a deliberation, not the dispatch. Its own (2026-08-25-backfilled)
frontmatter says so and names where the seated record lives:

> "The per-seat roster is NOT enumerated in this synthesis-only DAG FEED; the
> seated record lives in the council posterior anchor
> `council_gc5_schmidhuber_micro_macro_bridge_20260728`."

Demanding a dispatch roster from a companion is a filename-glob scope defect.
`_CHECK_346_COMPANION_FILENAME_MARKERS = ("_DAG_FEED_",)` excludes it.
**Coverage is preserved:** it is the only `council_*_DAG_FEED_*.md` in the repo,
and its parent memo is itself in scope — and is one of the 16 genuine cases
waived below. Waiver-stamping the companion was rejected: the waiver text would
have asserted "the roster was incomplete" when the defect was in the detector.

## Controls (executed, both directions)

`src/tac/tests/test_preflight_check_346_scope.py` — 11 tests; every exclusion is
paired with a positive control on byte-identical content.

| Control | Result |
|---|---|
| under-rostered memo under a normal `council_` name | **fires** (positive control) |
| byte-identical content under a `_DAG_FEED_` name | excluded |
| companion + parent both present | parent still fires; companion does not |
| live companion cites a parent that exists and is in scope | verified |
| era-sensitive T3 memo dated 2026-05-20 | not charged for post-dated seats |
| byte-identical content dated 2026-06-15 | **fires** (positive control) |
| inner-council omission dated 2026-05-19 | **fires** — era never excuses inner seats |
| substantive waiver / `<rationale>` placeholder | accepted / **rejected** |
| `strict=True` | raises `PreflightError` |

`src/tac/tests/test_canonical_council_roster.py` — +26 tests, including the
negative controls that matter: `Ballard` does **not** satisfy `Balle`; `Rudin`
does **not** satisfy `Rudin_Grand`; `Nielsen` does **not** satisfy
`FrankNielsen`; normalization does **not** mask a genuinely absent seat; a
missing co-lead is still blocking; the era filter never shrinks the inner
council. **98 tests pass.**

## The 16 genuine under-rosterings

Each received an append-only dated note naming the absent seats, plus the
machine-readable `# COUNCIL_ROSTER_INCOMPLETE_OK:<rationale>` token. Seat lists
were generated from the live gate verdict, never hand-typed.

| Memo | Tier | Inner-council seats absent |
|---|---|---|
| `council_t2_cascade_c_prime_frame_1_segnet_waterfill_..._20260526` | T2 | Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author |
| `council_t2_compound_c_standalone_..._20260528` | T2 | Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author |
| `council_per_substrate_symposium_dqs1_..._20260529` | T2 | MacKay, Balle |
| `council_per_substrate_symposium_pr101_..._20260529` | T2 | Selfcomp, MacKay, Balle |
| `council_per_substrate_symposium_pr110_..._20260529` | T2 | Selfcomp, MacKay, Balle, PR95Author |
| `council_t3_grand_council_..._z8_hierarchical_predictive_coding_..._20260530` | T3 | Quantizr, Selfcomp, Balle |
| `council_grand_council_negative_findings_extreme_rigor_audit_20260531` | T3 | Quantizr, Selfcomp, Balle |
| `council_pose_carrier_optimal_form_symposium_20260703` | T3 | Quantizr, Hotz, Selfcomp, MacKay, Balle |
| `council_grand_symposium_ce_plateau_20260704` | T3 | Quantizr, Selfcomp |
| `council_grand_symposium_curriculum_derivation_20260705` | T3 | Quantizr, Selfcomp |
| `council_grand_symposium_levelset_loss_geometry_20260705` | T3 | Quantizr, Selfcomp |
| `council_symposium_clean_config_20260705` | T3 | Quantizr, Hotz, Selfcomp, MacKay |
| `council_v6_eikonal_cure_symposium_20260705` | T3 | Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author |
| `council_t3_symposium_islands_treatment_arm_20260706` | T3 | Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author |
| **`council_v752_relaunch_shape_concurrent_vs_single_20260710`** | T2 | **Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author, Rudin, Daubechies** |
| `council_gc5_schmidhuber_micro_macro_bridge_20260728` | T3 | MacKay, Balle |

**`v752` is the one CO-LEAD omission.** `Rudin` and `Daubechies` are half the
4-co-lead shared-leadership core, BLOCKING at T2+ per the CLAUDE.md "Council
conduct" 2026-05-19 amendment. Its note says so explicitly and flags the
deliberation as reduced in binding weight when cited. The other 15 omitted only
sister voices; every one of them seated all 4 co-leads.

## The finding under the finding

**`Quantizr`, `Selfcomp`, `MacKay`, `Balle`, `Hotz` are the recurring absentees**
— present in 14, 13, 12, 12 and 7 of the 16 respectively. This is not scattered
sloppiness; it is a systematic convocation habit. The rostered core drifted to
"the sextet + Rudin + Daubechies + whichever topical specialists the subject
suggests", and the competitor-facing / codec-practitioner voices
(Quantizr adversarial, Selfcomp, Ballé neural-compression, MacKay MDL) fell out
by default. CLAUDE.md "Experiment design" is explicit that this is not allowed:
*"All ten voices are permanently active. No member may be silenced or deferred
in any deliberation."*

The re-flipped gate now refuses the habit structurally rather than relying on a
convener remembering. That is the point of the ratchet.

## Honest residue

1. **The 16 waivers do not un-do the under-rostering.** Each names the absent
   seats so a future citation can weigh the deliberation accordingly. Any of the
   16 may be re-convened with the full roster; none is retracted.
2. **The `_DAG_FEED_` exclusion is a filename rule**, so a real deliberation
   memo named `..._DAG_FEED_...` would escape. Live count of that case: 0. The
   parent-memo obligation is the backstop, and the test asserts the live
   companion has an in-scope parent.
3. **`RaoBallard` and `Time-Traveler_Mentor` remain unmatched tokens.** Both
   appear only where they cannot change a verdict. Splitting or folding them
   would collapse a distinction; left as-is deliberately.
4. **Two pre-existing test failures are untouched and unrelated:**
   `test_check_363_council_recursive_self_reflection::test_live_repo_regression_guard`
   (14 > warn-only ceiling 5) and
   `test_ema_wireins_council_d::test_check_88_passes_strict_on_live_codebase`.
   The #363 failure was verified against the **HEAD (pre-edit)** copies of all
   16 memos: the same **7** memos violate it before and after this landing, so
   these appends caused **zero** new #363 violations. Catalog #363 debt is real
   and belongs to a separate arm.
5. **No score claim.** This is apparatus. The exact pointer is unmoved by this
   landing and nothing here is goal progress.

## Cross-references

- CLAUDE.md "Experiment design — non-negotiable" · "Council conduct" ·
  "Grand Council (advisory)" · "Council hierarchy: 4-tier protocol"
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (no body mutation)
- Catalog #292 (per-deliberation assumption surfacing) · #300 (v2 frontmatter) ·
  #325 (per-substrate symposium) · #363 (recursive self-reflection — separate debt)
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
  (strict-flip atomicity: the re-flip rides the final cure)
