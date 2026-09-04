#!/usr/bin/env python3
"""A1 -- measured agreement study: Catalog #344's regex vs the fmtools advisory lane.

The question
------------
Catalog #344 fires on a memo that "states a measured empirical finding" and then
demands an equations-leg citation. ddm_eq1 (2026-09-04) adjudicated the 29 memos
the gate reported live at commit ``d3212bed1`` into two classes:

* **MEASURED** (20) -- the memo states a measured empirical finding of its own
  about the campaign's object of study, and names a law for it.
* **WAIVER** (9) -- the memo is a review / verdict-on-process / hygiene /
  consumption-sweep / apparatus-debt memo; it carries a
  ``# FORMALIZATION_PENDING`` waiver naming the law it would need.

That human adjudication is this study's ground truth. Three lanes are scored
against it:

1. ``regex_before`` -- plain substring matching of the finding tokens, the gate
   as it stood before eq1's fix.
2. ``regex_after`` -- the shipped gate, with ``(?<!st)ratified`` so the token
   stops matching inside "stratified".
3. ``fmtools`` -- the on-device Foundation Model, advisory, via
   ``tools.fmtools_advisory``.

Method notes that bound the result
----------------------------------
* Memo text is read at ``d3212bed1`` -- BEFORE eq1 appended its equations-leg
  addenda. Reading the memos as they stand today would leak the answer key into
  both lanes (the addendum literally prints "MEASURED" or the waiver).
* The regex lanes are scored TWICE and both numbers are reported. ``_FULL`` is
  production-faithful -- the gate reads the whole memo, so that row describes the
  live gate. ``_EXCERPT`` is the MATCHED-INPUT control: the same first
  ``EXCERPT_CHARS`` characters the advisory lane gets, because the on-device
  context window is finite and the repo's writing discipline puts the finding in
  the head. Reporting only one framing would credit or penalise a lane for how
  much text it saw rather than for how well it judged.
* The regex lanes were designed as HIGH-RECALL triggers, not as classifiers.
  Low precision is partly by design -- the gate would rather over-ask than miss
  an equations leg. Read the precision column as "how much of the gate's noise a
  second lane could remove", not as "the gate is broken".
* The FM is non-deterministic. ``--repeats`` runs the lane N times so the report
  can state stability instead of assuming it.

Payload retention: every per-memo row is persisted, not only the summary
counts, so a later arm can re-score without re-running the model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.fmtools_advisory import classify_texts  # noqa: E402

#: The commit at which Catalog #344 reported these 29 memos live (ddm_eq1).
BASELINE_COMMIT = "d3212bed1"
#: Bounded excerpt handed to every lane.
EXCERPT_CHARS = 4000

POSITIVE = "states_measured_finding"
NEGATIVE = "review_or_process"

#: ddm_eq1's human adjudication -- this study's ground truth.
#: `.omx/research/ddm_eq1_equations_leg_backfill_20260904.md` section 2.
GROUND_TRUTH: dict[str, str] = {
    # 20 MEASURED
    "ddm_ft1_shipped_renderer_aligned_finetune_20260903": POSITIVE,
    "ddm_ar1_aa_render_price_on_born_field_20260903": POSITIVE,
    "ddm_fcd1_field_for_coder_diagonal_20260829": POSITIVE,
    "ddm_gf2_static_dynamic_generator_form_20260903": POSITIVE,
    "ddm_lc3_lane_carriage_rung_20260831": POSITIVE,
    "ddm_jc1_afr_rc64_joint_redesign_20260901": POSITIVE,
    "ddm_na11_negative_regrade_20260829": POSITIVE,
    "ddm_nx1_next_object_route_20260831": POSITIVE,
    "ddm_qbr1_born_fairform_burn_prep_20260902": POSITIVE,
    "ddm_qn1_qbr1_n600_realization_ticket_20260903": POSITIVE,
    "SPEC_ddm_qbflow_packet_schema_v1_20260827": POSITIVE,
    "ddm_qbt1_r1_r2_qbflow_verdict_20260827": POSITIVE,
    "ddm_qbt2b_r3_ce_birth_verdict_20260827": POSITIVE,
    "ddm_qbt2b_r4_extended_ce_verdict_20260828": POSITIVE,
    "ddm_qbt2b_r5_balanced_ce_verdict_20260828": POSITIVE,
    "ddm_qbt2b_r6_born_field_margin_verdict_20260828": POSITIVE,
    "ddm_qbt2b_r7_constrained_margin_verdict_20260828": POSITIVE,
    "ddm_qbt2b_r8_constrained_margin_verdict_20260828": POSITIVE,
    "ddm_qbt2b_r9_constrained_margin_verdict_20260829": POSITIVE,
    "ddm_qbt2b_r10_third_doubling_stop_verdict_20260829": POSITIVE,
    # 9 WAIVER (review / process)
    "ddm_fr2_final_fresh_eyes_pr_review_20260903": NEGATIVE,
    "ddm_pq13_pr_body_refresh_verdict_20260901": NEGATIVE,
    "ddm_ht1_red_debt_hygiene_verdict_20260901": NEGATIVE,
    "ddm_hv3_done_arm_consumption_20260827": NEGATIVE,
    "ddm_hv4_recovery_consumption_sweep_20260829": NEGATIVE,
    "ddm_rc_precheck_folded_never_fired_20260831": NEGATIVE,
    "ddm_hp1_premise_lint_canonicalization_20260831": NEGATIVE,
    "ddm_fpr1_falsified_premise_registrations_20260831": NEGATIVE,
    "ddm_ql2_apparatus_debt_20260903": NEGATIVE,
}

#: The advisory instruction. Written once from the CONCEPT eq1 adjudicated on
#: (object-of-study result vs apparatus/process report), not tuned per memo.
INSTRUCTION = (
    "You are reading the opening of a research memo from a machine-learning "
    "video-compression campaign. Decide what kind of memo it is.\n\n"
    "Return 'states_measured_finding' when the memo's own substantive content is "
    "an empirical result THIS memo's author measured or derived about the "
    "campaign's OBJECT OF STUDY -- the codec, the model, the score, the encoded "
    "bytes, a training run, an experiment, a probe. Verdict memos count when the "
    "verdict rests on numbers the author measured from a run.\n\n"
    "Return 'review_or_process' when the memo is instead about the campaign's "
    "APPARATUS OR PROCESS: it reviews someone else's pull request or memo, "
    "records a submission or registration, audits documentation, sweeps for "
    "stale or owed items, reports hygiene or technical debt, or tracks whether "
    "steps fired. These memos often quote counts of the things they swept -- "
    "counting items is not an empirical result about the object of study.\n\n"
    "The test is what the memo is FOR, not whether numbers appear in it."
)


@dataclass(frozen=True, slots=True)
class MemoRow:
    """One memo's excerpt plus each lane's verdict."""

    name: str
    truth: str
    excerpt_chars: int
    full_chars: int
    regex_before_full: str
    regex_after_full: str
    regex_before: str
    regex_after: str
    fm_labels: list[str]


def read_memo_at_baseline(name: str, commit: str = BASELINE_COMMIT) -> str | None:
    """Return the memo's text at ``commit``, or None when it is not there."""
    proc = subprocess.run(
        ["git", "show", f"{commit}:.omx/research/{name}.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def score(rows: list[tuple[str, str]]) -> dict[str, float | int]:
    """Precision / recall / F1 / accuracy for the POSITIVE class.

    Args:
        rows: ``(predicted, truth)`` pairs.

    Returns:
        Counts plus the three rates. Rates are 0.0 when their denominator is 0,
        and the denominators are reported so a vacuous rate cannot read as a
        real one.
    """
    tp = sum(1 for p, t in rows if p == POSITIVE and t == POSITIVE)
    fp = sum(1 for p, t in rows if p == POSITIVE and t != POSITIVE)
    fn = sum(1 for p, t in rows if p != POSITIVE and t == POSITIVE)
    tn = sum(1 for p, t in rows if p != POSITIVE and t != POSITIVE)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", default=BASELINE_COMMIT, help="baseline commit")
    parser.add_argument(
        "--repeats", type=int, default=3, help="FM runs, for stability (default 3)"
    )
    parser.add_argument(
        "--excerpt-chars", type=int, default=EXCERPT_CHARS, help="chars per memo"
    )
    parser.add_argument(
        "--reuse-fm",
        default=None,
        help=(
            "re-score from a prior run's persisted per-memo rows instead of "
            "re-running the model (the FM is non-deterministic, so a re-run "
            "would not reproduce the companion numbers)"
        ),
    )
    parser.add_argument(
        "--out",
        default=".omx/research/ddm_fm3_catalog344_agreement_20260904.json",
        help="where to persist every per-memo row (payload retention)",
    )
    args = parser.parse_args(argv)

    from tac.preflight import (  # noqa: PLC0415 - keep import cost off --help
        _CHECK_344_EMPIRICAL_FINDING_TOKENS,
        _check_344_text_has_empirical_finding,
    )

    full_texts: dict[str, str] = {}
    excerpts: dict[str, str] = {}
    missing: list[str] = []
    for name in GROUND_TRUTH:
        text = read_memo_at_baseline(name, args.commit)
        if text is None:
            missing.append(name)
            continue
        full_texts[name] = text
        excerpts[name] = text[: args.excerpt_chars]
    if missing:
        print(f"WARNING: {len(missing)} memo(s) absent at {args.commit}: {missing}")

    # --- deterministic lanes ---
    # TWO framings, because they answer different questions and reporting only
    # one would be a units-of-the-claim error:
    #   * FULL text  = what the gate actually reads in production. This is the
    #     production-faithful number and the one that describes the live gate.
    #   * EXCERPT    = the same bounded input the advisory lane gets. This is
    #     the MATCHED-INPUT control: it isolates classifier quality from input
    #     budget, so neither lane is credited or penalised for how much it saw.
    def _regex_pair(text: str) -> tuple[str, str]:
        low = text.lower()
        before = any(tok in low for tok in _CHECK_344_EMPIRICAL_FINDING_TOKENS)
        after = _check_344_text_has_empirical_finding(text)
        return (POSITIVE if before else NEGATIVE, POSITIVE if after else NEGATIVE)

    regex_before_full: dict[str, str] = {}
    regex_after_full: dict[str, str] = {}
    regex_before: dict[str, str] = {}
    regex_after: dict[str, str] = {}
    for name in excerpts:
        regex_before_full[name], regex_after_full[name] = _regex_pair(full_texts[name])
        regex_before[name], regex_after[name] = _regex_pair(excerpts[name])

    # --- advisory lane, repeated for stability ---
    fm_runs: list[dict[str, str]] = []
    fm_reasons: list[str] = []
    if args.reuse_fm:
        # Payload retention paying off: re-score from persisted per-memo rows
        # instead of re-running a non-deterministic model. Re-running would give
        # DIFFERENT labels and silently change the regex numbers' companion.
        prior = json.loads((REPO_ROOT / args.reuse_fm).read_text(encoding="utf-8"))
        n_runs = max((len(r["fm_labels"]) for r in prior["rows"]), default=0)
        for index in range(n_runs):
            fm_runs.append(
                {
                    r["memo"]: r["fm_labels"][index]
                    for r in prior["rows"]
                    if index < len(r["fm_labels"]) and r["fm_labels"][index] != "no_advice"
                }
            )
        fm_reasons = list(prior.get("fm_not_run_reasons", []))
        print(f"  reusing {n_runs} persisted FM run(s) from {args.reuse_fm}")
    for run_index in range(0 if args.reuse_fm else max(1, args.repeats)):
        verdict = classify_texts(
            excerpts,
            labels=[POSITIVE, NEGATIVE],
            instruction=INSTRUCTION,
            max_chars=args.excerpt_chars,
        )
        if not verdict.ran:
            fm_reasons.append(verdict.reason or "unknown")
            print(f"  FM run {run_index + 1}: DID NOT RUN ({verdict.reason})")
            break
        fm_runs.append(verdict.labels)
        agree = sum(1 for k, v in verdict.labels.items() if v == GROUND_TRUTH.get(k))
        print(
            f"  FM run {run_index + 1}: labelled {len(verdict.labels)}/{len(excerpts)}"
            f", {agree} match truth"
        )

    rows: list[MemoRow] = [
        MemoRow(
            name=name,
            truth=GROUND_TRUTH[name],
            excerpt_chars=len(excerpts[name]),
            full_chars=len(full_texts[name]),
            regex_before_full=regex_before_full[name],
            regex_after_full=regex_after_full[name],
            regex_before=regex_before[name],
            regex_after=regex_after[name],
            fm_labels=[run.get(name, "no_advice") for run in fm_runs],
        )
        for name in excerpts
    ]

    lanes: dict[str, dict[str, float | int]] = {
        # production-faithful: the gate reads the whole memo
        "regex_before_fix_FULL": score([(r.regex_before_full, r.truth) for r in rows]),
        "regex_after_fix_FULL": score([(r.regex_after_full, r.truth) for r in rows]),
        # matched-input control: same bounded excerpt the advisory lane sees
        "regex_before_fix_EXCERPT": score([(r.regex_before, r.truth) for r in rows]),
        "regex_after_fix_EXCERPT": score([(r.regex_after, r.truth) for r in rows]),
    }
    # Majority vote across repeats; ties and no-advice fall to the NEGATIVE side
    # so the advisory lane can never invent a positive it did not actually reach.
    if fm_runs:
        majority: dict[str, str] = {}
        for row in rows:
            votes = [label for label in row.fm_labels if label in (POSITIVE, NEGATIVE)]
            majority[row.name] = (
                POSITIVE if votes.count(POSITIVE) * 2 > len(votes) else NEGATIVE
            )
        lanes["fmtools_majority"] = score(
            [(majority[r.name], r.truth) for r in rows]
        )
        # COMPOSITION lanes -- the actual deployment question. The advisory lane
        # is not a replacement for the gate; it is a second opinion beside it, so
        # what matters is what the PAIR does.
        #   UNION  (regex OR fm): the recall lane -- catches memos the post-fix
        #          token set no longer reaches.
        #   INTERSECT (regex AND fm): the precision lane -- what a
        #          confirm-before-nagging policy would flag.
        lanes["union_regexafterFULL_or_fm"] = score(
            [
                (
                    POSITIVE
                    if POSITIVE in (r.regex_after_full, majority[r.name])
                    else NEGATIVE,
                    r.truth,
                )
                for r in rows
            ]
        )
        lanes["intersect_regexafterFULL_and_fm"] = score(
            [
                (
                    POSITIVE
                    if r.regex_after_full == POSITIVE and majority[r.name] == POSITIVE
                    else NEGATIVE,
                    r.truth,
                )
                for r in rows
            ]
        )
        lanes["intersect_regexbeforeFULL_and_fm"] = score(
            [
                (
                    POSITIVE
                    if r.regex_before_full == POSITIVE and majority[r.name] == POSITIVE
                    else NEGATIVE,
                    r.truth,
                )
                for r in rows
            ]
        )
        for index in range(len(fm_runs)):
            lanes[f"fmtools_run_{index + 1}"] = score(
                [(r.fm_labels[index], r.truth) for r in rows]
            )
        unstable = [
            r.name for r in rows if len(set(r.fm_labels)) > 1
        ]
    else:
        majority = {}
        unstable = []

    report = {
        "study": "ddm_fm3 A1 -- Catalog #344 regex vs fmtools advisory lane",
        "baseline_commit": args.commit,
        "excerpt_chars": args.excerpt_chars,
        "ground_truth_source": (
            ".omx/research/ddm_eq1_equations_leg_backfill_20260904.md section 2"
        ),
        "instruction": INSTRUCTION,
        "n_memos": len(rows),
        "n_positive_truth": sum(1 for r in rows if r.truth == POSITIVE),
        "n_negative_truth": sum(1 for r in rows if r.truth == NEGATIVE),
        "fm_ran": bool(fm_runs),
        "fm_repeats_completed": len(fm_runs),
        "fm_not_run_reasons": fm_reasons,
        "fm_unstable_memos": unstable,
        "lanes": lanes,
        "rows": [
            {
                "memo": r.name,
                "truth": r.truth,
                "regex_before_fix_full": r.regex_before_full,
                "regex_after_fix_full": r.regex_after_full,
                "regex_before_fix_excerpt": r.regex_before,
                "regex_after_fix_excerpt": r.regex_after,
                "fm_labels": r.fm_labels,
                "fm_majority": majority.get(r.name),
                "excerpt_chars": r.excerpt_chars,
                "full_chars": r.full_chars,
            }
            for r in rows
        ],
    }

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{len(rows)} memos at {args.commit} "
          f"({report['n_positive_truth']} MEASURED / {report['n_negative_truth']} WAIVER)")
    header = f"{'lane':<26} {'prec':>6} {'rec':>6} {'F1':>6} {'acc':>6}  tp/fp/fn/tn"
    print(header)
    print("-" * len(header))
    for lane, s in lanes.items():
        print(
            f"{lane:<26} {s['precision']:>6.3f} {s['recall']:>6.3f} "
            f"{s['f1']:>6.3f} {s['accuracy']:>6.3f}  "
            f"{s['tp']}/{s['fp']}/{s['fn']}/{s['tn']}"
        )
    if unstable:
        print(f"\nFM label unstable across repeats on {len(unstable)}: {unstable}")
    if not fm_runs:
        print("\nADVISORY LANE DID NOT RUN -- regex numbers stand alone; "
              "fmtools confirmation OWED.")
    print(f"\nper-memo rows persisted -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
