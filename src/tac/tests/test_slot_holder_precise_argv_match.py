# SPDX-License-Identifier: MIT
"""ddm_gh1 #829 — slot-holder guards must not REFUSE A LAUNCH on a substring coincidence.

Bug class (false refusal): ``tools/ru1_token_quantum_calibration.py`` and ``tools/sb1_seg_batch.py``
decided "is a scorer slot busy?" with ``any(tok in line for line in ps_output.splitlines())`` — a
bare SUBSTRING scan of the process table. Any unrelated process whose command line merely CONTAINED
a token (a background ``grep``/``rg`` for it, an editor, a pager, or an ``--out`` path with the
token in a DIRECTORY component) made the guard report the slot busy and the tool refused to launch.
A gate that wrongly refuses is worse than no gate: it trains everyone to pass ``--skip-slot-check``.

The cure is structural and canonical: ``tools.argv_role.cmdline_names_entrypoint`` requires the
token to appear in the BASENAME of a PATH-SHAPED argument of a process whose PROGRAM is not a
reader/viewer. These tests are the positive control (real holders must STILL be caught) plus the
constructed false-positive corpus (must no longer fire).
"""
from __future__ import annotations

import pytest

argv_role = pytest.importorskip("tools.argv_role")

SLOT_TOKENS = ("pb1_receiver_realized_verdict", "train_levelset_witness",
               "train_witness_realized", "ddm_lv1_s2_nullspace",
               "contest_auth_eval", "evaluate.py", "ru1_endpoint_residual", "pb1_qdbs")

# --- the measured false-positive corpus: NONE of these holds a scorer slot -------------------
FALSE_POSITIVES = (
    'grep -rn train_levelset_witness src/',
    'grep -rn "evaluate.py" tools/',
    'rg --json contest_auth_eval experiments/',
    '/opt/homebrew/bin/rg -n pb1_qdbs',
    'vim /Users/x/notes/train_levelset_witness.md',
    'less /Volumes/VertigoDataTier/pact/train_witness_realized_run/telemetry.jsonl',
    'tail -f /Volumes/D/ddm_lv1_s2_nullspace_run/log.txt',
    # a real, unrelated job whose OUTPUT path merely contains a token in a directory component
    '.venv/bin/python tools/other_tool.py --out /Volumes/D/train_levelset_witness_run/t.jsonl',
    # an observer/monitor carrying the trainer name as a flag VALUE (the #406/#512 sister class)
    '.venv/bin/python tools/witness_checkin.py --training-sig train_levelset_witness',
    'sed -n 1,50p tools/ru1_endpoint_residual.py',
    'cat experiments/contest_auth_eval_notes.txt',
)

# --- the positive control: every one of these IS a live slot holder and MUST still fire -------
TRUE_HOLDERS = (
    '.venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py --epochs 4',
    'OMP_NUM_THREADS=1 .venv/bin/python tools/run_ddm_pb1_receiver_realized_verdict.py --n 600',
    'python3 upstream/evaluate.py --device cpu',
    '/usr/bin/python3 tools/pb1_qdbs.py --pairs 600',
    'bash -c "cd /repo && .venv/bin/python tools/ru1_endpoint_residual.py --out x"',
    'python experiments/train_witness_realized_through_R_mlx.py',
)


@pytest.mark.parametrize("line", FALSE_POSITIVES)
def test_reader_and_path_coincidences_do_not_hold_the_slot(line):
    assert argv_role.cmdline_names_entrypoint(line, SLOT_TOKENS) is False, (
        f"FALSE REFUSAL: {line!r} does not run a scorer entrypoint but was classified as one"
    )


@pytest.mark.parametrize("line", TRUE_HOLDERS)
def test_real_slot_holders_are_still_caught(line):
    """POSITIVE CONTROL. A narrowing that guts the detector must fail loudly here."""
    assert argv_role.cmdline_names_entrypoint(line, SLOT_TOKENS) is True, (
        f"DETECTOR GUTTED: {line!r} is a genuine slot holder and must be caught"
    )


def test_legacy_substring_form_would_have_refused_every_false_positive():
    """Pins the DEFECT itself, so a regression to the old form is visible as a behaviour change."""
    legacy = [ln for ln in FALSE_POSITIVES if any(tok in ln for tok in SLOT_TOKENS)]
    assert len(legacy) == len(FALSE_POSITIVES), "corpus must exercise the old substring rule"
    precise = [ln for ln in FALSE_POSITIVES
               if argv_role.cmdline_names_entrypoint(ln, SLOT_TOKENS)]
    assert precise == []


def test_process_table_holders_returns_the_offending_lines():
    ps_output = "\n".join(FALSE_POSITIVES + TRUE_HOLDERS)
    holders = argv_role.process_table_entrypoint_holders(ps_output, SLOT_TOKENS)
    assert sorted(holders) == sorted(line.strip() for line in TRUE_HOLDERS)


def test_self_exclusion_is_precise_not_substring():
    """A tool must not see itself — and must not go blind to an unrelated process just because
    that process's argv mentions the tool's own name."""
    ps_output = "\n".join((
        '.venv/bin/python tools/sb1_seg_batch.py --stage a',            # self  -> excluded
        'grep -rn sb1_seg_batch tools/',                                # reader -> ignored
        # a REAL holder whose argv also mentions the self token: must still be reported
        '.venv/bin/python tools/pb1_qdbs.py --note sb1_seg_batch_followup',
    ))
    holders = argv_role.process_table_entrypoint_holders(
        ps_output, SLOT_TOKENS, self_tokens=("sb1_seg_batch",))
    assert holders == ['.venv/bin/python tools/pb1_qdbs.py --note sb1_seg_batch_followup']


def _substring_scans_over_process_lines(tree) -> list[int]:
    """Line numbers of ``<tok> in <line>`` membership tests inside a comprehension that iterates
    process-table lines. AST-based on purpose: a prose mention of the old form in a docstring or
    comment must NOT count (the first draft of this test failed on its own explanatory docstring —
    exactly the substring-vs-structure error the fix is about)."""
    import ast

    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            continue
        targets = {
            gen.target.id for gen in node.generators if isinstance(gen.target, ast.Name)
        }
        if not targets & {"line", "row", "cmd", "cmdline"}:
            continue
        for inner in ast.walk(node.elt):
            if isinstance(inner, ast.Compare) and any(
                isinstance(op, ast.In) for op in inner.ops
            ):
                comparators = [
                    c.id for c in inner.comparators if isinstance(c, ast.Name)
                ]
                if set(comparators) & targets:
                    hits.append(inner.lineno)
    return hits


def test_both_tools_route_through_the_canonical_helper():
    """Wire-in proof (Catalog #125 pillar 1): the fix is not stranded in the helper."""
    import ast
    import pathlib

    for rel in ("tools/ru1_token_quantum_calibration.py", "tools/sb1_seg_batch.py"):
        text = pathlib.Path(rel).read_text()
        tree = ast.parse(text)
        assert "process_table_entrypoint_holders" in text, f"{rel} does not use the helper"
        scans = _substring_scans_over_process_lines(tree)
        assert scans == [], f"{rel} still substring-scans process lines at {scans}"


def test_the_ast_detector_itself_still_fires_on_the_defect():
    """POSITIVE CONTROL for the detector above: if it stops flagging the original defective
    source, the wire-in test has silently become a no-op."""
    import ast

    defective = (
        "def slot_is_live():\n"
        "    return any(tok in line for line in out.splitlines()\n"
        "               for tok in SLOT_TOKENS if 'sb1_seg_batch' not in line)\n"
    )
    assert _substring_scans_over_process_lines(ast.parse(defective)) != []
