from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_pr135ps_truncated_search_resume as audit


def test_extract_f26_trajectory_and_fold_zero_accept() -> None:
    readme = """
F26 continued from the seed. Accepted-row counts by pass were:

```text
4, 2, 0
```
"""
    trajectory = audit.extract_f26_trajectory(readme)
    assert trajectory == (4, 2, 0)
    decision = audit.classify_resume(trajectory)
    assert decision["classification"] == "FOLDED_SOURCE_CONVERGED"
    assert decision["resume_exact_reference_form"] is False


def test_still_accepting_trajectory_requires_resume() -> None:
    decision = audit.classify_resume((4, 2, 1))
    assert decision["classification"] == "RESUME_REQUIRED_STILL_ACCEPTING"
    assert decision["disposition"] == "QUEUED-WITH-A-FIRE-ORDER"
    assert decision["resume_exact_reference_form"] is True


def test_empty_trajectory_is_rejected() -> None:
    with pytest.raises(audit.PR135PSAuditError, match="empty trajectory"):
        audit.classify_resume(())


def test_solver_contract_requires_zero_accept_break() -> None:
    source = "\n".join(
        (
            "for delta in (-1, 1)",
            "improve = best_error < errors - 1e-15",
            "accepted = int(np.count_nonzero(improve))",
            "checkpoint.write_bytes(archive)",
            'state["converged"] = True',
            "if accepted == 0:\n            break",
            "default=12",
        )
    )
    result = audit.verify_solver_contract(source)
    assert result["stopping_rule"] == "accepted_rows == 0"
    with pytest.raises(audit.PR135PSAuditError, match="zero_breaks"):
        audit.verify_solver_contract(source.replace("            break", "            pass"))


def test_signed_codes_from_delta_zigzag_wraps_int12() -> None:
    encoded = np.zeros((audit.FRAMES, audit.DIMENSIONS), dtype=np.uint16)
    encoded[0, 0] = 4095  # zigzag(-2048), represented on the 12-bit ring
    codes = audit.signed_codes_from_delta_zigzag(encoded)
    assert codes[0, 0] == -2048
    assert codes[-1, 0] == -2048


def test_retained_cp135_canonical_carrier_matches_pr135() -> None:
    paths = (
        audit.DEFAULT_BOOK,
        audit.DEFAULT_PR135,
        audit.DEFAULT_CP135,
        audit.DEFAULT_RUNTIME,
    )
    if not all(path.exists() for path in paths):
        pytest.skip("retained PR135/CP135 custody is unavailable")
    args = audit.parser().parse_args([])
    receipt = audit.build_receipt(args)
    assert receipt["decision"]["classification"] == "FOLDED_SOURCE_CONVERGED"
    assert receipt["reference_form"]["final_pass_singleton_proposal_slots"] == 14_280
    assert receipt["reference_form"]["final_pass_valid_singleton_proposal_denominator"] == 14_277
    assert receipt["cp135_carrier_custody"]["canonical_carrier_equal"] is True
    assert receipt["cp135_carrier_custody"]["coefficient_lattice_equal"] is True
    assert receipt["cp135_carrier_custody"]["selector_equal"] is True
    assert Path(receipt["source_pins"]["cp135_archive"]["path"]).is_file()
