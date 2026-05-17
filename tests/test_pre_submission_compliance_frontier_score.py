from scripts.pre_submission_compliance_check import _selected_axis_submission_score


def test_selected_axis_submission_score_derives_cpu_strict_formula() -> None:
    score, source = _selected_axis_submission_score(
        explicit_score=None,
        selected_axis="contest_cpu",
        sections={
            "contest_cpu_auth_eval": {
                "record": {"score": 0.199},
                "strict_formula": {"score": 0.1915},
            },
            "auth_eval": {
                "record": {"score": 0.230},
                "strict_formula": {"score": 0.229},
            },
        },
    )

    assert score == 0.1915
    assert source == "strict_formula"


def test_selected_axis_submission_score_uses_cuda_record_without_strict_formula() -> None:
    score, source = _selected_axis_submission_score(
        explicit_score=None,
        selected_axis="contest_cuda",
        sections={
            "auth_eval": {
                "record": {"score": 0.2053},
            },
        },
    )

    assert score == 0.2053
    assert source == "auth_eval_record"


def test_selected_axis_submission_score_explicit_cli_overrides_records() -> None:
    score, source = _selected_axis_submission_score(
        explicit_score=0.188,
        selected_axis="contest_cpu",
        sections={
            "contest_cpu_auth_eval": {
                "record": {"score": 0.199},
                "strict_formula": {"score": 0.1915},
            },
        },
    )

    assert score == 0.188
    assert source == "explicit_cli"
