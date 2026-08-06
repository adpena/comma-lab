import numpy as np

from tac.optimization.evaluator_invisibility_basis import CAMERA_H, CAMERA_W
from tac.optimization.fd_integer_near_margin_proposals import (
    IntegerNearMarginProposalGenerator,
    near_margin_sites,
)


def test_near_margin_sites_sorts_wrong_pixels_by_current_minus_target_margin():
    logits = np.zeros((3, 4, 4), dtype=np.float64)
    current = np.zeros((4, 4), dtype=np.uint8)
    target = np.zeros((4, 4), dtype=np.uint8)
    current[0, 0] = 1
    target[0, 0] = 2
    logits[1, 0, 0] = 0.4
    logits[2, 0, 0] = 0.2
    current[2, 2] = 1
    target[2, 2] = 2
    logits[1, 2, 2] = 0.7
    logits[2, 2, 2] = 0.65

    sites = near_margin_sites(
        logits_chw=logits,
        realized_argmax=current,
        target_argmax=target,
        max_sites=2,
    )

    assert [(s.scorer_row, s.scorer_col) for s in sites] == [(2, 2), (0, 0)]
    assert sites[0].margin_current_minus_target == 0.04999999999999993


def test_generator_validates_realized_candidate_loop():
    frame = np.full((CAMERA_H, CAMERA_W, 3), 127, dtype=np.uint8)
    base = np.full((384, 512, 3), 127, dtype=np.uint8)
    target_rgb = base.copy()
    target_rgb[0:2, 0:2] = 130
    logits = np.zeros((3, 384, 512), dtype=np.float64)
    current = np.zeros((384, 512), dtype=np.uint8)
    target = np.zeros((384, 512), dtype=np.uint8)
    current[0, 0] = 1
    target[0, 0] = 2
    logits[1, 0, 0] = 0.2
    logits[2, 0, 0] = 0.1

    calls = []

    def validator(candidate, context):
        calls.append((candidate, context))
        return {"accepted": True, "flips_before": 1, "flips_after": 0}

    result = IntegerNearMarginProposalGenerator(method="naive").generate(
        camera_frame=frame,
        base_scorer_hwc=base,
        target_scorer_hwc=target_rgb,
        logits_chw=logits,
        realized_argmax=current,
        target_argmax=target,
        max_proposals=1,
        validator=validator,
    )

    assert result["n_proposals"] == 1
    assert result["n_accepted"] == 1
    assert calls[0][1]["site"]["scorer_row"] == 0
    assert calls[0][1]["selected_method"] == "naive"
