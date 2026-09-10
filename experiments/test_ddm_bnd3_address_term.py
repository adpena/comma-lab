"""Real-input regression checks; random n32 is an implementation check, not a price."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import ddm_bnd3_address_term as run
from experiments import ddm_bnd3_columnar as columnar


def test_real_random32_joint_skip_decode_and_objective():
    binding, _ = run.bind("tests")
    selected = sorted(np.random.default_rng(run.SEED).choice(600, 32, replace=False).tolist())
    skip_total = 0
    summaries = []
    for index, frame in enumerate(selected):
        data = run.old.load_frame(frame)
        gap = (0, 1, 2, 4)[index % 4]
        orientation = index % 2
        segments, stats = run.joint.joint_segments(data["tokens"], data["best"], data["miss_pos"], orientation, gap)
        greedy, _ = run.old.frame_segments(data["tokens"], data["best"], data["miss_pos"], orientation)
        greedy = [[*s, [False] * len(s[2])] for s in greedy]
        primary, repeat, control = run.pack([segments], 1), run.pack([segments], 1), run.pack([greedy], 1)
        for tag, payload in (("primary", primary), ("repeat", repeat), ("greedy", control)):
            for name, value in payload.items():
                run.blob(run.ROOT / "tests" / f"frame_{frame:04d}.{tag}.{name}", value)
        assert primary == repeat
        assert stats["raw_address_bytes"] == len(primary["address.raw"])
        assert len(primary["address.raw"]) <= len(control["address.raw"])
        assert run.unpack(primary["segments"]) == run.unpack(control["segments"])
        for key, vertex, directions, positions, skip in segments:
            expected_positions = []
            for direction, skipped in zip(directions, skip, strict=True):
                y, x = divmod(vertex, run.old.W + 1)
                run.old.verify_crack_sides(data["tokens"], [(y, x, direction, key[0], key[1])])
                py, px = run.old.sample_edge(y, x, direction, key[2])
                if skipped:
                    assert data["tokens"][py, px] == data["best"][py * run.old.W + px]
                    skip_total += 1
                else:
                    expected_positions.append(py * run.old.W + px)
                vertex = run.joint.end_vertex(vertex, direction)
            assert expected_positions == positions
        with pytest.raises(ValueError):
            run.unpack(primary["segments"] + b"\0")
        summaries.append({"frame": frame, **stats})
    assert skip_total > 0
    run.record(
        run.ROOT / "tests/RESULT.json",
        {
            "binding": binding,
            "selection": selected,
            "purpose": "implementation only; no population prices",
            "skip_edges_verified": skip_total,
            "rows": summaries,
            "passed": True,
        },
    )


def test_original_split_twin_binding():
    result = json.loads((run.ROOT / "decompose/RESULT.json").read_text())
    assert result["twin_identical"]
    assert result["original_total"] == result["original_residual"] + result["original_segment_plus_length"]
    assert (
        result["split_total"]
        == result["original_residual"]
        + result["split_address_bytes"]
        + result["split_content_bytes"]
        + result["split_framing_bytes"]
    )


def test_columnar_real32_support_and_retained_twins():
    selected = json.loads((run.ROOT / "tests/RESULT.json").read_text())["selection"]
    for index, frame in enumerate(selected):
        gap = (0, 1, 2, 4)[index % 4]
        geometry = json.loads((run.ROOT / f"geometry_o1_k{gap}" / f"frame_{frame:04d}.json").read_text())
        segments = geometry["segments"]
        for minimum in (1, 2, 4):
            primary, repeat, fixed = (
                columnar.pack([segments], minimum),
                columnar.pack([segments], minimum),
                run.pack([segments], minimum),
            )
            for tag, payload in (("primary", primary), ("repeat", repeat), ("fixed", fixed)):
                for name, value in payload.items():
                    run.blob(run.ROOT / "tests_planar" / f"frame_{frame:04d}.m{minimum}.{tag}.{name}", value)
            assert primary == repeat
            assert columnar.unpack(primary["segments"]) == run.unpack(fixed["segments"])
            with pytest.raises(ValueError):
                columnar.unpack(primary["segments"] + b"\0")
