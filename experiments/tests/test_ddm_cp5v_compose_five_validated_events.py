from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments import ddm_cp5v_compose_five_validated_events as cp5v
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_charter_event_order_and_additivity_pairs_are_pinned() -> None:
    assert cp5v.EVENT_IDS == (
        "ec1_0164_3a4e239de5b9",
        "ec1_0168_818a3c77af51",
        "ec1_0004_3bc2b69c706c",
        "ec1_0104_f4e219067530",
        "ec1_0003_fcb5ca3a4453",
    )
    assert cp5v.AFFECTED_PAIRS == (7, 18, 53, 73, 76, 96)


def test_projected_score_uses_measured_byte_delta() -> None:
    expected = cp5v.jo1.BASE_SCORE - cp5v.EXPECTED_SEG_SCORE_GAIN + 25 / 37_545_489
    assert cp5v.projected_score(1) == expected


def test_actual_token_diffs_reports_only_changed_cells() -> None:
    base = np.zeros((2, 2, cp5v.jo1.WIDTH), dtype=np.uint8)
    candidate = base.copy()
    candidate[0, 1, 3] = 4
    assert cp5v._actual_token_diffs(base, candidate) == [
        {
            "frame": 0,
            "index": cp5v.jo1.WIDTH + 3,
            "y": 1,
            "x": 3,
            "source_class": 0,
            "target_class": 4,
        }
    ]


def test_exact_eval_command_is_canonical_and_sha_pinned() -> None:
    archive = {"path": "/retained/archive.zip", "bytes": 186_253, "sha256": "a" * 64}
    commands = cp5v.exact_eval_commands(archive, Path("/retained/runtime"))
    assert "experiments/modal_auth_eval.py::main" in commands["dispatch"]
    assert "--expected-archive-sha256 " + "a" * 64 in commands["dispatch"]
    assert "--claim-policy require_active" in commands["dispatch"]
    assert "--detach --provider-detach-ack" in commands["dispatch"]


def test_canonical_sparse_logits_are_retained_and_reused(tmp_path: Path) -> None:
    import torch

    calls = {"base": 0}

    class FakeSparse:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.plans = [object(), object()]

        def selected_logits(self, current: object, _context: object, group: int) -> object:
            calls["base"] += 1
            return current.reshape(-1, 1).float() + group

    def decode(_parts: object, _renderer: object, code_dir: Path, _device: object) -> tuple:
        sparse = fake._sparse_class(code_dir)(None)
        current = torch.zeros((1, 1, 2), dtype=torch.long)
        rows = [sparse.selected_logits(current, None, group) for group in (0, 1)]
        return torch.stack(rows), {"decoded_token_sha256": "synthetic"}

    fake = SimpleNamespace(
        _sparse_class=lambda _code_dir: FakeSparse,
        decode_production_tokens=decode,
    )
    code_dir = tmp_path / "cpr1"
    code_dir.mkdir()
    cache = tmp_path / "cache"
    cp5v._decode_with_retained_sparse_logits(fake, None, None, code_dir, torch.device("cpu"), cache)
    assert calls["base"] == 2
    cp5v._decode_with_retained_sparse_logits(fake, None, None, code_dir, torch.device("cpu"), cache)
    assert calls["base"] == 2
    assert len(list(cache.glob("frame_*/group_*.json"))) == 2


def test_cp5v_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_cp5v_compose_five_validated_events.py",
            "experiments/tests/test_ddm_cp5v_compose_five_validated_events.py",
        ),
    )
    assert findings == []
