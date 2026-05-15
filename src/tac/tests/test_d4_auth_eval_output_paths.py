# SPDX-License-Identifier: MIT
from __future__ import annotations

import tempfile
from pathlib import Path

from experiments.train_substrate_d4_wyner_ziv_frame_0 import (
    _resolve_auth_eval_json_paths,
)


def test_d4_auth_eval_json_uses_non_temp_gate_path_for_modal_tmp_workspace() -> None:
    output_dir = Path(tempfile.gettempdir()) / "pact" / "d4_results" / "output"
    gate_json, local_copy_json = _resolve_auth_eval_json_paths(
        output_dir,
        durable_root=Path("/durable/d4_auth_eval"),
    )

    assert gate_json == Path("/durable/d4_auth_eval/output/contest_auth_eval.json")
    assert local_copy_json == output_dir / "contest_auth_eval.json"


def test_d4_auth_eval_json_stays_in_output_dir_when_output_is_durable() -> None:
    durable_output = Path.cwd() / "experiments" / "results" / "d4_test_output"
    gate_json, local_copy_json = _resolve_auth_eval_json_paths(
        durable_output,
        durable_root=Path("/durable/d4_auth_eval"),
    )

    assert gate_json == durable_output / "contest_auth_eval.json"
    assert local_copy_json == durable_output / "contest_auth_eval.json"
