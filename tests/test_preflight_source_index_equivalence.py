from __future__ import annotations

from pathlib import Path

from tac import preflight
from tac.source_index import source_index_context


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _with_source_index(root: Path, fn):
    with source_index_context(root):
        return fn(repo_root=root, strict=False, verbose=False)


def test_comment_only_contract_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "remote_lane_demo.sh",
        """#!/usr/bin/env bash
# THE DEPLOY SCRIPT CALLS the real trainer later.
python - <<'PY'
print("stub")
PY
""",
    )
    _write(
        tmp_path / "src" / "tac" / "guarded_contract.py",
        '''def guarded() -> None:
    """The deploy script calls the real implementation."""
    raise RuntimeError("wrapper did not replace guarded")
''',
    )

    no_index = preflight.check_no_comment_only_contracts(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(tmp_path, preflight.check_no_comment_only_contracts)

    assert with_index == no_index
    assert len(with_index) == 1
    assert "scripts/remote_lane_demo.sh:2" in with_index[0]


def test_bare_round_roundtrip_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "tac" / "roundtrip_bad.py",
        """import torch.nn.functional as F

def eval_roundtrip(x):
    y = F.interpolate(x, scale_factor=2)
    return y.round()
""",
    )
    _write(
        tmp_path / "experiments" / "roundtrip_good.py",
        """import torch.nn.functional as F

def eval_roundtrip_ok(x):
    y = F.interpolate(x, scale_factor=2)
    return y.round() + (y - y.round()).detach()
""",
    )

    no_index = preflight.check_no_bare_round_in_eval_roundtrip(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(tmp_path, preflight.check_no_bare_round_in_eval_roundtrip)

    assert with_index == no_index
    assert len(with_index) == 1
    assert "src/tac/roundtrip_bad.py:5" in with_index[0]


def test_profile_resolver_source_index_matches_no_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE", "1")
    monkeypatch.setattr(
        preflight,
        "_extract_profile_keys",
        lambda: {"alpha_key", "beta_key"},
    )
    _write(tmp_path / "src" / "tac" / "profiles.py", "PROFILES = {}\n")
    _write(
        tmp_path / "experiments" / "train_demo.py",
        """def train(alpha_key: int = 1) -> int:
    return alpha_key
""",
    )
    _write(
        tmp_path / "src" / "tac" / "irrelevant.py",
        "VALUE = 'no profile keys here'\n",
    )

    no_index = preflight.check_profile_keys_have_resolvers(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(tmp_path, preflight.check_profile_keys_have_resolvers)

    assert with_index == no_index
    assert len(with_index) == 1
    assert "beta_key" in with_index[0]


def test_codebase_drift_source_index_matches_no_index_for_forbidden_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE", "1")
    _write(
        tmp_path / "scripts" / "bad_dispatch.py",
        """import subprocess

def launch() -> None:
    cmd = "while pgrep -f train_distill > /dev/null; do sleep 1; done; python train_distill.py"
    subprocess.run(cmd, shell=True)

def tmp_exec() -> None:
    cmd = "python /tmp/run_remote.py --profile green --padding filler"
    subprocess.run(cmd, shell=True)
""",
    )
    _write(
        tmp_path / "scripts" / "safe_dispatch.py",
        """import subprocess

def launch() -> None:
    subprocess.run(["python", "experiments/pipeline.py"])
""",
    )

    no_index = preflight.check_codebase_drift(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(tmp_path, preflight.check_codebase_drift)

    assert with_index == no_index
    assert len(with_index) == 2
    assert any("pgrep -f train_distill" in row for row in with_index)
    assert any("/tmp/*.{" in row for row in with_index)


def test_codebase_drift_source_index_candidate_filter_does_not_call_rg(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE", "1")
    _write(
        tmp_path / "scripts" / "nohup_dispatch.py",
        """import subprocess

def launch() -> None:
    subprocess.run(["nohup", "python", "experiments/pipeline.py"])
""",
    )

    def _forbidden_rg_call(*_args, **_kwargs):
        raise AssertionError("SourceIndex path must not shell out to rg prefilter")

    monkeypatch.setattr(preflight, "_rg_python_files_matching_regex", _forbidden_rg_call)

    with source_index_context(tmp_path):
        rows = preflight.check_codebase_drift(
            repo_root=tmp_path,
            strict=False,
            verbose=False,
        )

    assert len(rows) == 1
    assert "nohup arg" in rows[0]


def test_training_synthetic_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "experiments" / "train_bad.py",
        """def make_synthetic_pair_batch():
    return object()

def main() -> int:
    batch = make_synthetic_pair_batch()
    return 0 if batch else 1
""",
    )
    _write(
        tmp_path / "experiments" / "train_large_safe.py",
        "\n".join(["def main() -> int:", "    return 0"] + ["# filler"] * 200),
    )

    no_index = preflight.check_training_scripts_use_real_data_in_nonsmoke_mode(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_training_scripts_use_real_data_in_nonsmoke_mode,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "train_bad.py:5" in with_index[0]


def test_state_writer_strict_load_source_index_matches_no_index(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src" / "tac" / "state_bad.py",
        """import json

def update_records_locked(path):
    rows = _load_records(path)
    path.write_text(json.dumps(rows))
""",
    )

    no_index = preflight.check_state_writers_strict_load_for_mutating_path(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_state_writers_strict_load_for_mutating_path,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "state_bad.py:3" in with_index[0]


def test_phase_b_auth_memo_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "experiments" / "phase_b_bad.py",
        """from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status

def main():
    return phase_b_preconditions_status(auth_memo_path="/tmp/operator.md")
""",
    )

    no_index = preflight.check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_phase_b_auth_memo_in_repo,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "phase_b_bad.py:4" in with_index[0]


def test_block_fp_qint_exp_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "tac" / "consumer_bad.py",
        """def decode(state):
    return state["weight_qint"]
""",
    )
    _write(
        tmp_path / "src" / "tac" / "consumer_good.py",
        """def decode(state):
    return state["weight_qint"], state["weight_exponents"]
""",
    )

    no_index = preflight.check_block_fp_exponents_alongside_qint(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_block_fp_exponents_alongside_qint,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "consumer_bad.py" in with_index[0]


def test_segmap_export_roundtrip_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "experiments" / "export_bad.py",
        """from tac.block_fp_codec import pack_payload_tar_xz

def export(state, path):
    return pack_payload_tar_xz(state, path)
""",
    )
    _write(
        tmp_path / "experiments" / "export_good.py",
        """from tac.block_fp_codec import pack_payload_tar_xz, verify_roundtrip

def export(state, path):
    payload = pack_payload_tar_xz(state, path)
    verify_roundtrip(state, path)
    return payload
""",
    )

    no_index = preflight.check_segmap_export_calls_verify_roundtrip(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_segmap_export_calls_verify_roundtrip,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "export_bad.py" in with_index[0]


def test_phase3_dispatch_gate_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "tac" / "phase3_callsite.py",
        """def build_gate():
    return Phase3DispatchGate(
        phase2_anchor_verified=True,
    )
""",
    )

    no_index = preflight.check_phase3_dispatch_gate_fail_closed(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_phase3_dispatch_gate_fail_closed,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "phase3_callsite.py:2" in with_index[0]


def test_setup_first_seen_transactional_source_index_matches_no_index(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "scripts" / "verify_vast_instances.py",
        """def _load_setup_first_seen():
    return {}

def _save_setup_first_seen(state):
    return None

def main():
    state = _load_setup_first_seen()
    state["abc"] = 1
    _save_setup_first_seen(state)
""",
    )

    no_index = preflight.check_setup_first_seen_uses_transactional_update_inside_lock(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_setup_first_seen_uses_transactional_update_inside_lock,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "verify_vast_instances.py:8" in with_index[0]


def test_packet_no_op_proof_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "tac" / "packet_bad.py",
        """def finalize_packet():
    blockers = []
    proof = _build_no_op_proof()
    if proof is not None:
        blockers.append("advisory_only")
    return blockers
""",
    )

    no_index = preflight.check_packet_compiler_no_op_proof_promotes_to_blocker(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_packet_compiler_no_op_proof_promotes_to_blocker,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "packet_bad.py:1" in with_index[0]


def test_paid_job_register_before_submit_source_index_matches_no_index(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src" / "tac" / "deploy" / "lightning_submit_bad.py",
        """class Job:
    @staticmethod
    def run(*args):
        return object()

def submit():
    return Job.run("train")
""",
    )

    no_index = preflight.check_paid_job_register_before_submit(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_paid_job_register_before_submit,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "lightning_submit_bad.py:6" in with_index[0]


def test_setup_first_seen_no_split_source_index_matches_no_index(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "scripts" / "verify_vast_instances.py",
        """def reconcile(observed, left):
    update_setup_first_seen_locked(observed)
    remove_setup_first_seen_locked(left)
""",
    )

    no_index = preflight.check_setup_first_seen_no_split_transactions(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(
        tmp_path,
        preflight.check_setup_first_seen_no_split_transactions,
    )

    assert with_index == no_index
    assert len(with_index) == 1
    assert "verify_vast_instances.py:1" in with_index[0]


def test_custody_window_tokenizer_handles_mid_block_indentation() -> None:
    lines = [
        "if current_score is not None:",
        "    is_authoritative = result.evidence_tag in AUTHORITATIVE_TAGS",
        "elif delta_vs_baseline is not None and delta_vs_baseline > score_tolerance:",
        "    verdict = result.validate_custody()",
    ]

    assert preflight._line_window_code_contains_any(
        lines,
        1,
        ("validate_custody",),
        before=0,
        after=3,
    )


def test_authoritative_tag_scanner_ignores_eval_axis_classification() -> None:
    assert not preflight._line_has_authoritative_tag_bypass_pattern(
        'if eval_axis == "[contest-CUDA]":'
    )
    assert not preflight._line_has_authoritative_tag_bypass_pattern(
        'if "[contest-CPU]" != eval_axis:'
    )
    assert preflight._line_has_authoritative_tag_bypass_pattern(
        'if result.evidence_tag == "[contest-CUDA]":'
    )
