"""Regression controls for the rp1 empty-dump ledger-loss incident."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.preflight import (
    PreflightError,
    _ddm_ledger_source_violations,
    check_ddm_ledger_before_optional_dump,
)


def findings(body: str) -> list[tuple[int, str]]:
    return _ddm_ledger_source_violations(body)


@pytest.mark.parametrize('operation', ['concatenate', 'stack'])
def test_empty_accepted_list_rejected(operation):
    assert findings(f'accepted = []\nnp.{operation}(accepted)')


@pytest.mark.parametrize('guard', ['if accepted:', 'if len(accepted) > 0:', 'if len(accepted) != 0:'])
def test_nonempty_guard_passes(guard):
    assert not findings(f'accepted = []\n{guard}\n    np.concatenate(accepted)')


def test_wrong_direction_guard_rejected():
    assert findings('accepted = []\nif not accepted:\n    np.stack(accepted)')


def test_other_list_guard_is_not_evidence():
    assert findings('accepted = []\nif other:\n    np.stack(accepted)')


def test_early_empty_return_passes():
    assert not findings('def f():\n    accepted = []\n    if not accepted:\n        return\n    return np.stack(accepted)')


def test_ternary_passes():
    assert not findings('accepted = []\nx = np.concatenate(accepted) if accepted else np.empty(0)')


def test_guard_in_sibling_branch_is_not_dominating():
    assert findings('accepted = []\nif accepted:\n    pass\nnp.stack(accepted)')


def test_alias_is_tracked():
    assert findings('accepted = []\nparts = accepted\nnp.stack(parts)')


def test_filtered_candidates_are_tracked():
    assert findings('candidates = []\nfor item in source:\n    if condition:\n        candidates.append(item)\nnp.stack(candidates)')


def test_nonempty_literal_passes():
    assert not findings('accepted = [one]\nnp.stack(accepted)')


def test_ledger_must_precede_optional_dump():
    assert findings('np.savez_compressed("optional.npz", x=x)\nnp.save("bits_per_frame.npy", bits)')


def test_ledger_first_passes():
    assert not findings('np.save("bits_per_frame.npy", bits)\nnp.savez_compressed("optional.npz", x=x)')


def test_checkpoint_that_itself_persists_ledger_passes():
    assert not findings('np.savez("state.npz", per_frame=bits)\nnp.save("bits_per_frame.npy", bits)')


def test_in_memory_row_append_is_not_persistence():
    assert findings('rows.append(row)\nnp.savez("optional.npz", x=x)\noutput.write_text(json.dumps(rows))')


@pytest.mark.parametrize('rationale', ['TODO add substantive explanation later', '<rationale with at least four words>', 'safe'])
def test_placeholder_waiver_rejected(rationale):
    assert findings(f'accepted = []\nnp.stack(accepted)  # DDM_LEDGER_DUMP_OK:{rationale}')


def test_substantive_same_line_waiver_passes():
    assert not findings('accepted = []\nnp.stack(accepted)  # DDM_LEDGER_DUMP_OK:caller supplies one accepted row under validated contract')


def test_previous_line_waiver_does_not_pass():
    assert findings('accepted = []\n# DDM_LEDGER_DUMP_OK:caller supplies one accepted row under validated contract\nnp.stack(accepted)')


def test_strict_gate_and_scope(tmp_path):
    experiments = tmp_path / 'experiments'
    experiments.mkdir()
    (experiments / 'ddm_bad.py').write_text('accepted = []\nnp.stack(accepted)')
    (experiments / 'unrelated.py').write_text('invalid python text')
    with pytest.raises(PreflightError, match='possibly-empty'):
        check_ddm_ledger_before_optional_dump(repo_root=tmp_path)
    assert len(check_ddm_ledger_before_optional_dump(repo_root=tmp_path, strict=False)) == 1


def test_parse_errors_fail_closed(tmp_path):
    (tmp_path / 'experiments').mkdir()
    (tmp_path / 'experiments/ddm_bad.py').write_text('def broken(')
    with pytest.raises(PreflightError, match='AST-audit'):
        check_ddm_ledger_before_optional_dump(repo_root=tmp_path)


def load_rp1_writer(numpy):
    """Exercise the real pure writer without importing coder/scorer dependencies."""
    path = Path(__file__).resolve().parents[3] / 'experiments/ddm_rp1_rate_rank.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_persist_rank_artifacts')
    namespace = {'np': numpy, 'Path': Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_persist_rank_artifacts']


def empty_bags():
    return {name: [] for name in ('frame', 'pos', 'sym', 'best', 'bits_sym', 'bits_best', 'is_sj1_edit')}


def test_real_rp1_empty_candidates_preserve_ledger_and_typed_dump(tmp_path):
    bits = np.array([1.25, 9.75])
    assert load_rp1_writer(np)(tmp_path, bits, empty_bags()) == 0
    np.testing.assert_array_equal(np.load(tmp_path / 'bits_per_frame.npy'), bits)
    with np.load(tmp_path / 'candidates.npz') as blob:
        assert blob['frame'].dtype == np.int16
        assert blob['pos'].dtype == np.int32
        assert blob['bits_sym'].dtype == np.float32
        assert all(blob[name].size == 0 for name in blob.files)


def test_real_rp1_dump_failure_cannot_erase_ledger(tmp_path):
    def fail(*args, **kwargs):
        raise OSError('injected optional dump failure')
    proxy = SimpleNamespace(**{name: getattr(np, name) for name in ['save', 'int16', 'int32', 'uint8', 'float32', 'ndarray', 'concatenate', 'empty']}, savez_compressed=fail)
    bits = np.array([2.5, 0.125])
    with pytest.raises(OSError, match='injected'):
        load_rp1_writer(proxy)(tmp_path, bits, empty_bags())
    np.testing.assert_array_equal(np.load(tmp_path / 'bits_per_frame.npy'), bits)


def test_real_rp1_nonempty_values_unchanged(tmp_path):
    bags = {name: [np.array([1]), np.array([2, 3])] for name in empty_bags()}
    assert load_rp1_writer(np)(tmp_path, np.array([4.0]), bags) == 3
    with np.load(tmp_path / 'candidates.npz') as blob:
        for name in blob.files:
            np.testing.assert_array_equal(blob[name], np.concatenate(bags[name]))


def test_accepted_parameter_is_possibly_empty():
    assert findings('def f(accepted):\n    return np.stack(accepted)')


@pytest.mark.parametrize('mutation', ['accepted.clear()', 'accepted = []'])
def test_mutation_invalidates_early_guard(mutation):
    assert findings(f'def f():\n    accepted = []\n    if not accepted:\n        return\n    {mutation}\n    return np.stack(accepted)')


def test_literal_empty_list_is_rejected():
    assert findings('np.concatenate([])')


def test_string_is_not_a_waiver_comment():
    assert findings('accepted = []\nresult = np.stack(accepted); text = "# DDM_LEDGER_DUMP_OK:caller supplies one accepted row under validated contract"')


def test_conditional_earlier_ledger_does_not_dominate_dump():
    assert findings('if optional:\n    np.save("bits_per_frame.npy", bits)\nnp.savez("dump.npz", x=x)\nnp.save("bits_per_frame.npy", bits)')


def test_ledger_named_append_is_never_persistence():
    assert findings('ledger_rows = []\nledger_rows.append(row)\nnp.savez("optional.npz", x=x)\noutput.write_text(json.dumps(ledger_rows))')


def test_inner_loop_ledger_does_not_dominate_outer_dump():
    assert findings('for batch in batches:\n    for row in selected:\n        ledger.write_text(json.dumps(row))\n    np.savez("optional.npz", x=x)\n    ledger.write_text(json.dumps(rows))')


def test_outer_ledger_precedes_nested_dump():
    assert not findings('for batch in batches:\n    ledger.write_text(json.dumps(rows))\n    for row in selected:\n        np.savez("optional.npz", x=x)\n    ledger.write_text(json.dumps(rows))')


def test_nested_dump_before_outer_ledger_is_rejected():
    assert findings('for batch in batches:\n    for row in selected:\n        np.savez("optional.npz", x=x)\n    ledger.write_text(json.dumps(rows))')
