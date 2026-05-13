"""Tests for the reusable HNeRV trainer parity guard."""

from __future__ import annotations

from pathlib import Path

from tac.hnerv_training_parity_guard import (
    assert_hnerv_training_parity_file,
    inspect_hnerv_training_parity_source,
)
from tac.preflight import check_hnerv_training_parity_guard

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_guard_accepts_primary_sane_hnerv_trainer() -> None:
    assert_hnerv_training_parity_file(
        REPO_ROOT / "experiments" / "train_substrate_sane_hnerv.py"
    )


def test_guard_accepts_wave3_tc_nerv_trainer() -> None:
    assert_hnerv_training_parity_file(
        REPO_ROOT / "experiments" / "train_substrate_tc_nerv.py"
    )


def test_preflight_wrapper_enforces_live_hnerv_trainer_parity() -> None:
    assert check_hnerv_training_parity_guard(strict=False, verbose=False) == []


def test_guard_rejects_runtime_without_exact_file_list_signature() -> None:
    text = _GOOD_TRAINER_SOURCE.replace(
        '        "FILE_LIST=\\"$3\\"\\n"\n',
        "",
    ).replace(
        '        "\\"$DATA_DIR\\" \\"$OUTPUT_DIR\\" \\"$FILE_LIST\\"\\n"\n',
        '        "\\"$DATA_DIR\\" \\"$OUTPUT_DIR\\"\\n"\n',
    )
    report = inspect_hnerv_training_parity_source(text, path_label="bad_runtime.py")
    assert not report.passed
    assert any("file_list" in violation for violation in report.violations)


def test_guard_rejects_scorer_load_before_yuv6_patch() -> None:
    text = _GOOD_TRAINER_SOURCE.replace(
        "    yuv6_token = patch_upstream_yuv6_globally()\n"
        "    posenet, segnet = load_differentiable_scorers(args.upstream_dir)\n",
        "    posenet, segnet = load_differentiable_scorers(args.upstream_dir)\n"
        "    yuv6_token = patch_upstream_yuv6_globally()\n",
    )
    report = inspect_hnerv_training_parity_source(text, path_label="bad_yuv6.py")
    assert not report.passed
    assert any("before patching" in violation for violation in report.violations)


def test_guard_rejects_missing_archive_build_loop_and_ema_shadow() -> None:
    text = _GOOD_TRAINER_SOURCE.replace("    ema.apply(model)\n", "").replace(
        "    ema_state = ema.state_dict()\n",
        "",
    ).replace(
        "    bin_bytes = pack_archive(decoder_sd, latents, meta)\n",
        "",
    ).replace(
        "    _build_archive_zip(\n"
        "        args.output_dir / 'archive.zip',\n"
        "        bin_bytes=bin_bytes,\n"
        "        submission_dir=args.output_dir / 'submission',\n"
        "    )\n",
        "",
    )
    report = inspect_hnerv_training_parity_source(text, path_label="bad_archive.py")
    assert not report.passed
    joined = "\n".join(report.violations)
    assert "ema.apply" in joined
    assert "ema.state_dict()" in joined
    assert "pack_archive" in joined
    assert "_build_archive_zip" in joined


def test_guard_rejects_comment_only_or_dead_string_contracts() -> None:
    text = '''
def _write_runtime(submission_dir):
    inflate_sh = (
        "#!/usr/bin/env bash\\n"
        "set -euo pipefail\\n"
        "DATA_DIR=\\"$1\\"\\n"
        "OUTPUT_DIR=\\"$2\\"\\n"
        "FILE_LIST=\\"$3\\"\\n"
        "exec python3 inflate.py \\"$DATA_DIR\\" \\"$OUTPUT_DIR\\" \\"$FILE_LIST\\"\\n"
    )
    inflate_py = (
        "import sys\\n"
        "from pathlib import Path\\n"
        "def main():\\n"
        "    if len(sys.argv) != 4:\\n"
        "        return 2\\n"
        "    archive_dir = Path(sys.argv[1])\\n"
        "    output_dir = Path(sys.argv[2])\\n"
        "    file_list_path = Path(sys.argv[3])\\n"
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\\n"
        "        pass\\n"
    )


def _full_main(args):
    """
    patch_upstream_yuv6_globally()
    load_differentiable_scorers()
    apply_eval_roundtrip=True
    EMA(model); ema.update(model); ema.apply(model); ema.state_dict()
    pack_archive(...); _write_runtime(...); _build_archive_zip(...)
    """
    # A comment mentioning patch_upstream_yuv6_globally before load_differentiable_scorers
    note = "apply_eval_roundtrip=True EMA( ema.update ema.apply ema.state_dict() pack_archive _write_runtime _build_archive_zip"
    def not_called():
        patch_upstream_yuv6_globally()
        load_differentiable_scorers()
        loss_fn(apply_eval_roundtrip=True)
        ema = EMA(model)
        ema.update(model)
        ema.apply(model)
        ema.state_dict()
        pack_archive()
        _write_runtime()
        _build_archive_zip()
    return note
'''
    report = inspect_hnerv_training_parity_source(text, path_label="dead_contracts.py")
    assert not report.passed
    joined = "\n".join(report.violations)
    assert "missing patch_upstream_yuv6_globally" in joined
    assert "missing load_differentiable_scorers" in joined
    assert "missing apply_eval_roundtrip=True" in joined
    assert "EMA" in joined
    assert "pack_archive" in joined


def test_guard_rejects_inflate_py_docstring_only_runtime_contracts() -> None:
    spoofed_inflate_py = """\
import sys
from pathlib import Path

def main() -> int:
    '''
    if len(sys.argv) != 4: pass
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / '0.bin').read_bytes()
    for line in file_list_path.read_text().splitlines(): pass
    '''
    dead_contract = "sys.argv[3] splitlines() archive_dir / '0.bin'"
    return 0
"""
    text = _source_with_runtime(_GOOD_INFLATE_SH, spoofed_inflate_py)
    report = inspect_hnerv_training_parity_source(
        text,
        path_label="spoofed_inflate_py.py",
    )
    assert not report.passed
    joined = "\n".join(report.violations)
    assert "does not enforce 3 positional args" in joined
    assert "does not consume file_list argv" in joined
    assert "does not iterate file_list lines" in joined
    assert "does not consume archive_dir/0.bin" in joined


def test_guard_rejects_inverted_sys_argv_arity_check() -> None:
    inverted_inflate_py = _GOOD_INFLATE_PY.replace(
        "if len(sys.argv) != 4:",
        "if len(sys.argv) == 4:",
    )
    text = _source_with_runtime(_GOOD_INFLATE_SH, inverted_inflate_py)
    report = inspect_hnerv_training_parity_source(
        text,
        path_label="inverted_arity_runtime.py",
    )
    assert not report.passed
    assert any(
        "does not enforce 3 positional args" in violation
        for violation in report.violations
    )


def test_guard_rejects_inflate_sh_comment_only_signature() -> None:
    spoofed_inflate_sh = """\
#!/usr/bin/env bash
# set -euo pipefail
# DATA_DIR="$1"
# OUTPUT_DIR="$2"
# FILE_LIST="$3"
# exec python3 inflate.py "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
exec python3 inflate.py
"""
    text = _source_with_runtime(spoofed_inflate_sh, _GOOD_INFLATE_PY)
    report = inspect_hnerv_training_parity_source(
        text,
        path_label="spoofed_inflate_sh.py",
    )
    assert not report.passed
    joined = "\n".join(report.violations)
    assert "archive_dir/output_dir/file_list" in joined
    assert "missing set -e/pipefail" in joined


def test_guard_rejects_executable_scorer_import_in_inflate_py() -> None:
    scorer_import_inflate_py = _GOOD_INFLATE_PY.replace(
        "from pathlib import Path\n",
        "from pathlib import Path\nfrom upstream.modules import PoseNet\n",
    )
    text = _source_with_runtime(_GOOD_INFLATE_SH, scorer_import_inflate_py)
    report = inspect_hnerv_training_parity_source(
        text,
        path_label="scorer_import_runtime.py",
    )
    assert not report.passed
    assert any("forbidden scorer token" in violation for violation in report.violations)


def _source_with_runtime(inflate_sh: str, inflate_py: str) -> str:
    return f'''
def _write_runtime(submission_dir):
    inflate_sh = {inflate_sh!r}
    inflate_py = {inflate_py!r}

{_GOOD_FULL_MAIN_SOURCE}
'''


_GOOD_INFLATE_SH = """\
#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
"""


_GOOD_INFLATE_PY = """\
import sys
from pathlib import Path

def main() -> int:
    if len(sys.argv) != 4:
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / '0.bin').read_bytes()
    for line in file_list_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
    return 0
"""


_GOOD_FULL_MAIN_SOURCE = '''
def _full_main(args):
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.scorer import load_differentiable_scorers
    from tac.training import EMA

    yuv6_token = patch_upstream_yuv6_globally()
    posenet, segnet = load_differentiable_scorers(args.upstream_dir)
    ema = EMA(model, decay=args.ema_decay)
    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, archive_bytes_proxy,
        apply_eval_roundtrip=True,
    )
    ema.update(model)
    ema.apply(model)
    ema_state = ema.state_dict()
    torch.save({"state_dict": ema_state}, ckpt_best_path)
    bin_bytes = pack_archive(decoder_sd, latents, meta)
    _write_runtime(args.output_dir / 'submission')
    _build_archive_zip(
        args.output_dir / 'archive.zip',
        bin_bytes=bin_bytes,
        submission_dir=args.output_dir / 'submission',
    )
    provenance = {"ready_for_exact_eval_dispatch": False}
'''


_GOOD_TRAINER_SOURCE = '''
def _write_runtime(submission_dir):
    inflate_sh = (
        "#!/usr/bin/env bash\\n"
        "set -euo pipefail\\n"
        "DATA_DIR=\\"$1\\"\\n"
        "OUTPUT_DIR=\\"$2\\"\\n"
        "FILE_LIST=\\"$3\\"\\n"
        "exec \\"${PYTHON:-python3}\\" \\"$HERE/inflate.py\\" "
        "\\"$DATA_DIR\\" \\"$OUTPUT_DIR\\" \\"$FILE_LIST\\"\\n"
    )
    inflate_py = (
        "import sys\\n"
        "from pathlib import Path\\n"
        "def main():\\n"
        "    if len(sys.argv) != 4:\\n"
        "        return 2\\n"
        "    archive_dir = Path(sys.argv[1])\\n"
        "    output_dir = Path(sys.argv[2])\\n"
        "    file_list_path = Path(sys.argv[3])\\n"
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\\n"
        "        line = line.strip()\\n"
        "    return 0\\n"
    )


def _full_main(args):
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.scorer import load_differentiable_scorers
    from tac.training import EMA

    yuv6_token = patch_upstream_yuv6_globally()
    posenet, segnet = load_differentiable_scorers(args.upstream_dir)
    ema = EMA(model, decay=args.ema_decay)
    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, archive_bytes_proxy,
        apply_eval_roundtrip=True,
    )
    ema.update(model)
    ema.apply(model)
    ema_state = ema.state_dict()
    torch.save({"state_dict": ema_state}, ckpt_best_path)
    bin_bytes = pack_archive(decoder_sd, latents, meta)
    _write_runtime(args.output_dir / 'submission')
    _build_archive_zip(
        args.output_dir / 'archive.zip',
        bin_bytes=bin_bytes,
        submission_dir=args.output_dir / 'submission',
    )
    provenance = {"ready_for_exact_eval_dispatch": False}
'''
