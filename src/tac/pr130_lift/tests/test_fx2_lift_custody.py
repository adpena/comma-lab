from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tac.pr130_lift import LIFTED_AT_HEAD, SOURCE_REPO_HEAD, SOURCE_REPO_ROOT

REPO_ROOT = Path(__file__).resolve().parents[4]
LIFT_ROOT = REPO_ROOT / "src/tac/pr130_lift"
SOURCE_SHA256_SCOPE = (
    "exact source_path bytes at source_repo_head before adaptations"
)
ADMISSION_ADAPTATION = "governed_admission_guard_after_argparse"


@dataclass(frozen=True)
class CustodyExpectation:
    source_path: str
    adaptation: str = "none"


EXPECTED_CUSTODY = {
    "lifted/evaluate_semantic_quantization.py": CustodyExpectation(
        "code/evaluate_semantic_quantization.py"
    ),
    "lifted/semantic_renderer_oracle.py": CustodyExpectation(
        "code/semantic_renderer_oracle.py"
    ),
    "lifted/train_semantic_full.py": CustodyExpectation(
        "code/train_semantic_full.py", ADMISSION_ADAPTATION
    ),
    "lifted/train_semantic_quantized.py": CustodyExpectation(
        "code/train_semantic_quantized.py", ADMISSION_ADAPTATION
    ),
    "pose/lifted/carrier_codec.py": CustodyExpectation("code/carrier_codec.py"),
    "pose/lifted/learned_pose_carrier_oracle.py": CustodyExpectation(
        "code/learned_pose_carrier_oracle.py"
    ),
    "pose/lifted/pack_semantic_pose.py": CustodyExpectation(
        "code/pack_semantic_pose.py"
    ),
    "pose/lifted/pose_basis_oracle.py": CustodyExpectation(
        "code/pose_basis_oracle.py"
    ),
    "pose/lifted/refine_pose_coeff_codes.py": CustodyExpectation(
        "code/refine_pose_coeff_codes.py"
    ),
    "pose/lifted/repack_carrier.py": CustodyExpectation("code/repack_carrier.py"),
    "pose/lifted/search_pose_coeff_cpu.py": CustodyExpectation(
        "code/search_pose_coeff_cpu.py"
    ),
    "pose/lifted/train_pose_carrier_full.py": CustodyExpectation(
        "code/train_pose_carrier_full.py", ADMISSION_ADAPTATION
    ),
}


ADMISSION_PATCHES = {
    "code/train_semantic_full.py": (
        b"    args = parser.parse_args()\n\n    torch.manual_seed(args.seed)\n",
        b"    args = parser.parse_args()\n"
        b"    from tac.admission_guard import assert_governed_admission\n"
        b"    assert_governed_admission(\"train_semantic_full\")\n\n"
        b"    torch.manual_seed(args.seed)\n",
    ),
    "code/train_semantic_quantized.py": (
        b"    args = parser.parse_args()\n    if not 2 <= args.bits <= 8:\n",
        b"    args = parser.parse_args()\n"
        b"    from tac.admission_guard import assert_governed_admission\n"
        b"    assert_governed_admission(\"train_semantic_quantized\")\n"
        b"    if not 2 <= args.bits <= 8:\n",
    ),
    "code/train_pose_carrier_full.py": (
        b"    args = parser.parse_args()\n    if args.steps < 1:\n",
        b"    args = parser.parse_args()\n"
        b"    from tac.admission_guard import assert_governed_admission\n"
        b"    assert_governed_admission(\"train_pose_carrier_full\")\n"
        b"    if args.steps < 1:\n",
    ),
}


def _header_fields(data: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in data.splitlines():
        if raw_line == b"# borrowed_substrate_accounting:":
            continue
        if not raw_line.startswith(b"#   "):
            if fields:
                break
            continue
        key, separator, value = raw_line[4:].partition(b":")
        if separator:
            fields[key.decode("utf-8")] = value.strip().decode("utf-8")
    return fields


def _body_without_accounting_header(data: bytes) -> bytes:
    shebang_end = data.index(b"\n") + 1
    body_start = data.index(b'"""', shebang_end)
    return data[:shebang_end] + data[body_start:]


def _source_bytes(source_path: str) -> bytes:
    return subprocess.check_output(
        [
            "git",
            "-C",
            SOURCE_REPO_ROOT,
            "show",
            f"{SOURCE_REPO_HEAD}:{source_path}",
        ]
    )


def _apply_declared_adaptation(
    original: bytes, expectation: CustodyExpectation
) -> bytes:
    if expectation.adaptation == "none":
        return original
    if expectation.adaptation != ADMISSION_ADAPTATION:
        raise AssertionError(f"unimplemented adaptation {expectation.adaptation!r}")
    needle, replacement = ADMISSION_PATCHES[expectation.source_path]
    assert original.count(needle) == 1, (
        f"pinned source no longer has exactly one patch anchor: "
        f"{expectation.source_path}"
    )
    return original.replace(needle, replacement, 1)


def test_all_original_backed_lifts_reconstruct_from_pinned_intake() -> None:
    assert Path(SOURCE_REPO_ROOT).is_dir(), "pinned PR130 intake is unavailable"
    current_source_head = subprocess.check_output(
        ["git", "-C", SOURCE_REPO_ROOT, "rev-parse", "HEAD"], text=True
    ).strip()
    assert current_source_head == SOURCE_REPO_HEAD

    discovered = {
        str(path.relative_to(LIFT_ROOT))
        for directory in (LIFT_ROOT / "lifted", LIFT_ROOT / "pose/lifted")
        for path in directory.glob("*.py")
        if "source_sha256:" in path.read_text(encoding="utf-8")
    }
    assert discovered == set(EXPECTED_CUSTODY), (
        "every original-backed lift must be enumerated by the reconstructive custody test"
    )

    for relative_path, expectation in EXPECTED_CUSTODY.items():
        lifted = (LIFT_ROOT / relative_path).read_bytes()
        fields = _header_fields(lifted)
        original = _source_bytes(expectation.source_path)

        assert fields["source_repo"] == SOURCE_REPO_ROOT
        assert fields["source_repo_head"] == SOURCE_REPO_HEAD
        assert fields["lifted_at_head"] == LIFTED_AT_HEAD
        assert fields["source_path"] == expectation.source_path
        assert fields["source_sha256"] == hashlib.sha256(original).hexdigest()
        assert fields["source_sha256_scope"] == SOURCE_SHA256_SCOPE
        assert fields["adaptations"] == expectation.adaptation

        expected_body = _apply_declared_adaptation(original, expectation)
        assert _body_without_accounting_header(lifted) == expected_body, (
            f"{relative_path} contains an edit outside its declared adaptation"
        )


def test_lift_tree_denominator_keeps_local_initializer_explicit() -> None:
    all_lifted_python = {
        str(path.relative_to(LIFT_ROOT))
        for directory in (LIFT_ROOT / "lifted", LIFT_ROOT / "pose/lifted")
        for path in directory.glob("*.py")
    }
    assert len(all_lifted_python) == 13
    assert all_lifted_python - set(EXPECTED_CUSTODY) == {"pose/lifted/__init__.py"}
