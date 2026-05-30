# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tac.local_acceleration.mlx_canonicalization_audit import (
    WAIVER_TOKEN,
    build_mlx_canonicalization_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_audit_flags_direct_mlx_substrate_without_canonical_route(
    tmp_path: Path,
) -> None:
    substrate = tmp_path / "src" / "tac" / "substrates" / "demo" / "mlx_renderer.py"
    substrate.parent.mkdir(parents=True)
    substrate.write_text(
        "import mlx.core as mx\n\n"
        "def forward(x):\n"
        "    mx.eval(x)\n"
        "    return x\n",
        encoding="utf-8",
    )

    audit = build_mlx_canonicalization_audit(
        repo_root=tmp_path,
        scan_roots=["src/tac/substrates"],
    )

    assert audit["mlx_canonicalization_ready"] is False
    assert audit["review_required_count"] == 1
    row = audit["review_required_rows"][0]
    assert row["path"] == "src/tac/substrates/demo/mlx_renderer.py"
    assert row["status"] == "needs_canonical_helper_or_unique_method_waiver"
    assert "mx.eval" in row["primitive_tokens"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_audit_accepts_canonical_route_or_unique_method_waiver(
    tmp_path: Path,
) -> None:
    routed = tmp_path / "tools" / "routed_mlx.py"
    routed.parent.mkdir(parents=True)
    routed.write_text(
        "import mlx.core as mx\n"
        "from tac.local_acceleration.pr95_hnerv_mlx import pixel_shuffle_2x_nhwc\n",
        encoding="utf-8",
    )
    waived = tmp_path / "tools" / "unique_mlx.py"
    waived.write_text(
        f"import mlx.core as mx  # {WAIVER_TOKEN}EXACT_TEST: fixture\n",
        encoding="utf-8",
    )

    audit = build_mlx_canonicalization_audit(
        repo_root=tmp_path,
        scan_roots=["tools"],
    )

    rows = {row["path"]: row for row in audit["rows"]}
    assert audit["mlx_canonicalization_ready"] is True
    assert rows["tools/routed_mlx.py"]["status"] == "routes_canonical_helper"
    assert rows["tools/unique_mlx.py"]["status"] == "unique_method_waived"


def test_audit_treats_canonical_source_root_as_routed(tmp_path: Path) -> None:
    helper = tmp_path / "src" / "tac" / "local_acceleration" / "new_helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "import mlx.core as mx\n\n"
        "def sync(x):\n"
        "    mx.eval(x)\n",
        encoding="utf-8",
    )

    audit = build_mlx_canonicalization_audit(
        repo_root=tmp_path,
        scan_roots=["src/tac/local_acceleration"],
    )

    assert audit["mlx_canonicalization_ready"] is True
    assert audit["rows"][0]["status"] == "canonical_source_root"


def test_audit_mlx_canonicalization_cli_writes_outputs(tmp_path: Path) -> None:
    target = tmp_path / "src" / "tac" / "substrates" / "demo.py"
    target.parent.mkdir(parents=True)
    target.write_text("import mlx.nn as nn\n", encoding="utf-8")
    json_out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "audit_mlx_canonicalization.py"),
            "--scan-root",
            str(target),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert '"mlx_canonicalization_ready": false' in json_out.read_text(
        encoding="utf-8"
    )
    assert "Review Required" in md_out.read_text(encoding="utf-8")
