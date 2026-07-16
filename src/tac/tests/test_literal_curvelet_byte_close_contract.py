from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from tac.boundary_math.localized_basis_frames import (
    BasisProgramConfig,
    basis_program_taper_config_sha256,
    literal_basis_program_config,
)
from tools import levelset_byte_close_and_eval as byte_close


def _write_literal_checkpoint(tmp_path, *, program: BasisProgramConfig, program_sha: str | None = None):
    path = tmp_path / "literal.npz"
    payload = {
        "code": np.zeros((2, 1), np.float32),
        "in_proj.weight": np.ones((3, 80), np.float32),
        "in_proj.bias": np.zeros(3, np.float32),
        "out_sdf.weight": np.ones((5, 3), np.float32),
        "out_sdf.bias": np.zeros(5, np.float32),
        "out_tex.weight": np.ones((3, 3), np.float32),
        "out_tex.bias": np.zeros(3, np.float32),
        "palette": np.zeros((5, 3), np.float32),
        "__cfg_basis": np.asarray("literal_polar_curvelet"),
        "__cfg_basis_program_json": np.asarray(
            json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"))
        ),
        "__cfg_basis_program_sha256": np.asarray(
            program.canonical_sha256() if program_sha is None else program_sha
        ),
        "__cfg_basis_taper_folded": np.asarray(0),
        "__basis_taper_unfolded": np.ones(80, np.float32),
        "__render_hw": np.asarray([3, 3]),
    }
    np.savez(path, **payload)
    return path


def test_literal_checkpoint_program_and_unfolded_taper_are_strictly_loaded(tmp_path):
    taper_hash = basis_program_taper_config_sha256(strength=1.0, scale=0.0, floor=0.05)
    program = literal_basis_program_config(
        taper_enabled=True, taper_train_config_sha256=taper_hash
    )
    path = _write_literal_checkpoint(tmp_path, program=program)
    params, cfg = byte_close._load_levelset_ckpt(tmp_path, path.name)
    assert params["in_proj.weight"].shape == (3, 80)
    assert cfg["basis_program"] == program
    np.testing.assert_array_equal(cfg["basis_taper_unfolded"], np.ones(80, np.float32))


def test_literal_checkpoint_refuses_semantic_hash_drift(tmp_path):
    program = literal_basis_program_config()
    path = _write_literal_checkpoint(tmp_path, program=program, program_sha="0" * 64)
    with pytest.raises(ValueError, match="canonical SHA-256 mismatch"):
        byte_close._load_levelset_ckpt(tmp_path, path.name)


def test_literal_generated_receiver_is_self_contained_and_unknown_hashes_refuse():
    source = byte_close._inflate_source_for_manifest(
        {"basis_family": "literal_polar_curvelet"}
    )
    compile(source, "<literal-curvelet-inflate>", "exec")
    module = types.ModuleType("literal_receiver_test")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "<literal-curvelet-inflate>", "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    namespace = module.__dict__
    program = literal_basis_program_config()
    features = namespace["_basis_feats"](
        namespace["_coords"](3, 3),
        {
            "basis_family": "literal_polar_curvelet",
            "basis_program": program.to_dict(),
            "basis_program_sha256": program.canonical_sha256(),
        },
    )
    assert features.shape == (9, 80)
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        namespace["_basis_feats"](
            namespace["_coords"](3, 3),
            {
                "basis_family": "literal_polar_curvelet",
                "basis_program": program.to_dict(),
                "basis_program_sha256": "0" * 64,
            },
        )
