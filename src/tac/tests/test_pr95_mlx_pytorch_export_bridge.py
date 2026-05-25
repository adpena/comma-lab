from __future__ import annotations

from pathlib import Path

from tac.local_acceleration.pr95_hnerv_mlx import (
    HNeRVSyntheticTrainingBundleMLX,
    pytorch_state_dict_from_mlx,
    write_pr95_public_archive_zip,
)
from tools.export_pr95_mlx_to_pytorch_state_dict import (
    DEFAULT_SOURCE_MODEL,
    PR95_MLX_EXPORT_BRIDGE_SCHEMA,
    export_pr95_public_archive_to_pytorch_state_dict,
)


def test_pr95_mlx_pytorch_export_bridge_smoke_false_authority(
    tmp_path: Path,
) -> None:
    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=8,
        seed=17,
    )
    archive_zip = tmp_path / "archive.zip"
    write_pr95_public_archive_zip(
        pytorch_state_dict_from_mlx(bundle.decoder),
        bundle.latents,
        meta={"latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]},
        output_zip_path=archive_zip,
    )
    output_pt = tmp_path / "state.pt"
    report_path = tmp_path / "report.json"

    report = export_pr95_public_archive_to_pytorch_state_dict(
        archive_zip=archive_zip,
        output_pytorch_state_dict=output_pt,
        source_model=DEFAULT_SOURCE_MODEL,
        report_out=report_path,
        sample_indices=[0],
    )

    assert report["schema_version"] == PR95_MLX_EXPORT_BRIDGE_SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["archive_packet"]["member_name"] == "0.bin"
    assert report["pytorch_export_forward_parity_established"] is True
    assert output_pt.is_file()
    assert report_path.is_file()
    assert report["pt_path"] == str(output_pt)
    parity = report["pytorch_export_forward_parity"]["forward_parity"]["parity"]
    assert parity["max_abs"] <= 2e-3
