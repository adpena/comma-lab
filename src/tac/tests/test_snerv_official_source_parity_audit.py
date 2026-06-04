# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.snerv_official_primitive_replay import (
    RECEIVER_RUNTIME_DECODE_SCHEMA,
)
from tac.analysis.snerv_official_source_parity_audit import (
    SCHEMA,
    build_snerv_official_mfu_hfr_tub_forward_parity_artifact,
    build_snerv_official_source_parity_audit,
    render_snerv_official_source_parity_markdown,
    summarize_snerv_official_source_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_snerv_official_source_audit_preserves_blocker_until_local_parity_proof(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_snerv_repo(tmp_path)

    report = build_snerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        generated_utc="20260603T000000Z",
    )

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["official_source_markers_present"] is True
    assert report["local_receiver_safe_adapter_present"] is True
    primitive = report["official_mfu_hfr_tub_primitive_replay_binding"]
    assert primitive["all_primitive_numeric_graph_replay_proven"] is True
    assert primitive["all_receiver_primitive_replay_proven"] is True
    assert primitive["all_primitive_numeric_source_fixture_replay_proven"] is True
    assert primitive["all_primitive_source_replay_proven"] is False
    assert primitive["full_stack_source_forward_replay_proven"] is False
    assert primitive["receiver_export_self_consistency_verified"] is True
    assert primitive["receiver_source_forward_replay_bound"] is False
    assert primitive["receiver_export_bound"] is True
    assert primitive["native_mlx_export_bound"] is True
    assert primitive["receiver_native_export_bound"] is True
    assert primitive["official_export_bound"] is False
    assert primitive["official_export_bound_semantics"] == (
        "requires_receiver_export_native_mlx_export_and_source_forward_replay"
    )
    runtime = report["official_receiver_runtime_decode_contract"]
    assert runtime["schema"] == RECEIVER_RUNTIME_DECODE_SCHEMA
    assert runtime["all_runtime_modules_import_safe"] is True
    assert runtime["all_numeric_source_replay_tests_hashed"] is True
    assert runtime["receiver_runtime_decode_proven"] is True
    assert runtime["receiver_export_self_consistency_verified"] is True
    assert runtime["receiver_source_forward_replay_bound"] is False
    assert "snerv_official_mfu_hfr_tub_source_forward_replay_missing" in (
        runtime["blockers"]
    )
    assert runtime["native_mlx_export_bound"] is True
    assert runtime["receiver_native_export_bound"] is True
    assert runtime["official_export_bound"] is False
    assert runtime["official_export_bound_semantics"] == (
        "requires_receiver_export_native_mlx_export_and_source_forward_replay"
    )
    assert (
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
        not in runtime["blockers"]
    )
    assert report["official_mfu_hfr_tub_parity_proven"] is False
    assert report["blockers"] == ["snerv_official_mfu_hfr_tub_parity_missing"]
    assert report["official_forward_parity_artifact_row"]["status"] == "missing"
    states = {row["component_id"]: row for row in report["component_state_rows"]}
    assert states["mfu"]["classification"] == (
        "source_forward_parity_falsified_receiver_safe_analogue_only"
    )
    assert states["hfr"]["classification"] == (
        "source_forward_parity_falsified_receiver_safe_analogue_only"
    )
    assert states["tub"]["classification"] == (
        "official_haar_temporal_lowpass_primitive_proven_full_tub_falsified"
    )
    assert "snerv_mfu_source_forward_parity_falsified_receiver_safe_analogue_only" in (
        states["mfu"]["blockers"]
    )
    assert states["mfu"]["local_source_forward_markers_present"] is False
    assert states["mfu"]["primitive_parity_markers_present"] is False
    assert states["tub"]["primitive_parity_markers_present"] is True
    assert all(row["source_forward_parity_proven"] is False for row in states.values())
    assert all(row["status"] == "present" for row in report["official_file_rows"])
    assert all(row["all_markers_present"] for row in report["official_marker_group_rows"])

    summary = summarize_snerv_official_source_audit(report)
    assert summary["official_source_markers_present"] is True
    assert summary["official_mfu_hfr_tub_receiver_primitives_proven"] is True
    assert summary["official_mfu_hfr_tub_numeric_graph_replay_proven"] is True
    assert summary["official_mfu_hfr_tub_primitives_proven"] is True
    assert summary["official_receiver_runtime_decode_proven"] is True
    assert summary["official_receiver_source_forward_replay_bound"] is False
    assert "snerv_official_mfu_hfr_tub_source_forward_replay_missing" in (
        summary["official_receiver_runtime_decode_blockers"]
    )
    assert summary["full_stack_source_forward_replay_proven"] is False
    assert summary["official_mfu_hfr_tub_parity_proven"] is False
    assert {row["classification"] for row in summary["component_states"]} == {
        "source_forward_parity_falsified_receiver_safe_analogue_only",
        "official_haar_temporal_lowpass_primitive_proven_full_tub_falsified",
    }
    assert "snerv_official_mfu_hfr_tub_parity_missing" in summary["blockers"]

    md = render_snerv_official_source_parity_markdown(report)
    assert "SNeRV Official Source-Parity Audit" in md
    assert "official MFU/HFR/TUB parity proven: `False`" in md
    assert "official MFU/HFR/TUB receiver primitive replay proven: `True`" in md
    assert "official MFU/HFR/TUB numeric graph replay proven: `True`" in md
    assert "official receiver runtime decode proven: `True`" in md
    assert "official receiver source-forward replay bound: `False`" in md
    assert "official MFU/HFR/TUB parity falsified: `False`" in md
    assert (
        "| `tub` | "
        "`official_haar_temporal_lowpass_primitive_proven_full_tub_falsified` |"
    ) in md


def test_snerv_official_source_audit_fails_closed_on_missing_official_markers(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_snerv_repo(tmp_path)
    (official / "model/snerv_t.py").write_text(
        "class SNeRV_T:\n    pass\n",
        encoding="utf-8",
    )

    report = build_snerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        generated_utc="20260603T000000Z",
    )

    assert report["official_source_markers_present"] is False
    assert "snerv_official_source_marker_missing:official_tub_temporal_extension" in (report["blockers"])
    assert "snerv_official_mfu_hfr_tub_parity_missing" in report["blockers"]


def test_snerv_official_source_audit_rejects_marker_only_parity(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_snerv_repo(tmp_path)
    local = _write_marker_only_local_snerv_repo(tmp_path)

    report = build_snerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=local,
        generated_utc="20260603T000000Z",
    )

    assert report["official_source_markers_present"] is True
    assert report["local_receiver_safe_adapter_present"] is True
    assert report["local_official_parity_marker_row"]["all_markers_present"] is True
    assert report["official_forward_parity_artifact_row"]["status"] == "missing"
    assert report["official_mfu_hfr_tub_parity_proven"] is False
    assert report["blockers"] == ["snerv_official_mfu_hfr_tub_parity_missing"]
    assert all(
        "snerv_official_forward_parity_artifact_missing_or_failed" in row["blockers"]
        for row in report["component_state_rows"]
    )


def test_snerv_official_forward_parity_artifact_round_trips_falsification(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_snerv_repo(tmp_path)
    artifact = build_snerv_official_mfu_hfr_tub_forward_parity_artifact(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        generated_utc="20260603T000000Z",
    )
    assert artifact["score_claim"] is False
    primitive = artifact["official_mfu_hfr_tub_primitive_replay_binding"]
    assert primitive["all_primitive_numeric_source_fixture_replay_proven"] is True
    assert primitive["all_primitive_source_replay_proven"] is False
    assert primitive["full_stack_source_forward_replay_proven"] is False
    assert artifact["official_mfu_hfr_tub_forward_parity_passed"] is False
    assert artifact["official_mfu_hfr_tub_forward_parity_falsified"] is True
    artifact_states = {row["component_id"]: row for row in artifact["component_rows"]}
    assert artifact_states["mfu"]["source_forward_parity_falsified"] is True
    assert artifact_states["mfu"]["primitive_parity_markers_present"] is False
    assert artifact_states["hfr"]["source_forward_parity_falsified"] is True
    assert artifact_states["hfr"]["primitive_parity_markers_present"] is False
    assert artifact_states["tub"]["primitive_parity_markers_present"] is True

    artifact_path = tmp_path / "forward_parity.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    report = build_snerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        official_forward_parity_artifact_path=artifact_path,
        generated_utc="20260603T000000Z",
    )

    assert report["official_mfu_hfr_tub_parity_proven"] is False
    artifact_row = report["official_forward_parity_artifact_row"]
    assert artifact_row["status"] == "present"
    assert artifact_row["parity_passed"] is False
    assert artifact_row["parity_falsified"] is True
    states = {row["component_id"]: row for row in report["component_state_rows"]}
    assert states["mfu"]["source_forward_parity_falsified"] is True
    assert "snerv_official_forward_parity_artifact_falsifies_parity" in (
        states["mfu"]["blockers"]
    )


def test_snerv_official_forward_parity_artifact_rejects_boolean_only_pass(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_snerv_repo(tmp_path)
    local = _write_marker_only_local_snerv_repo(tmp_path)
    artifact_path = tmp_path / "forged_pass.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "snerv_official_mfu_hfr_tub_forward_parity.v1",
                "official_mfu_hfr_tub_forward_parity_passed": True,
                "component_rows": [
                    {
                        "component_id": "mfu",
                        "source_forward_parity_proven": True,
                    },
                    {
                        "component_id": "hfr",
                        "source_forward_parity_proven": True,
                    },
                    {
                        "component_id": "tub",
                        "source_forward_parity_proven": True,
                    },
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_snerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=local,
        official_forward_parity_artifact_path=artifact_path,
        generated_utc="20260603T000000Z",
    )

    artifact_row = report["official_forward_parity_artifact_row"]
    assert artifact_row["parity_passed"] is False
    assert "snerv_official_forward_parity_weight_manifest_missing" in artifact_row[
        "blockers"
    ]
    assert "snerv_official_forward_parity_source_replay_missing" in artifact_row[
        "blockers"
    ]
    assert "snerv_official_forward_parity_receiver_runtime_decode_missing" in (
        artifact_row["blockers"]
    )
    assert any(
        blocker.startswith("numeric_max_abs_error_missing:mfu")
        for blocker in artifact_row["blockers"]
    )
    assert report["official_mfu_hfr_tub_parity_proven"] is False
    assert "snerv_official_mfu_hfr_tub_parity_missing" in report["blockers"]


def test_snerv_official_forward_parity_artifact_accepts_numeric_replay_evidence(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_snerv_repo(tmp_path)
    local = _write_marker_only_local_snerv_repo(tmp_path)
    artifact_path = tmp_path / "numeric_pass.json"
    component_rows = [
        _numeric_component_row("mfu"),
        _numeric_component_row("hfr"),
        _numeric_component_row("tub"),
    ]
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "snerv_official_mfu_hfr_tub_forward_parity.v1",
                "official_weight_manifest": {
                    "state_dict_sha256": "1" * 64,
                    "state_dict_key_count": 9,
                },
                "source_forward_replay": {
                    "backend": "torch_vs_numpy",
                    "input_bundle_sha256": "2" * 64,
                },
                "receiver_runtime_decode": _receiver_runtime_decode_contract(),
                "official_mfu_hfr_tub_forward_parity_passed": True,
                "official_mfu_hfr_tub_forward_parity_falsified": False,
                "component_rows": component_rows,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_snerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=local,
        official_forward_parity_artifact_path=artifact_path,
        generated_utc="20260603T000000Z",
    )

    artifact_row = report["official_forward_parity_artifact_row"]
    assert artifact_row["parity_passed"] is True
    assert artifact_row["blockers"] == []
    assert report["official_mfu_hfr_tub_parity_proven"] is True
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_snerv_official_forward_parity_artifact_rejects_partial_receiver_decode(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_snerv_repo(tmp_path)
    local = _write_marker_only_local_snerv_repo(tmp_path)
    receiver_decode = _receiver_runtime_decode_contract()
    receiver_rows = list(receiver_decode["component_rows"])
    receiver_rows[0] = dict(receiver_rows[0])
    receiver_rows[0].pop("runtime_output_sha256")
    receiver_decode["component_rows"] = receiver_rows
    artifact_path = tmp_path / "partial_receiver_decode.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "snerv_official_mfu_hfr_tub_forward_parity.v1",
                "official_weight_manifest": {
                    "state_dict_sha256": "1" * 64,
                    "state_dict_key_count": 9,
                },
                "source_forward_replay": {
                    "backend": "torch_vs_numpy",
                    "input_bundle_sha256": "2" * 64,
                },
                "receiver_runtime_decode": receiver_decode,
                "official_mfu_hfr_tub_forward_parity_passed": True,
                "official_mfu_hfr_tub_forward_parity_falsified": False,
                "component_rows": [
                    _numeric_component_row("mfu"),
                    _numeric_component_row("hfr"),
                    _numeric_component_row("tub"),
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_snerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=local,
        official_forward_parity_artifact_path=artifact_path,
        generated_utc="20260603T000000Z",
    )

    artifact_row = report["official_forward_parity_artifact_row"]
    assert artifact_row["parity_passed"] is False
    assert "runtime_output_sha256_missing:mfu" in artifact_row["blockers"]
    assert report["official_mfu_hfr_tub_parity_proven"] is False


def _numeric_component_row(component_id: str) -> dict[str, object]:
    return {
        "component_id": component_id,
        "source_forward_parity_proven": True,
        "max_abs_error": 0.0,
        "tolerance": 1.0e-6,
        "input_sha256": "3" * 64,
        "official_output_sha256": "4" * 64,
        "portable_output_sha256": "4" * 64,
        "official_weight_sha256": "5" * 64,
    }


def _receiver_runtime_decode_contract() -> dict[str, object]:
    return {
        "schema": RECEIVER_RUNTIME_DECODE_SCHEMA,
        "receiver_runtime_decode_proven": True,
        "receiver_export_self_consistency_verified": True,
        "receiver_source_forward_replay_bound": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "component_rows": [
            _receiver_runtime_decode_row("mfu"),
            _receiver_runtime_decode_row("hfr"),
            _receiver_runtime_decode_row("tub"),
        ],
    }


def _receiver_runtime_decode_row(component_id: str) -> dict[str, object]:
    return {
        "component_id": component_id,
        "receiver_runtime_decode_proven": True,
        "receiver_export_self_consistency_verified": True,
        "receiver_source_forward_replay_bound": False,
        "runtime_module_import_safe": True,
        "runtime_module_sha256": "6" * 64,
        "numeric_test_sha256": "7" * 64,
        "archive_section_sha256": "8" * 64,
        "decoded_input_sha256": "9" * 64,
        "runtime_output_sha256": "a" * 64,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _write_minimal_official_snerv_repo(tmp_path: Path) -> Path:
    root = tmp_path / "SNeRV"
    (root / "model").mkdir(parents=True)
    (root / "model/snerv.py").write_text(
        """
from pytorch_wavelets import DWT, IDWT
import torch
import torch.nn as nn

class SNeRV:
    def forward(self, input):
        yl, _ = DWT(J=1, wave='haar', mode='periodization').cuda()(input)
        yl_norm = torch.as_tensor([yl.min(), yl.max()])
        idwt = IDWT(wave='haar', mode='periodization').cuda()
        decoder_layer2 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')
        decoder_layer3 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')
        decoder_layer4 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')
        upsample_5 = nn.ConvTranspose2d(ngf_list[-3], ngf_list[-3], args.dec_strds[-2], args.dec_strds[-2], 0)
        decoder_layer5 = RB(in_channels=ngf_list[-3]+ngf_list[-2], out_channels=ngf_list[-2], num_blocks=args.num_blocks)
        upsample_6 = nn.ConvTranspose2d(ngf_list[-2], ngf_list[-2], args.dec_strds[-1], args.dec_strds[-1], 0)
        decoder_layer6 = RB(in_channels=ngf_list[-2]+new_ngf, out_channels=new_ngf, num_blocks=args.num_blocks)
        up1 = self.decoder[self.decoder_len+3](embed_list[-3])
        unet1 = self.decoder[self.decoder_len+4](torch.cat([up1, embed_list[-2]], dim=1))
        unet1_up = self.decoder[self.decoder_len+5](unet1)
        pyr_out = self.decoder[self.decoder_len+6](torch.cat([unet1_up, embed_list[-1]], dim=1))
        self.up = nn.ConvTranspose2d(1, 1, 2, 2, 0)
        self.rb = RB()
        HF_in = pyr_out
        lh_out = self.decoder[self.decoder_len](HF_in)
        hl_out = self.decoder[self.decoder_len+1](HF_in)
        hh_out = self.decoder[self.decoder_len+2](HF_in)
        yh_out = torch.stack([lh_out, hl_out, hh_out], dim=2)
        return idwt([yl, [yh_out]])
""",
        encoding="utf-8",
    )
    (root / "model/snerv_t.py").write_text(
        """
from pytorch_wavelets import DWT1D

class SNeRV_T:
    def forward(self, input, input_p, input_n):
        embed_lv_p, embed_hv_p = DWT1D(J=1, wave='haar', mode='periodization').cuda()(input_p)
        embed_lv_n, embed_hv_n = DWT1D(J=1, wave='haar', mode='periodization').cuda()(input_n)
        embed_hv_p = self.encoder[1]((embed_lv_p.permute(2,1,0).reshape(1,c,h,w))/2)
        embed_hv_n = self.encoder[2]((embed_lv_n.permute(2,1,0).reshape(1,c,h,w))/2)
        output_2 = self.decoder[self.decoder_len-1](embed_hv_p)
        output = layer(output, output_2)
        temp_emb_layer = UpsampleBlock(input)
        return embed_hv_p, embed_hv_n, temp_emb_layer
""",
        encoding="utf-8",
    )
    (root / "model/layers.py").write_text(
        "class UpsampleBlock: pass\nclass RB: pass\n",
        encoding="utf-8",
    )
    (root / "train_snerv.py").write_text(
        """
parser.add_argument('--modelsize')
parser.add_argument('--fc_dim')
parser.add_argument('--quant_model_bit')
parser.add_argument('--quant_embed_bit')
parser.add_argument('--quant_embed2_bit')
parser.add_argument('--quant_axis')
embed_param = 1
decoder_size = 2
args.fc_dim = int(np.roots([a, b, c - decoder_size]).max())
""",
        encoding="utf-8",
    )
    return root


def _write_marker_only_local_snerv_repo(tmp_path: Path) -> Path:
    root = tmp_path / "local"
    source_root = root / "src/tac/substrates/snerv_inverse_steg_carrier"
    source_root.mkdir(parents=True)
    (source_root / "carrier.py").write_text(
        """
MultiResolutionFusionUnit = object
HighFrequencyRestorer = object
SnervTemporalExtension = object
SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF = True
SNERV_OFFICIAL_TEMPORAL_HAAR_DWT1D_PROOF = True
SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF = True
""",
        encoding="utf-8",
    )
    return root
