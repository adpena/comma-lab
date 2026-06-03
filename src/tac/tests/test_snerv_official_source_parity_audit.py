# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from tac.analysis.snerv_official_source_parity_audit import (
    SCHEMA,
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
    assert report["official_mfu_hfr_tub_parity_proven"] is False
    assert report["blockers"] == ["snerv_official_mfu_hfr_tub_parity_missing"]
    assert all(row["status"] == "present" for row in report["official_file_rows"])
    assert all(row["all_markers_present"] for row in report["official_marker_group_rows"])

    summary = summarize_snerv_official_source_audit(report)
    assert summary["official_source_markers_present"] is True
    assert summary["official_mfu_hfr_tub_parity_proven"] is False
    assert "snerv_official_mfu_hfr_tub_parity_missing" in summary["blockers"]

    md = render_snerv_official_source_parity_markdown(report)
    assert "SNeRV Official Source-Parity Audit" in md
    assert "official MFU/HFR/TUB parity proven: `False`" in md


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
