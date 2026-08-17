"""Does optimize_sparse_evaluator change VALUES or only speed?

Decides whether ddm_rc4's amplification/pose drop sets (built without it) match
the drop sets the shipping decoder path would build.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

GEN = Path("/Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/generations/hv1_ep0634_s1p25_c1p0_brotli_q10")
TOK = Path("/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8")
STORE = Path("/Volumes/APDataStore/pact/ddm_rc4_rung4_token_drop_20260816")
sys.path.insert(0, str(GEN))
sys.path.insert(0, str(GEN / "cpr1"))
import torch

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
from runtime.hpac_inference import optimize_sparse_evaluator
from runtime.ihs2 import materialize_ihs1
from runtime.residual_archive import _boundary_buckets, _probability_table, _sparse_class, read_residual_archive

spec = importlib.util.spec_from_file_location("_rc4_r", GEN / "cpr1" / "inflate.py")
rt = importlib.util.module_from_spec(spec)
sys.modules["_rc4_r"] = rt
spec.loader.exec_module(rt)

parts = read_residual_archive(GEN / "archive.zip")
tokens = np.fromfile(TOK, dtype=np.uint8).reshape(rt.N, rt.EVAL_H, rt.EVAL_W)
dev = torch.device("cpu")

def run(frame: int, optimize: bool):
    model = rt.load_hpac(materialize_ihs1(parts.hpac_blob, rt), dev)
    sparse = _sparse_class(GEN / "cpr1")(model, rt.EVAL_H, rt.EVAL_W)
    plans = [(torch.from_numpy(np.flatnonzero(m.detach().cpu().numpy().reshape(-1))),
              np.flatnonzero(m.detach().cpu().numpy().reshape(-1))) for m in rt.group_masks(dev)]
    with torch.inference_mode():
        if optimize:
            optimize_sparse_evaluator(sparse)
        idx = torch.tensor([frame], dtype=torch.long)
        prev = tokens[frame - 1] if frame else np.zeros_like(tokens[0])
        previous = torch.from_numpy(prev.astype(np.int64))[None]
        ctx = model.prepare_frame_context(idx, previous)
        bnd = _boundary_buckets(prev).reshape(-1) if frame else np.full(rt.EVAL_H * rt.EVAL_W, 4, np.uint8)
        truth = tokens[frame].reshape(-1).astype(np.int64)
        cur = torch.zeros((1, rt.EVAL_H, rt.EVAL_W), dtype=torch.long)
        pm = np.zeros(truth.size)
        ar = np.zeros(truth.size, dtype=np.int64)
        for g, (dp, fp) in enumerate(plans):
            bl = sparse.selected_logits(cur, ctx, g).cpu().numpy()
            pr = bl.argmax(axis=1).astype(np.int64)
            corr = bl + parts.table.values[bnd[fp].astype(np.int64) * 5 + pr]
            prob = _probability_table(corr, rt.HPAC_LOGIT_PRECISION).astype(np.float64)
            a = prob.argmax(axis=1)
            pm[fp] = prob[np.arange(a.size), a]
            ar[fp] = a
            cur.reshape(-1)[dp] = torch.from_numpy(truth[fp])
    return pm, ar

out = {}
for frame in (0, 137, 411):
    pm_a, ar_a = run(frame, False)
    pm_b, ar_b = run(frame, True)
    out[str(frame)] = {
        "argmax_identical": bool((ar_a == ar_b).all()),
        "argmax_mismatches": int((ar_a != ar_b).sum()),
        "p_max_max_abs_diff": float(np.abs(pm_a - pm_b).max()),
        "drop_set_identical_u5": bool(((pm_a >= 1 - 2**-5.0) == (pm_b >= 1 - 2**-5.0)).all()),
        "drop_set_identical_u7": bool(((pm_a >= 1 - 2**-7.0) == (pm_b >= 1 - 2**-7.0)).all()),
        "drop_set_diff_u7": int((((pm_a >= 1 - 2**-7.0) != (pm_b >= 1 - 2**-7.0))).sum()),
    }
    print(frame, json.dumps(out[str(frame)]), flush=True)
(STORE / "OPTIMIZE_SPARSE_CONTROL.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print("wrote OPTIMIZE_SPARSE_CONTROL.json")
