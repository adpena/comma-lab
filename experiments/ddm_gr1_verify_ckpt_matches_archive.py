"""ddm_gr1 correctness gate: does the sb1 T3 checkpoint's quantized token grid
equal the pfs1 D1 archive dr7t codes? If yes, the fast MLX render path measures
the SHIPPED vehicle and my re-race is archive-faithful. $0-ish (one model load,
no scorer backward). Pointer 0.1910828242 [contest-CPU] UNMOVED; score_claim=false.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddm_r7_token_coder import decode_token_codes  # noqa: E402

CKPT = ("/Volumes/VertigoDataTier/pact/ddm_lv1_20260728/t3_long_burn_lotto_v2/"
        "checkpoints/stage_seg_trunk_tau_final.npz")
ARCHIVE = ("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/"
           "submissions/pfs1/archive.zip")


def main() -> int:
    # --- archive dr7t codes ---
    frame = zipfile.ZipFile(ARCHIVE).read("state/tokens.dr7t")
    arch_codes = np.asarray(decode_token_codes(frame), dtype=np.uint8)
    print(f"[archive] dr7t codes shape={arch_codes.shape} dtype={arch_codes.dtype} "
          f"min={arch_codes.min()} max={arch_codes.max()}", flush=True)

    # --- model quantized codes ---
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from experiments.train_tr1_partition_renderer_mlx import TR1Config, build_module

    z = np.load(CKPT, allow_pickle=False)
    meta = json.loads(bytes(z["meta::json"]).decode())
    cfgd = dict(meta["cfg"])
    cfgd.setdefault("token_init_mode", "zero")
    cfgd.setdefault("basin_handoff", "off")
    cfg = TR1Config(**{k: cfgd[k] for k in TR1Config.__dataclass_fields__ if k in cfgd})
    model = build_module(cfg)
    model.update(tree_unflatten(
        [(k[len("ema::"):], mx.array(z[k])) for k in z.files if k.startswith("ema::")]))
    mx.eval(model.parameters())
    L = cfg.token_quant_levels
    print(f"[model] levels={L} grid=({cfg.num_pairs},{cfg.grid_h},{cfg.grid_w},"
          f"{cfg.code_width})", flush=True)

    base = np.asarray(model.tokens_base, dtype=np.float32)   # (grid_h,grid_w,code_width)?
    delta = np.asarray(model.tokens_delta, dtype=np.float32)  # (num_pairs,...)
    print(f"[model] base shape={base.shape} delta shape={delta.shape}", flush=True)
    t_full = np.clip(delta + base[None], -1.0, 1.0)
    q = np.round((t_full + 1.0) * 0.5 * (L - 1)).astype(np.uint8)
    print(f"[model] quantized q shape={q.shape} min={q.min()} max={q.max()}", flush=True)

    if q.shape != arch_codes.shape:
        print(json.dumps({"MATCH": False, "reason": "shape",
                          "q_shape": list(q.shape),
                          "arch_shape": list(arch_codes.shape)}))
        return 1
    eq = int(np.count_nonzero(q == arch_codes))
    tot = int(q.size)
    frac = eq / tot
    print(json.dumps({"MATCH": bool(eq == tot), "equal": eq, "total": tot,
                      "frac_equal": round(frac, 6),
                      "max_abs_code_diff": int(np.abs(q.astype(np.int16)
                                                      - arch_codes.astype(np.int16)).max())},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
