"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``127:129: invalid decimal literal``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``hpac.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/public_pr86_intake_20260504_codex/training/hpac.py'
__recovery_spec__ = 'hpac.recovery_spec.json'
__recovery_ast_error__ = '127:129: invalid decimal literal'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: hpac.cpython-312.pyc (Python 3.12)

'''HPAC: hierarchical parallel autoregressive token compressor.

Single-file consolidation of the HPAC entropy model, arithmetic codec, and
training loop.

  * Architecture: patch+group autoregressive over 32x32 patches with stride
    delta=2 scan. Each frame decoded in 94 sequential group-steps (vs
    196,608 for raster AR). SCN-quantized layers learn per-channel bit
    budgets jointly with the model.
  * Codec: group-by-group arithmetic coding via constriction.
  * Training: residual-token objective. Compress
        res[i] = (tok[i] - tok[i-1]) mod 5      (i > 0)
        res[0] = tok[0]                          (no prev frame)
    The residual alphabet is still 5 classes but heavily skewed toward 0
    (most pixels unchanged frame-to-frame on driving video). The model
    still sees raw prev_tokens for spatial-temporal context via conv_past,
    but predicts the RESIDUAL at the current frame.
    Decoder reconstructs:  tok[i] = (res[i] + tok[i-1]) mod 5.

Usage:
    python hpac.py train --save hpac.pt
'''
import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import nn

functional
import constriction = import torch.nn.functional, nn
from safetensors.torch import load_file
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import preload_rgb_pairs_av, SEGNET_IN_W, SEGNET_IN_H
from modules import SegNet, segnet_sd_path
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 600
NUM_CLASSES = 5
SAVE_DIR = Path(__file__).resolve().parent.parent / 'training_workspace'
B_INIT = 4
E_INIT = -3
B_MIN = 0.5
B_MAX = 8

def _scn_quantize(w, b, e):
    b_clip = b.clamp(B_MIN, B_MAX)
    shape = [
        1] * w.ndim
    shape[0] = -1
    bv = b_clip.view(shape)
    ev = e.view(shape)
    scale = torch.pow(2, ev)
    max_q = torch.pow(2, bv - 1) - 1
    min_q = -torch.pow(2, bv - 1)
    q = torch.clamp(w / scale, min_q, max_q)
    q_round = q + (q.round() - q).detach()
    return q_round * scale


class SCNConv2d(nn.Module):
    pass
# WARNING: Decompyle incomplete


class SCNLinear(nn.Module):
    pass
# WARNING: Decompyle incomplete


def patch_group_mask(k, dilation, delta, type_):
    '''Return [k, k] bool mask. Center is at index (k-1)//2.

    For kernel offset (dr_idx - center)*dilation, (dc_idx - center)*dilation,
    the input position relative to output is offset (dr*dil, dc*dil).
    Group difference is dc*dil + delta*(dr*dil) = dil * (dc + delta*dr).

    Type-A: mask = 1 iff dc + delta*dr < 0
    Type-B: mask = 1 iff dc + delta*dr <= 0
    '''
    mask = torch.zeros(k, k, dtype = torch.float32)
    center = (k - 1) // 2
    for dr_idx in range(k):
        for dc_idx in range(k):
            dr = dr_idx - center
            dc = dc_idx - center
            val = dc + delta * dr
            if type_ == 'A':
                if not val < 0:
                    continue
                mask[(dr_idx, dc_idx)] = 1
                continue
            if type_ == 'B':
                if not val <= 0:
                    continue
                mask[(dr_idx, dc_idx)] = 1
                continue
            raise ValueError(type_)
    return mask


class SCNMaskedConv2dPG(nn.Module):
    pass
# WARNING: Decompyle incomplete


class ChannelNorm2d(nn.Module):
    pass
# WARNING: Decompyle incomplete


class CausalSPM(nn.Module):
    pass
# WARNING: Decompyle incomplete


class HPACMini(nn.Module):
    pass
# WARNING: Decompyle incomplete

causality_check = (lambda gen, H, W, n_classes, P, delta = (64, 64, 5, 32, 2): dev = next(gen.parameters()).devicegen.eval()B = 1H = PW = Ptokens = torch.zeros(B, H, W, dtype = torch.long, device = dev)prev = torch.zeros_like(tokens)idx = torch.zeros(B, dtype = torch.long, device = dev)base = gen(tokens, idx, prev).clone()c0 = P // 2r0 = P // 2s0 = c0 + delta * r0tokens2 = tokens.clone()tokens2[(0, r0, c0)] = n_classes - 1out = gen(tokens2, idx, prev)diff = (out - base).abs()diff_per_pos = diff.amax(dim = 1)[0]changed = diff_per_pos > 1e-05rs = torch.arange(P, device = dev).view(P, 1).expand(P, P)cs = torch.arange(P, device = dev).view(1, P).expand(P, P)s_grid = cs + delta * rsleaky_positions = changed & (s_grid < s0)n_leaks = int(leaky_positions.sum().item())if n_leaks == 0:
print(f'''[hpac causality] OK (perturbed s={s0} at ({r0},{c0}); all changes at s>=s0)''')Trueidxs = torch.nonzero(leaky_positions, as_tuple = False)[:5]print(f'''[hpac causality] LEAK: {n_leaks} positions changed at s<{s0}: {idxs.tolist()}''')False)()

def _patch_group_grid(P, delta, device):
    '''[P, P] tensor of group indices s = c + delta * r.'''
    rs = torch.arange(P, device = device).view(P, 1).expand(P, P)
    cs = torch.arange(P, device = device).view(1, P).expand(P, P)
    return cs + delta * rs


def _full_mask_for_group(s_grid, s, NRp, NCp):
    '''[H, W] bool mask: True at every position whose intra-patch group == s.'''
    P = s_grid.shape[0]
    mask_p = s_grid == s
    full = mask_p.unsqueeze(0).unsqueeze(0).expand(NRp, NCp, P, P)
    return full.permute(0, 2, 1, 3).reshape(NRp * P, NCp * P)

encode_frame = (lambda gen, gt_tokens, idx, prev_tokens, encoder, P, delta, prob_eps = (32, 2, 1e-07): dev = gt_tokens.device(H, W) = gt_tokens.shape[-2:]NCp = W // PNRp = H // Ps_grid = _patch_group_grid(P, delta, dev)n_groups = int((1 + delta) * P - delta)current = torch.zeros_like(gt_tokens)n_total = 0for s in range(n_groups):
full_mask = _full_mask_for_group(s_grid, s, NRp, NCp)n_pos = int(full_mask.sum().item())if n_pos == 0:
continuelogits = gen(current, idx, prev_tokens)probs = F.softmax(logits.float(), dim = 1)probs_at_s = probs[0][(:, full_mask)].permute(1, 0).contiguous()gt_at_s = gt_tokens[0][full_mask].cpu().numpy().astype(np.int32)probs_np = probs_at_s.cpu().numpy().astype(np.float64)probs_np = np.clip(probs_np, prob_eps, 1)probs_np = probs_np / probs_np.sum(axis = 1, keepdims = True)for i in range(n_pos):
cat = constriction.stream.model.Categorical(probabilities = probs_np[i], perfect = False)encoder.encode(int(gt_at_s[i]), cat)current[(0, full_mask)] = gt_tokens[(0, full_mask)]n_total += n_posn_total)()
decode_frame = (lambda gen, decoder, idx, prev_tokens, H, W, P, delta, prob_eps = (32, 2, 1e-07): dev = prev_tokens.deviceNCp = W // PNRp = H // Ps_grid = _patch_group_grid(P, delta, dev)n_groups = int((1 + delta) * P - delta)current = torch.zeros((1, H, W), dtype = torch.long, device = dev)for s in range(n_groups):
full_mask = _full_mask_for_group(s_grid, s, NRp, NCp)n_pos = int(full_mask.sum().item())if n_pos == 0:
continuelogits = gen(current, idx, prev_tokens)probs = F.softmax(logits.float(), dim = 1)probs_at_s = probs[0][(:, full_mask)].permute(1, 0).contiguous()probs_np = probs_at_s.cpu().numpy().astype(np.float64)probs_np = np.clip(probs_np, prob_eps, 1)probs_np = probs_np / probs_np.sum(axis = 1, keepdims = True)decoded = np.empty(n_pos, dtype = np.int64)for i in range(n_pos):
cat = constriction.stream.model.Categorical(probabilities = probs_np[i], perfect = False)decoded[i] = decoder.decode(cat)current[(0, full_mask)] = torch.from_numpy(decoded).to(dev)current)()

def compute_residuals(gt_tokens):
    '''gt_tokens: [N, H, W] long. Returns gt_residuals same shape.
    res[0] = tok[0]; res[i] = (tok[i] - tok[i-1]) % NUM_CLASSES
    '''
    res = torch.empty_like(gt_tokens)
    res[0] = gt_tokens[0]
    res[1:] = (gt_tokens[1:] - gt_tokens[:-1]) % NUM_CLASSES
    return res


def cmd_train(args):
    pass
# WARNING: Decompyle incomplete


def _build_train_parser(sub):
    ap = sub.add_parser('train', help = 'Train HPAC on residual tokens.')
    ap.add_argument('--epochs', type = int, default = 100)
    ap.add_argument('--bs', type = int, default = 16)
    ap.add_argument('--lr', type = float, default = 0.0001)
    ap.add_argument('--lr-emb', type = float, default = 0.001)
    ap.add_argument('--ch', type = int, default = 32)
    ap.add_argument('--P', type = int, default = 16)
    ap.add_argument('--delta', type = int, default = 1)
    ap.add_argument('--scn-from', type = int, default = 50)
    ap.add_argument('--lam-init', type = float, default = 1e-05)
    ap.add_argument('--lam-final', type = float, default = 0.001)
    ap.add_argument('--save', type = str, default = str(SAVE_DIR / 'hpac_residual.pt'))
    ap.add_argument('--smoke', action = 'store_true')
    ap.set_defaults(func = cmd_train)
    return ap


def main():
    parser = argparse.ArgumentParser(description = 'HPAC: entropy model + arithmetic codec + training')
    sub = parser.add_subparsers(dest = 'cmd', required = True)
    _build_train_parser(sub)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
    return None

"""
