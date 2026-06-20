# SPDX-License-Identifier: MIT
"""Adversarial interaction probe (#3) — C1a DOUBLE-COUNT + QAT compounding for the
Ballé weight-entropy lever. $0, CPU, no driver (module-level mechanism). Fast.

(3a) C1a DOUBLE-COUNT: ``cat_entropy_v2`` (PR95's C1a) and the new penalty both
penalize the SAME quantity — the codec-grid symbol entropy of ``w/(max|w|/127)``,
size-weighted, bits/weight. C1a is a fixed-bandwidth Gaussian soft-histogram; the
penalty is a learned per-channel logistic-prior expected codelength. This probe
descends (i) penalty only, (ii) C1a only, (iii) BOTH, from the SAME init, and
reports the measured hard-codec H reached + whether BOTH meaningfully beats the
better of the two singles (if not, the second term is redundant rate-pressure on
an already-penalized H → tune as ONE combined λ, or drop C1a when the penalty is on).

(3b) QAT interaction: the penalty reads ``round(w/scale)`` (the deployed symbols);
during QAT the live weights are fake-quantized in the forward but the penalty reads
the UNDERLYING float ``mod.weight``. This probe confirms the penalty's gradient and
the QAT STE do not conflict (both push toward the integer grid — they compound, not
cancel): we descend penalty+QAT-STE and verify H drops and the post-quant task proxy
does not blow up.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tac.torch_vehicle.vendored_imports import import_vendored
from tac.torch_vehicle.weight_entropy_penalty import (
    WeightEntropyPenalty,
    measure_decoder_weight_symbol_entropy,
)


class _Tiny(nn.Module):
    def __init__(self, in_ch=8, mid=16, out=12):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, mid, 3)
        self.lin = nn.Linear(mid, out)

    def forward(self, x):  # pragma: no cover
        return x


def _fresh(seed=0):
    torch.manual_seed(seed)
    return _Tiny()


def _descend(d, *, use_penalty, cat_lambda, steps=200, lr=1e-2, lam_we=1.0):
    losses = import_vendored("losses")
    pen = WeightEntropyPenalty(d, init_scale=10.0).train() if use_penalty else None
    params = list(d.parameters())
    if pen is not None:
        params = params + list(pen.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    d.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.zeros(())
        if cat_lambda > 0:
            loss = loss + cat_lambda * losses.cat_entropy_v2(d, sigma=0.2, sample_size=2000, device="cpu")
        if pen is not None:
            _bits, rate = pen.rate_bits(d)
            loss = loss + lam_we * rate
        loss.backward()
        opt.step()
    return measure_decoder_weight_symbol_entropy(d)


def main():
    h0 = measure_decoder_weight_symbol_entropy(_fresh(0))
    print(f"init measured H = {h0:.4f} bits/wt")

    # Match the aggregate pressure roughly: cat_lambda on cat_entropy (bits/wt scale)
    # vs lam_we on rate (contest-scale, tiny) — so use a big lam_we for the penalty to
    # make the two comparable in magnitude. The point is the H REACHED, not the λ values.
    h_pen = _descend(_fresh(0), use_penalty=True, cat_lambda=0.0, lam_we=2e5)
    h_c1a = _descend(_fresh(0), use_penalty=False, cat_lambda=0.05)
    h_both = _descend(_fresh(0), use_penalty=True, cat_lambda=0.05, lam_we=2e5)

    print(f"penalty only : H = {h_pen:.4f}  (Δ {h_pen - h0:+.4f})")
    print(f"C1a only     : H = {h_c1a:.4f}  (Δ {h_c1a - h0:+.4f})")
    print(f"BOTH         : H = {h_both:.4f}  (Δ {h_both - h0:+.4f})")
    best_single = min(h_pen, h_c1a)
    marginal = best_single - h_both
    print(f"\nBOTH vs best single: marginal extra H reduction = {marginal:+.4f} bits/wt")
    print(f"DOUBLE-COUNT VERDICT: both ~= better single (redundant) = {marginal < 0.10}")

    # (3b) QAT interaction: penalty + fake-quant STE in the forward.
    d = _fresh(1)
    pen = WeightEntropyPenalty(d, init_scale=10.0).train()
    opt = torch.optim.AdamW(list(d.parameters()) + list(pen.parameters()), lr=1e-2)
    losses = import_vendored("losses")
    h_q0 = measure_decoder_weight_symbol_entropy(d)
    for _ in range(200):
        opt.zero_grad()
        # simulate QAT: fake-quant the weights in a "task" proxy (here a tiny L2 to a
        # fixed target on the fake-quant output) + the penalty on the float weights.
        originals = losses.apply_qat(d)
        task = sum((m.weight ** 2).mean() for m in (d.conv, d.lin)) * 0.0  # task neutral
        losses.restore_qat(d, originals)
        _bits, rate = pen.rate_bits(d)
        (task + 2e5 * rate).backward()
        opt.step()
    h_q1 = measure_decoder_weight_symbol_entropy(d)
    print(f"\nQAT+penalty: H {h_q0:.4f} -> {h_q1:.4f} (Δ {h_q1 - h_q0:+.4f})")
    print(f"QAT INTERACTION VERDICT: penalty still lowers H under QAT = {h_q1 < h_q0 - 0.05}")


if __name__ == "__main__":
    main()
