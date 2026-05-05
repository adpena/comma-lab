---
name: CRITICAL DX — Auto Auth Eval at Checkpoints, Early Stop on AUTH Not Proxy
description: The proxy-auth drift went undetected for 2560 epochs because we never auth-evalled between ep1000 and ep3560. Auth got WORSE while proxy improved.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Proxy-auth PoseNet ratio went from 2.1x (ep300) to 11.1x (ep3560). The training
OVERFITTED to proxy-specific texture patterns that DALI hardware decode destroys.
This went undetected for 2560 epochs because train_distill.py only does proxy eval.

**Non-negotiable DX requirement:**
- Auto auth eval every 500 training epochs (generate frames → write .raw → run evaluate.py)
- Log auth results alongside proxy in the training log
- Early stopping based on AUTH score plateau, not proxy
- Alert when proxy-auth ratio exceeds 3x (indicating drift)
- Save both proxy and auth scores for every checkpoint

**Why:** Proxy with simulate_resize is an APPROXIMATION. At low PoseNet values,
tiny differences between bilinear interpolation and DALI NVDEC cause 11x gaps.
Training past the auth sweet spot wastes GPU money and produces worse submissions.

**The lesson:** ep1000 + pose TTO (auth 0.36) was better than ep3560 (auth 0.59).
We burned $1.50 training from ep1000 to ep3560 and made things WORSE on auth.
Auto auth eval would have caught this at ep1500 and saved $1.00+.
