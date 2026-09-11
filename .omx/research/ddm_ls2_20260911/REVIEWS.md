# LS2 code review receipt

Source: experiments/ddm_ls2_probability_bound.py (final hash in BINDING.json).
Parent reviewed model definitions, int8/Q10 accounting, frozen base and box tangent. Independent
agent ls2_recall reviewed all source and protocol before fitting.

First review found resume-price outputs not bound to parameter bytes, incomplete NPZ sidecar restart,
unverified fit completion, unpinned LS1 helpers, missing split-specific omission accounting and
missing numerical/zero controls. Fixed before feature launch: price-stage binding, completion markers,
verified parameter/result receipts, source pins, retained per-frame summaries, zero objective
reconciliation, real-row NumPy loss/gradient/Hessian control, and zero correction integer-row identity.
Second source review was clean for bounded scorer-free launch. Final two review-tracker passes applied.
Ruff E9/F63/F7/F82 and compilation passed. Runtime controls live in consumer store.

Limitation: zero correction control repeats the pricing override instead of calling one shared helper.
This is weaker regression protection, not a current differing algorithm. No production implementation
is claimed. Shared assumption: fixed inherited weights with four small causal count banks; relaxing it
could find other gains, so these bounds do not close all models or generators.

Final source transition: concurrent fit startup exposed a shared BINDING.json.new rename race;
one child exited before fitting. Whole-stage fcntl lock plus automatic retention manifests added
only in main. Independent AST review verified all other 15 functions identical and validated both
source pins and retained old source/binding. New source received two clean review-tracker passes.
Both full fits subsequently completed through the locked CLI; no numerical function was modified.

Independent final measurements: imported shipped codec functions match experiment ASTs; all four
payload headers/hashes/lengths pass. Numerical upper bounds recompute exactly. C-vs-NumPy loss,
gradient and Hessian controls pass. Repricing all four models on real frames 0,299,599 through the
shipped codec matches all 12 class/row/boundary tables exactly, max error 0; all frequencies positive
and balanced to 2^31. Full n600 result is 385.550480 B net joint and 266.162261 B net separate,
integer likelihood equivalents only. No native receiver timing or physical recode was performed.

<!-- # FORMALIZATION_PENDING: measurement memo; no score row -->
