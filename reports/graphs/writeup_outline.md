# best write-up outline

## 1. premise

- Track A is the only intentionally non-rule-faithful lane.
- Track B is the honest scorer-backed lane.
- The writeup should foreground rigor, not just cleverness.
- Two paradigms have produced floors: codec + post-filter (Era 1, peak `1.73`) and neural renderer (Era 2, peak `1.05`).

## 2. key result (current)

- Best contest-CUDA result: **`1.05`** (Lane G v3, Era 2 — neural renderer)
- Modal T4 reproduction: **`1.04`** (within 0.01 noise)
- Recipe: dilated-h64 renderer + KL distill weight=0.002 + pose TTO retry

## 3. historical key result (Era 1)

- Best honest Track B result: **`1.73`**
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / sharpness=1 / long500 QAT+EMA learned int8 post-filter (h=64)`

## 4. main thesis

- Two paradigms compose: AV1+CNN post-filter taught the lab how to measure; neural renderer broke through the codec assumption.
- evaluator-boundary bugs can completely mask achievements (the MPS-vs-CUDA 23x PoseNet drift was the most consequential).
- disciplined measurement (78 STRICT preflight checks), long-horizon training, and explicit state-keeping are the right operating model.

## 5. strongest visual story beats

1. honest baseline at `4.06`
2. x265 ladder down to `3.25`
3. AV1 bug at `97.45`
4. AV1 repair to `2.20`
5. honest AV1 tuning down to `2.08`
6. tiny learned post-filter to `2.05`
7. long-horizon QAT+EMA h16 to `1.99`
8. long-horizon QAT+EMA h32 to `1.95`
9. compound scaling all the way to `1.73` (Era 1 final)
10. NEW: paradigm shift — abandon the codec entirely → neural renderer baseline `0.90` (CUDA-true; MPS reading was `2.26`, 2.5× drift)
11. NEW: pose TTO from baseline poses → Lane A `1.15`
12. NEW: KL distill weight=0.002 + pose TTO retry → Lane G v3 **`1.05`**
13. NEW: leaderboard context — Quantizr 0.33 #1, Selfcomp 0.38 #2, Mask2mask 0.60 #3, ours `1.05` would rank ~4th
14. NEW: live Selfcomp-paradigm work (no scores yet — eight Modal lanes in flight)

## 6. strategic-secrecy guardrails

- public-facing surfaces: only [contest-CUDA] tagged scores
- do NOT expose Lane W / Lane Ω / Lane DARTS-S internals on public surfaces
- do NOT publicize the Cloudflare site URL until human says it is time
- arXiv / paper writeup CAN have full disclosure but coordinate timing with submission
