# best write-up outline

## 1. premise

- Track A is the only intentionally non-rule-faithful lane.
- Track B is the honest scorer-backed lane.
- The writeup should foreground rigor, not just cleverness.
- Three arcs matter: codec + post-filter (Era 1, peak `1.73`), neural renderer
  controls (Era 2, peak control `1.05`), and public semantic/neural
  sufficient-statistic replay culminating in PR100 at `0.22826947142244708`.

## 2. key result (current)

- Best exact contest-CUDA result: **`0.22826947142244708`** (PR100
  HNeRV-LC-v2 adapter replay, exact Tesla T4 A++)
- Archive: `178981` bytes,
  SHA-256 `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- Runtime tree SHA-256:
  `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`

## 3. historical key result (Era 1)

- Best honest Track B result: **`1.73`**
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / sharpness=1 / long500 QAT+EMA learned int8 post-filter (h=64)`

## 4. main thesis

- Two paradigms compose: AV1+CNN post-filter taught the lab how to measure; neural renderer broke through the codec assumption.
- evaluator-boundary bugs can completely mask achievements (the MPS-vs-CUDA 23x PoseNet drift was the most consequential).
- disciplined measurement, exact auth-eval custody, runtime tree hashing,
  release hygiene, AI-assisted adversarial review, and explicit state-keeping
  are the right operating model.

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
13. NEW: C067/PR67 fixed-slice reproduction -> exact T4 `0.31561703078448233`
14. NEW: PR85/PR95/PR99 semantic and HNeRV replays -> `0.2581`, `0.2309`, `0.2297`
15. NEW: PR100 HNeRV-LC-v2 adapter replay -> exact T4 `0.22826947142244708`
16. NEW: hidden-gem roadmap - HPM1/HPAC parity, native action atoms, scorer-gradient atoms, byte self-compression, and field-policy waterfill

## 6. strategic-secrecy guardrails

- public-facing surfaces: only exact CUDA auth-eval scores can rank
- do NOT expose Lane W / Lane Ω / Lane DARTS-S internals on public surfaces
- do NOT publicize the Cloudflare site URL until human says it is time
- arXiv / paper writeup CAN have full disclosure but coordinate timing with submission

## 7. candid postmortem appendix

- Core lesson: we had many of the winning ideas, but not enough of them were
  lowered into byte-closed, exact-evaluable archives early enough.
- Required public framing: the gap was research-to-archive conversion latency,
  not lack of imagination.
- Link: `docs/postmortem_bridge_gap_20260505.md`
- Include the compiler chain:
  `idea -> typed stream -> deterministic payload -> inflate runtime -> no-op control -> exact CUDA eval -> evidence ledger`.
