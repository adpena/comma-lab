# logic/related_work.md

Typed dependencies. Each entry says how the cited work relates: as an
import (we directly use a technique), a bound (we measure ourselves against
their result), or a baseline (we use them as the comparison floor).

---

## Frozen-scorer task-aware compression

- **import**: comma.ai challenge scoring formula and frozen scorer modules.
  Forensic binding: `../../../upstream/modules.py`,
  `../../../upstream/evaluate.py`. We do NOT modify upstream files; the
  pinned snapshot is the authoritative scorer behavior.
- **import**: SVT-AV1 (Era 1 codec). Used as `libsvtav1` via FFmpeg with
  `preset=0`, `crf=34`, `film-grain=22`. Forensic binding:
  `submissions/robust_current/encode.sh`.

## Quantization-aware training

- **import**: FakeQuant straight-through estimator (Jacob et al. 2018).
  We use it specifically to close the train-to-deploy gap in the presence
  of extreme downstream sensitivity (PoseNet trust radius < 1e-4 pixels).
  This combination is, to our knowledge, novel.

## Inverse steganalysis

- **methodology bound**: Fridrich and collaborators' UNIWARD framework
  (errors in textured regions are undetectable by CNN steganalysis). We
  weight our learned post-filter loss by inverse local variance for
  rate-aware error placement.
- **competitive intelligence bound**: Yousfi (challenge creator) was
  Fridrich's PhD student at Binghamton DDE Lab. The challenge IS inverse
  steganalysis. Our SegNet and PoseNet attack lanes are
  detector-informed embedding (Yousfi 2022).

## Public leaderboard (anonymized for this artifact)

- **baseline (#1)**: 0.33 entry. FiLM-conditioned depthwise-separable CNN at
  ~88K params. Reverse-engineered architecture characteristics live in our
  internal Lane Q-FAITHFUL design docs but are NOT published here.
- **baseline (#2)**: 0.38 entry. Self-compression at ~1.017 bpw, analytical
  pose via affine_delta, single-mask-per-pair + 6-DOF affine duality.
  Reverse-engineered shifts inform our Era 3 portfolio but the specific
  sequencing decisions are private.
- **our floor (#?)**: `1.05` [contest-CUDA] (Lane G v3, 694KB).

## Methodological reference for THIS artifact

- **import**: "The Last Human-Written Paper: Agent-Native Research Artifacts"
  (arXiv 2604.24658). The four-layer Ara structure (logic / src / trace /
  evidence) and the Live Research Manager event taxonomy come from this
  paper. Our adoption is partial: we use the layer structure verbatim and
  hand-compile from existing prose drafts; the Live Research Manager hook
  into our development workflow is scaffolded in `tools/ara_compile.py` but
  not yet automated.

## Negative space (what we do NOT use)

- **end-to-end neural codecs (DVC, FVC, DCVC)**: too big for the byte budget
  and incompatible with the 30-min CPU runtime constraint at inflate time.
- **training-time scorer modifications**: the strict-scorer rule prohibits
  loading PoseNet/SegNet at inflate time. All scorer access is compress-time
  only.
- **MPS-based authoritative measurement**: forbidden by CLAUDE.md as of
  2026-04-25; preflight Check 1 enforces.
