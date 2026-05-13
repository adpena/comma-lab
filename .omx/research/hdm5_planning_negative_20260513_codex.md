# HDM5 Planning Negative - 2026-05-13

research_only=true
score_claim=false
dispatch_attempted=false
ready_for_exact_eval_dispatch=false

## Question

After HDM4 became the current byte-closed PR106/R2 low-level decoder section,
the next obvious byte target was whether a more expressive HDM5 order/partition
search could beat HDM4 before any runtime/eval work.

## Evidence

Command:

```bash
/usr/bin/time -p .venv/bin/python tools/profile_hnerv_decoder_structural_recode.py \
  --source-archive experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/archive.zip \
  --source-label pr106_r2_hdm4_release_surface_20260513 \
  --include-hdm5-search \
  --hdm5-max-parts 8 \
  --hdm5-workers 8 \
  --hdm5-top-k 16 \
  --json-out experiments/results/hdm5_planning_profile_20260513_codex/profile.json
```

Runtime: 34.98s wall / 169.68s user on local macOS CPU.

Profile artifact:

- path: `experiments/results/hdm5_planning_profile_20260513_codex/profile.json`
- bytes: `77475`
- sha256: `8b067c0367939fb464b581bd95e14c1c76202fed076aee82011e0f4b6fadb4fd`

Source decoder section:

- codec: HDM4 fixed recipe DP4 q-Brotli/raw scales
- bytes: `169990`
- sha256: `76a1156369b6f3a54c011261137684ec1b4f70331e2d4335dea8761e5d28aa06`

HDM5 result:

- verdict: `hdm5_self_describing_search_does_not_beat_hdm4`
- best self-describing HDM5 bytes: `170024`
- delta vs HDM4 section: `+34` bytes
- best family: `hdm4_role_order`
- best part count: `4`
- best fixed-recipe projection bytes: `169990`
- best fixed-recipe projection delta vs HDM4 section: `0` bytes

## Classification

This is a measured planning negative, not a model result and not a score claim.
The self-describing HDM5 grammar loses to HDM4 by 34 bytes on the current HDM4
decoder section. The best hypothetical fixed-recipe projection only ties HDM4,
so it does not justify runtime, archive-builder, PacketIR identity, or exact
auth-eval work.

## Routing

Do not promote HDM5 from this search surface. Reopen only if a new objective
strictly beats `169990` charged decoder-section bytes including all headers,
length prefixes, record-order metadata, split metadata, and raw scales.

Higher-EV next score-lowering work remains:

- exact Linux x86 `[contest-CPU]` closure for HDM4 once provider/account
  billing permits;
- non-HDM4 decoder transforms that change the charged `169990` byte section;
- PR106 latent/sidecar materialization only when it creates a byte-closed exact
  archive candidate rather than a planning-only selector profile;
- HNeRV parity training and real substrate trainers before more local
  micro-grammar churn.

Verification:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_hnerv_decoder_recode.py
# 17 passed
```
