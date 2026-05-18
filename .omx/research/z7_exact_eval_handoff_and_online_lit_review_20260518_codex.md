# Z7 exact-eval handoff and online literature review - 2026-05-18

Author: Codex

## Summary

This pass converted the Z7 recurrent-vs-static score-aware smoke into a
machine-readable no-spend exact-eval handoff packet and adversarially checked the
current Z7-GRU / Z7-Mamba planning assumptions against current video-compression
and sequence-model literature.

No provider job was launched. No lane claim was opened. This is
`score_claim=false`, `promotion_eligible=false`,
`ready_for_paid_dispatch=false`.

## New handoff artifact

Tool:

```text
tools/verify_z7_exact_eval_handoff.py
src/tac/tests/test_verify_z7_exact_eval_handoff.py
```

Latest packet:

```text
.omx/state/z7_exact_eval_handoff/z7_exact_eval_handoff_20260518T133855Z.json
```

Packet facts:

```text
schema=z7_exact_eval_handoff_v1
lane_id=lane_per_substrate_symposium_z7_lstm_predictive_coding_20260517
current_pair_count=1
required_pair_count=600
ready_for_exact_eval_handoff=false
result_review_blockers=[z7_exact_handoff_current_packet_not_600_pairs]
same_archive_zip_bytes=true
runtime_output_changed_vs_recurrent=true
runtime_output_byte_differences_vs_recurrent=1011870
runtime_custody.aggregate_sha256=d7a3297e011dfc1d271b9e89997a3eda74271fa4237d577d4a7dd4f90e178d2f
score_claim=false
promotion_eligible=false
provider_dispatch_attempted=false
lane_claim_opened=false
```

Interpretation: the current Z7 packet is a valid byte-closed local mechanism
probe, not a dispatchable exact-eval candidate. The tool emits non-executing
paired CPU/CUDA command plans for custody visibility, but it emits no executable
commands until a ratified 600-pair packet exists.

False-authority guard added: if the stats JSON sets `score_claim`,
`promotion_eligible`, or `ready_for_paid_dispatch` truthy, the handoff verifier
suppresses all plan/execute commands.

## Online sources checked

- DCVC / Deep Contextual Video Compression:
  <https://arxiv.org/abs/2109.15047>
- HNeRV:
  <https://arxiv.org/abs/2304.02633>
- HiNeRV:
  <https://arxiv.org/abs/2306.09818>
- Mamba:
  <https://arxiv.org/abs/2312.00752>
- Mamba-2 / state-space duality:
  <https://arxiv.org/abs/2405.21060>
- DreamerV3:
  <https://arxiv.org/abs/2301.04104>

## Adversarial findings

### 1. Z7-GRU must not collapse into plain predictive residual coding

DCVC explicitly frames classic "predicted frame plus residual" video coding as
suboptimal versus conditional/contextual coding. Z7's current scaffold is still
useful because it proves archive grammar, scorer-free inflate, same-byte static
control, and score-aware training can coexist. But the next optimized Z7 design
should not optimize only:

```text
z_t = predictor(z_{t-1}, ego_t) + residual_t
```

as a raw residual channel. The stronger unique-per-method target is:

```text
context_t = GRU(z_{t-1}, ego_t)
decoder_features_t, entropy_scales_t = f(context_t)
z_t = context_conditioned_decoder(context_t, residual_symbols_t)
```

Concrete next design implication: the next Z7 full packet should compare at
least these modes under same archive bytes, same runtime, and same paired axes:

```text
A: current recurrent latent predictor + int8 residual baseline
B: recurrent context-conditioned decoder feature modulation
C: recurrent context-conditioned entropy scale / residual-symbol coder
```

The disambiguator should score recurrent temporal coherence as a conditional
coding advantage, not merely as lower L2 residual norm.

### 2. HNeRV / HiNeRV argue for content-adaptive and hierarchical decoder capacity

HNeRV's key lesson is content-adaptive embeddings plus high-resolution decoder
capacity, not frame-index-only regression. HiNeRV's key lesson is that simple INR
architectures underperform and that hierarchical frame/patch encodings plus
training/pruning/quantization pipeline matter.

Z7 consequence: adopting the Z6 decoder forever is not automatically optimal.
It is fine as the first same-byte control, but the next optimized Z7 packet
should include an explicit decoder-capacity decision:

```text
ADOPT_Z6_DECODER only for baseline isolation
FORK_Z7_DECODER if recurrent context needs feature-domain conditioning,
hierarchical patch/frame channels, or entropy-scale heads
```

This is the unique-and-complete-per-method rule applied at the decoder layer.

### 3. Mamba/Mamba-2 speed claims remain unproven for this contest geometry

Mamba and Mamba-2 provide strong evidence that selective SSMs are efficient and
competitive on long sequence workloads. They do not prove superiority over GRU
for this specific regime:

```text
sequence length=600 pairs
latent_dim=6 or 24
runtime budget=T4 exact inflate
archive target=single 0.bin contest packet
```

The Z7-Mamba-2 memo's cargo-cult classification is therefore correct. The first
action remains a measured timing/disambiguator smoke, not a paid full dispatch
based on language-scale throughput claims.

### 4. Z7-Mamba runtime dependency plan needs a contest-closure supersession

The current Z7-Mamba memo allows an inflate runtime with `torch + brotli +
mamba_ssm`, calling that a substrate-engineering waiver. That is too permissive
for contest promotion. The Z7-GRU runtime already hit and fixed the same bug
class by replacing external `brotli` with stdlib `zlib`.

Supersession for any future Z7-Mamba implementation:

```text
runtime may use torch
runtime must not require brotli
runtime must not require mamba_ssm unless the dependency is vendored and proven
  in the exact contest runtime closure
preferred path: pure-PyTorch exported selective-SSM recurrence in <=200 LOC
fallback path: research_only=true until dependency closure is proven
```

This keeps Mamba-2 distinct where it matters mathematically while refusing a
hidden runtime dependency that could win proxy training and fail exact inflate.

## Next optimized Z7 gate

The next artifact should be a 600-pair score-aware Z7-GRU packet builder that
emits:

```text
recurrent archive.zip
static_capacity_control/archive.zip
same archive byte count
submission_runtime/inflate.sh
stats JSON with score_claim=false
handoff JSON from tools/verify_z7_exact_eval_handoff.py
```

Only after that handoff is `ready_for_exact_eval_handoff=true` should the
paired dispatch lifecycle be opened through `tools/dispatch_modal_paired_auth_eval.py`.

## Verification

```bash
.venv/bin/python -m py_compile tools/verify_z7_exact_eval_handoff.py
.venv/bin/python -m pytest -q src/tac/tests/test_verify_z7_exact_eval_handoff.py
.venv/bin/python tools/verify_z7_exact_eval_handoff.py --json --write-artifact
```

Observed:

```text
3 passed in 0.15s
latest_handoff_artifact=.omx/state/z7_exact_eval_handoff/z7_exact_eval_handoff_20260518T133855Z.json
```
