# Sub-0.17 frontier innovation roadmap (2026-05-13)

Status: planning + implementation-routing artifact
Score claim: false
Evidence axes: keep [contest-CUDA], [contest-CPU], and proxy/advisory signals separate.

## Operator correction

The frontier push cannot be reduced to more byte shaving. At the current HNeRV
CUDA anchor, one KiB is only about `0.00068184` score. Moving an HLM1-like
packet from `0.20638` to below `0.17` by rate alone would require about
`54.6 KiB` of charged archive savings at unchanged components, which is not a
credible short-term generic-compressor path. The right next move is model and
pipeline innovation first, with structured byte work amplifying a better
representation.

This roadmap ranks the non-local-minimum routes surfaced by the parallel
adversarial research/review passes. It is intentionally contest-constrained:
every path must end in a byte-closed archive/runtime packet, exact auth eval,
and preserved custody before it changes the frontier.

## Priority 1: PR95/PR101/HNeRV parity retrain

Mechanism: recover and canonicalize the public HNeRV training discipline into a
reusable trainer: eval-roundtrip export, differentiable scorer preprocessing,
score-domain validation, EMA, QAT, and entropy-aware decoder/latent grammar.

Why it can move below 0.17: this attacks the dominant problem directly rather
than post-hoc packing. HNeRV-family public packets won because the model class
matched the one-video scorer/video contract: compact temporal representation,
small decoder, small latent stream, and acceptable SegNet/PoseNet distortion.
The missing leverage is not another blind refire; it is the exact training and
export loop that produced the public jump below 0.25.

First concrete work:
- complete forensic recovery of PR95/PR100/PR101/PR103/PR106 training scripts,
  optimizer schedules, loss weights, export code, and archive grammars;
- add parity guards that compare train-time decoded frames against
  runtime-consumed frames;
- require score-domain validation to select the runtime packet, not just a
  checkpoint proxy.

Kill condition: a byte-closed exact CUDA packet using the recovered discipline
cannot beat the HLM1/PR106 CUDA comparator after custody review.

## Priority 2: HDM5 / PacketIR structured decoder and latent grammar

Mechanism: replace high-byte opaque Brotli sections with typed q-streams,
shared codebooks, context models, and deterministic runtime decoders.

Why it matters: generic Brotli is close to saturation on the current bytestream,
but the payload still has semantic structure. This work should target decoder
weights and latent/sidecar grammar, not ZIP cosmetics.

First concrete work:
- implement runtime decode for the best non-self-describing q-stream candidate;
- preserve old/new section SHA, section offsets, exact charged bytes, and
  runtime consumption proof;
- keep all candidate packets `score_claim=false` until exact CUDA.

Kill condition: metadata/runtime overhead eats the entropy floor, or
full-frame/runtime parity breaks for a rate-only transform.

## Priority 3: scorer-aware residual atom compiler

Mechanism: use SIREN, FINER, WIRE, BACON, C3, wavelet, and related bases as
sparse residual atoms over the current frontier decode, selected by scorer
sensitivity per byte.

Why SIREN still matters: pure coordinate-SIREN-as-full-video is not the best
imminent spend. SIREN-style bases become higher value when used as targeted
score-aware residuals over PR106/HLM1: small payload, explicit no-op controls,
and selection against SegNet boundary or PoseNet hard-pair sensitivity rather
than L2 energy alone.

First concrete work:
- feed cached scorer sensitivity maps into
  `tools/materialize_siren_residual_pr106_sidecar.py`;
- compare SIREN/FINER/WIRE/wavelet atoms under one byte-closed manifest schema;
- exact-eval only candidates that show real component-targeted predicted
  delta per byte and pass no-op mutation proof.

Kill condition: L2/local ranking looks good but exact CUDA components regress
or the sidecar is not consumed by the runtime path.

## Priority 4: HiNeRV / FFNeRV / non-NeRV full-renderer fanout

Mechanism: test content-adaptive, frequency, flow, hierarchical, and codec
substrates as full RGB renderers with declared archive grammar before training.

Why it matters: the public frontier says the winning model class is not a
generic image/video compressor. The next step is to make alternate
representation classes exportable and exact-evaluable, not to dispatch L0
scaffolds.

First concrete work:
- wire real trainers, recipes, archive grammars, and exact-eval packet paths
  before any SIREN/Balle/Cool-Chic/VQ spend;
- prioritize HiNeRV/FFNeRV and Cool-Chic/C3 because they encode temporal and
  entropy structure, not just per-coordinate memorization.

Kill condition: exact CUDA anchor stays at or above the current HNeRV comparator
with no component win and no byte-floor advantage.

## Priority 5: dashcam-domain pose/geometry exploitation

Mechanism: integrate LAPose, ego-motion priors, road-plane/foveation atoms, and
legal runtime numerical choices into the exact-evaluable stack.

Why it matters: the video is one dashcam stream and the scorer is known.
Physical vehicle motion, forward flow, horizon/road-plane geometry, and
scorer-specific blind spots are part of the problem definition. They should
shape the model, not sit as disconnected tools.

First concrete work:
- build an A1/HLM1 plus LAPose no-op mutation proof and consumption proof;
- run a small exact CUDA canary only after the trailer changes RGB in the
  runtime path;
- keep CPU/CUDA drift as a measured axis, never as interchangeable evidence.

Kill condition: no CUDA PoseNet or SegNet improvement, or the pose trailer is
not consumed.

## Priority 6: HNeRV plus hyperprior / overfit codec stack

Mechanism: combine representation, prediction, quantization, hyperprior,
arithmetic coding, and pack passes as one stack, not isolated lanes.

Why it matters: sub-0.17 likely requires combined rate and component movement.
Ballé/CompressAI/Cool-Chic/C3 ideas should be treated as stack stages around a
contest-specific overfit renderer, not as generic library demos.

First concrete work:
- formalize typed contracts:
  `representation -> prediction -> quantization -> hyperprior -> arithmetic -> pack`;
- add small conformance vectors and no-scorer-in-inflate guards;
- only dispatch once runtime grammar and exact archive construction exist.

Kill condition: exported integer/entropy path destroys float gains or adds
contest-noncompliant dependencies.

## Priority 7: public-frontier forensic recovery

Mechanism: mine public PRs and author repositories for the training pipeline,
not just final inflate artifacts.

Why it matters: the archive tells us what ran; the training scripts explain why
it worked. Recovering optimizer, curriculum, score-loss, and export details is
higher information gain than another local micro-ablation.

First concrete work:
- queue PR95/PR100/PR101/PR103/PR106 source, writeup, and linked-repo intake;
- extract exact runnable primitives only after URL, commit, and license/status
  are recorded;
- keep detached clones out of the shared worktree.

Kill condition: only prose/proxy evidence exists and no runnable primitive or
exact replayable training artifact is recoverable.

## Anti-local-minimum guard

Before spending time on a small patch, answer yes to at least one:

- Does it make a candidate archive/runtime packet or exact replay more likely?
- Does it recover public-frontier training or archive mechanics?
- Does it reduce a known high-byte semantic section, not just container noise?
- Does it wire a substrate into the canonical exact-eval path?
- Does it expose a scorer-aware component target that can beat 0.19?

If all answers are no, it is probably local-minimum work.

