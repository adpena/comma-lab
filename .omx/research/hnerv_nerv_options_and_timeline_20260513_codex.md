# HNeRV / NeRV Options And Timeline Review

Date: 2026-05-13
Author: Codex
Evidence status: research/control-plane synthesis, not a score claim
Score authority: none
Promotion eligible: false

## Question

What HNeRV/NeRV-family options, alternatives, and adjacent replacements could plausibly beat the current ~0.19 frontier, and how long would each take?

## Local public-PR evidence

The relevant public frontier chain in the local intake is:

- PR95 `hnerv_muon`: the actual training source. It contains an 8-stage curriculum in `src/train.py` and `src/stages/*`, with the source claiming `~50 hours on a single GPU` for deterministic from-scratch reproduction.
- PR98 `hnerv_muon_finetuned_from_pr95`: finetune and decode-side postprocess on top of PR95.
- PR100 `hnerv_lc_v2`: repacks PR95/PR98 with schema-driven decoder packing, compact latent packing, and a ~1.2 KB latent-correction sidecar.
- PR101 `hnerv_ft_microcodec`: codec/microcodec refinement over PR95/PR98/PR100. Its README reports local CPU score `0.19284` at 178,258 bytes. It is codec-heavy, not a new from-scratch training curriculum.
- PR102: constants-only scale/nudge retune of PR100.
- PR103: arithmetic-coding direction over the HNeRV lineage.
- PR104 `qhnerv_ft_best`: A100 finetune with CUDA-only runtime, but local README reports about 0.23, worse than the PR101 chain.

The core correction is that PR95 is the training discipline; PR101 is the frontier codec/payload discipline. To beat 0.19 inside the HNeRV basin, we should port both: PR95's training loop and PR101's archive discipline.

## Why PR95 could run 29,650 epochs

The epoch count is large but each epoch is tiny:

- one fixed contest video;
- about 600 frame pairs;
- batch size 8;
- about 75 optimizer steps per epoch.

Therefore 29,650 epochs is about 2.22M optimizer steps. The PR95 source says this takes about 50 GPU-hours, which implies about 6.1 seconds per epoch or about 81 ms per step. That is plausible because the model is only about 229K decoder parameters plus latents; the heavy work is the scorer-forward training loop.

Current public cloud pricing checks on 2026-05-13 imply a clean 50 GPU-hour reproduction is roughly:

- Modal T4: about $30
- Modal L40S: about $98
- Modal A100 40GB: about $105
- Modal A100 80GB: about $125
- Modal H100: about $197
- Lambda H100 at $2.99/hr: about $150

This is reproduction cost, not discovery cost. The author likely paid more during failed runs, architecture tuning, and bug fixes.

## Ranked HNeRV / NeRV options

### 1. PR95 byte-faithful full reproduction

Mechanism: port PR95 `HNeRVDecoder`, differentiable YUV6 scorer training, eval-roundtrip, QAT, C1a entropy regularizer, EMA, and Stage 8 Muon into canonical `tac` / `experiments` surfaces.

Expected score lever: anchor PR95 and unlock faithful Stage 8 / curriculum ablations. It may not beat PR101 by itself, but it gives the missing training substrate needed to push below PR101.

Time:
- port + tests: 1-2 days;
- timing smoke: 1-2 hours;
- full run: source claims ~50 GPU-hours;
- exact CUDA/CPU closure: 0.5-1 day.

Verdict: highest-confidence HNeRV foundation work. Do it if we want HNeRV meat without guessing.

### 2. PR95 full reproduction + PR101 microcodec export

Mechanism: train with PR95 curriculum, then export through PR101-style schema-driven packing, fp16 scales, split Brotli streams, compact latent packing, sidecar/no-op table selection.

Expected score lever: direct path to sub-0.19 if training improves distortion at same byte discipline, or if PR101 packing is improved without regressing decoded outputs.

Time:
- after PR95 port: 2-4 days wiring export parity;
- full train: ~50 GPU-hours;
- exact eval: 0.5-1 day.

Verdict: best HNeRV-family path to a lower exact score.

### 3. Stage-8-only Muon finetune from parsed PR95/PR101 archive

Mechanism: parse existing `0.bin`, reconstruct decoder/latents, run the Stage 8 loss with Muon hidden-conv updates, QAT, and C1a.

Expected score lever: smaller than full reproduction, but fast. Tests whether the frontier still has local trainable slack.

Time:
- parse-to-train port: 1-2 days;
- 1k-5k epoch finetune: likely 2-10 GPU-hours after timing smoke;
- exact eval: same day.

Verdict: best short feedback loop. It should run before a full expensive campaign if timing smoke is green.

### 4. PR95 curriculum mutations

Mechanisms:
- lower EMA decay or dual EMA;
- Muon earlier than Stage 8;
- AdamW/Muon partition sweep;
- C1a lambda/sigma grid;
- L7 margin threshold sweep;
- stronger score-domain early stopping on exported archive, not proxy loss.

Expected score lever: 0.002-0.015 if PR95 was not fully tuned.

Time: 1-3 days to wire; 10-150 GPU-hours depending on grid size.

Verdict: useful only after byte-faithful PR95 timing/provenance is in place.

### 5. PR100/PR101 latent sidecar search

Mechanism: improve the existing one-dim per-pair correction sidecar into multi-token, low-rank, frame-conditional, or score-gradient-guided corrections, while preserving exact byte accounting.

Expected score lever: 0.001-0.008. Strong because PR100 already proves a tiny sidecar can move score.

Time:
- local scorer-search and no-op guards: 1-2 days;
- exact packet: 1-2 days;
- exact CUDA eval: same day after claim.

Verdict: highest near-term score-lowering path inside the current PR101 packet family.

### 6. HNeRV architecture upgrade: NeRV++ / HiNeRV style

Mechanism: replace PR95 decoder blocks with stronger but still compact blocks:
- NeRV++-style separable conv residual blocks and bilinear skip;
- HiNeRV hierarchical positional encoding / patch-frame unified representation;
- deeper/wider depthwise/MLP/interpolation blocks.

Expected score lever: potentially large in generic RD literature, but contest risk is byte/runtime budget. HiNeRV reports large BD-rate savings over HNeRV on standard datasets, but that does not directly transfer to this scorer.

Time:
- canonical substrate + archive grammar: 3-7 days;
- first serious train: 50-200 GPU-hours;
- compression/quant/export parity: 2-5 more days.

Verdict: best "replace HNeRV but stay NeRV-family" path; expensive but plausible.

### 7. FFNeRV / flow-guided frame-wise representation

Mechanism: add compact optical-flow or motion conditioning so the network spends fewer bytes on redundant ego-motion content.

Expected score lever: good for dashcam motion; flow side information must be tiny or implicit.

Time:
- minimal flow-conditioned prototype: 3-5 days;
- full contest-compliant train/export: 1-2 weeks plus 50-200 GPU-hours.

Verdict: high-upside if paired with ego-motion priors; risky if flow payload gets large.

### 8. DS-NeRV / static-dynamic decomposition

Mechanism: split road/background static structure from moving/dynamic residuals. This matches dashcam data better than a monolithic latent per pair.

Expected score lever: potentially 0.005-0.02 if the static road/scene manifold compresses well.

Time:
- substrate design + data split: 3-6 days;
- training/export: 50-150 GPU-hours.

Verdict: strong domain-specific NeRV alternative.

### 9. Frequency-enhanced NeRV variants: FF-NeRV, SNeRV, FANeRV

Mechanism: explicitly allocate high-frequency capacity, or frequency-separate low/high bands. This connects to our S2SBS and SABOR audits.

Expected score lever: high for SegNet boundaries and PoseNet edges, but only if frequency bytes are scorer-visible and payload-closed.

Time:
- residual prototype: 2-4 days;
- full replacement: 1-2 weeks.

Verdict: do as residual atoms first, then promote if exact eval moves.

### 10. Ego-NeRV / vehicle-motion-conditioned HNeRV

Mechanism: condition decoder on low-dimensional ego-motion / SE(3) / foveation atoms rather than unconstrained per-pair latents.

Expected score lever: potentially better than generic HNeRV for this contest because PoseNet dominates frontier marginal value.

Time:
- motion feature + trainer wiring: 3-5 days;
- train/export: 30-100 GPU-hours.

Verdict: best domain-specific HNeRV mutation.

### 11. BlockNeRV / patch-block HNeRV

Mechanism: split frame into blocks or foveated regions with separate capacity. Boundary and road-center regions get different budgets.

Expected score lever: medium. Strong if SegNet/PoseNet hard regions are spatially concentrated.

Time:
- prototype: 2-4 days;
- full train/export: 50-150 GPU-hours.

Verdict: useful if SABOR boundary audit finds high stable-interior fraction.

### 12. TCNeRV / temporal-context NeRV

Mechanism: condition frame-pair decode on neighboring temporal context or learned temporal grids.

Expected score lever: medium. Helps smooth ego-motion but may add runtime and side state.

Time: 4-7 days plus 50-150 GPU-hours.

Verdict: lower priority than PR95+PR101 or ego-motion conditioning.

### 13. Quantization-native HNeRV

Mechanism: train the representation as the archive format from day zero: INT8/INT4/FP4/block-FP, entropy regularization, and archive-in-loop checkpoint selection.

Expected score lever: high if it reduces PR101 bytes without hurting scorer components.

Time:
- integrate into PR95 curriculum: 2-4 days;
- full campaign: 50-200 GPU-hours.

Verdict: mandatory for any serious HNeRV replacement.

### 14. HNeRV + non-HNeRV residual stack

Mechanism: keep PR101/PR95 as base and add byte-closed residual atoms: SIREN/FINER, wavelet, C3/Cool-Chic, LA-pose/Telescope foveation, SABOR/S2SBS.

Expected score lever: highest composite path because it avoids retraining the whole base for every idea.

Time:
- per residual atom: 1-4 days;
- exact eval per promising atom: same day after packet closure.

Verdict: best route to sub-0.17 if we do not want to bet everything on a single replacement substrate.

## External literature anchors

- NeRV: frame-index-to-frame neural representation, faster than coordinate MLPs for video.
- HNeRV: content-adaptive embeddings into a decoder; closer to the PR95 family.
- FFNeRV: flow-guided frame-wise representation.
- NeRV++: separable conv residual blocks plus bilinear skip layers.
- HiNeRV: hierarchical positional encoding, deep/wide light layers, pruning/quantization pipeline; reports large RD gains over HNeRV on standard datasets.
- DS-NeRV: decomposed static and dynamic codes.
- SNeRV/FANeRV: frequency-preserving or high-frequency enhanced variants.

## Recommended execution order

1. Timing smoke PR95 source/port for real seconds per epoch.
2. Port PR95 faithfully, with tests proving differentiable YUV6, eval-roundtrip, QAT, C1a, EMA, Muon partitioning, and archive parse/rebuild.
3. Stage-8-only Muon finetune from parsed public archive as the fastest exact-eval candidate.
4. Full PR95 reproduction if the timing smoke matches the source's ~50 GPU-hour claim.
5. PR101 microcodec export over any improved PR95 weights.
6. In parallel, build residual atom packet paths for SABOR, S2SBS, SIREN/FINER, wavelet, and LA-pose/Telescope foveation.
7. Only after those anchors: launch a larger replacement substrate campaign for HiNeRV / FFNeRV / DS-NeRV / ego-motion-conditioned HNeRV.

## Non-NeRV options

The likely winner may be a retrained expensive representation, but it does not have to be NeRV/HNeRV. The non-NeRV space should be treated as first-class, especially because the contest score is not human PSNR. It is frozen-scorer equivalence-class compression.

### A. Score-aware SIREN / FINER / WIRE / BACON residual atoms

Mechanism: use coordinate MLPs only where they are efficient: small hard regions, boundaries, pose-sensitive patches, or residual maps over an HNeRV/A1 base. Full-frame SIREN is probably byte-inefficient; sparse residual SIREN is plausible.

Timeline:
- residual atom selector: 2-4 days;
- exact packet path: 2-5 days;
- training: 2-20 GPU-hours per serious atom family.

Why it could beat HNeRV: periodic/variable-frequency activations can fit boundaries and scorer-sensitive high-frequency detail with fewer local parameters than another full-frame renderer.

### B. Ballé / CompressAI hyperprior substrate

Mechanism: learned latent autoencoder with factorized or hyperprior entropy model, adapted to scorer-domain loss and exportable as an archive-native decoder.

Timeline:
- real byte-closed trainer/export path: 1-2 weeks;
- useful training: 50-300 GPU-hours depending on model size;
- exact runtime closure: 2-5 days.

Why it could beat HNeRV: it has a true entropy-model tradition instead of relying on generic Brotli over neural weights. The risk is decoder/runtime byte overhead and T4 execution complexity.

### C. Cool-Chic / C3 overfitted codec

Mechanism: overfit a lightweight hierarchical decoder plus entropy-coded latents per frame/region. This is closer to "train a tiny codec for this one video" than generic learned compression.

Timeline:
- residual/foveated prototype: 3-7 days;
- full replacement: 2-3 weeks;
- training: 20-150 GPU-hours.

Why it could beat HNeRV: Cool-Chic-like methods explicitly optimize per-signal decoder+latent rate/distortion and have low decoder complexity. This aligns with contest archive-as-model bytes.

### D. Wavelet / learned wavelet residual stack

Mechanism: keep the base renderer for low-frequency structure, then encode scorer-sensitive wavelet residuals with arithmetic/range coding and per-band bit allocation.

Timeline:
- residual prototype: 1-3 days;
- exact packet: 2-5 days;
- GPU: optional, 0-20 GPU-hours.

Why it could beat HNeRV: SegNet boundaries and PoseNet pose cues are sparse in frequency/space. Wavelets give a deterministic basis and cheap decode.

### E. SABOR boundary-only renderer

Mechanism: render/encode only pixels near SegNet decision boundaries; fill stable interiors with cheap constants or base output. This exploits argmax-only segmentation.

Timeline:
- audit: in progress;
- prototype: 2-5 days after audit;
- exact eval: same day after byte-closed packet.

Why it could beat HNeRV: HNeRV spends bytes on full-frame visual fidelity. The contest does not require human visual fidelity.

### F. S2SBS high-frequency byte-stuffing / blindspot channel

Mechanism: use stem/resampling blindspots as a controlled side channel or as a license to discard/inject high-frequency content without scorer change.

Timeline:
- audit: in progress;
- real codec with ECC + runtime recovery: 3-7 days;
- exact eval: same day after packet closure.

Why it could beat HNeRV: it attacks equivalence classes directly rather than improving reconstruction.

### G. LA-pose / Telescope / ego-motion foveation

Mechanism: exploit dashcam priors: forward ego-motion, road plane, vanishing point, central foveation, and pose-scorer sensitivity. Encode pose-relevant structure rather than full RGB fidelity.

Timeline:
- residual prototype: 2-5 days;
- replacement substrate: 1-3 weeks;
- training: 20-200 GPU-hours.

Why it could beat HNeRV: pose marginal value near the frontier is high; HNeRV is not explicitly vehicle-dynamics-aware.

### H. Arithmetic/range/ANS compiler stack

Mechanism: treat every payload stream as typed symbols; replace generic Brotli where the modeled entropy plus table overhead wins. This applies to HNeRV latents, SIREN weights, sidecars, wavelet bands, and Ballé latents.

Timeline:
- per-stream exact transform: 1-5 days;
- full stack compiler: 2-6 weeks.

Why it could beat HNeRV: PR101's microcodec shows archive grammar matters. The next byte frontier likely requires semantic streams, not generic wrapper compression.

### I. Direct scorer-inverse / adversarial-evaluation representation

Mechanism: optimize generated frames only for SegNet argmax and PoseNet first-six outputs, not human reconstruction. This is the purest "model is the scorer equivalence class" direction.

Timeline:
- proof-of-concept: 3-7 days;
- contest-compliant archive path: 1-3 weeks;
- training: 20-300 GPU-hours depending on representation.

Why it could beat HNeRV: it targets the actual objective. The risk is brittle CPU/CUDA drift and compliance/custody burden.

## Why we did not run the big reproduction during the contest

This is the uncomfortable but useful postmortem.

1. We over-weighted short exact-eval loops and under-weighted long-burn training campaigns. The operating habit became "produce a byte-closed candidate now" instead of "start the 50 GPU-hour source-faithful reproduction while other agents optimize packets." This was wrong given the operator's explicit no-budget/no-time-limit direction.

2. We misclassified PR101 as the main source of HNeRV truth for too long. PR101 was the best score and therefore attracted analysis, but the training discipline was in PR95. That delayed the obvious campaign: reproduce PR95 first, then layer PR101's microcodec.

3. Cost uncertainty was allowed to become decision paralysis. The clean PR95 source says about 50 GPU-hours. That is not cheap, but it is not prohibitive. We should have launched a timing smoke immediately and converted uncertainty into measured seconds/epoch.

4. Provider/custody failures burned cycles. Modal/Vast source-staleness, claim discipline, operator-authorize recipes, and exact-eval custody were real issues, but the right response was to harden them in parallel with a long run, not serialize all learning behind them.

5. We chased many local byte optimizations because they were measurable and safe. That produced useful guardrails, but it also kept us in a saturated PR101/HDM basin. The user explicitly allowed large time/cost; the system did not fully operationalize that as "start expensive training now."

6. We lacked a canonical campaign abstraction. A 50-300 GPU-hour campaign needs stages, gates, resumable checkpoints, live cost logging, exact stop criteria, and periodic harvest. Without that, every long run felt like a risky monolith rather than a managed experiment.

Corrective action: from here, any credible source-faithful training path gets a campaign plan, a timing smoke, and a launch/no-launch decision based on measured cost. It should run concurrently with short residual/codec lanes.

## Earlier horizon failure: NeRV, RAFT, SIREN, foveation were visible

This was not merely a PR95 reproduction miss. Several non-PR95 directions were visible early enough to run as long-burn campaigns:

- Lane 12 NeRV had a council design review by 2026-04-30 and later a production trainer `experiments/train_lane_12_v2_nerv_as_renderer.py`. It should have been promoted from horizon idea to a real staged training campaign.
- RAFT/radial pose had a 2026-04-30 council design with two safe modes: compress-time prior and compliance-gated inflate-time recompute. At minimum, the compress-time prior and disagreement measurement should have run.
- SIREN/coordinate INR was already a plausible model-class alternative. Full-frame SIREN may be weak, but score-aware SIREN/FINER/WIRE residual atoms should have had a campaign.
- Hyperbolic/telescopic foveation and LA-pose/ego-motion surfaced as domain priors. They should have been treated as representation campaigns, not memo-only ideas.

The root cause was execution architecture, not technical impossibility. We had too many "lanes" and not enough "campaigns." A lane produced a scaffold, memo, or small smoke. A campaign should have owned GPU-hours, checkpoints, costs, stage gates, exact eval closure, and failure criteria.

Corrective operating rule:

If a direction plausibly moves the frontier by more than 0.01 and requires long training, it must be converted within one session into:

1. a campaign ledger;
2. a timing smoke;
3. a resumable training command;
4. a cost model;
5. a claimed dispatch lane id;
6. a harvest/eval plan;
7. an explicit launch/block decision.

The long-burn campaign set that should now run in parallel:

| Campaign | Type | First gate | Full burn | Why |
|:--|:--|:--|:--|:--|
| PR95+PR101 | HNeRV source-faithful | 10-50 epoch timing smoke | ~50 GPU-hours from source claim | recover training truth + frontier codec |
| Lane 12-v2 NeRV | NeRV replacement | small CUDA real-pair smoke | 5-50 GPU-hours depending config | previously built full RGB renderer path |
| RAFT/ego-motion | domain prior | flow-vs-PoseNet disagreement audit | 0-20 GPU-hours | exploit dashcam physics and pose marginal |
| SIREN/FINER residual | non-HNeRV INR | hard-region atom smoke | 2-20 GPU-hours | escape HNeRV full-frame local minimum |
| LA-pose/Telescope foveation | domain/foveal | runtime-consumption smoke | 0-20 GPU-hours | exploit ego-motion/foveated scorer sensitivity |
| Ballé/Cool-Chic/C3 | learned codec | byte-closed residual smoke | 20-300 GPU-hours | entropy-model path outside HNeRV |

The point is not to pick one prematurely. The point is to keep every high-EV representation class training or proving failure while byte-level work continues.

## Sources

Local:
- `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/README.md`
- `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/train.py`
- `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/stages/common.py`
- `experiments/results/public_pr_archive_release_view/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/README.md`
- `experiments/results/public_pr_archive_release_view/public_pr100_intake_20260505_auto/source/submissions/hnerv_lc_v2/inflate.py`
- `experiments/results/public_pr_archive_release_view/public_pr98_intake_20260505_auto/source/submissions/hnerv_muon_finetuned_from_pr95/README.md`

Online:
- NeRV: https://arxiv.org/abs/2110.13903
- HNeRV: https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html
- FFNeRV: https://arxiv.org/abs/2212.12294
- HiNeRV: https://arxiv.org/abs/2306.09818
- DS-NeRV: https://arxiv.org/abs/2403.15679
- NeRV++: https://huggingface.co/papers/2402.18305
- SNeRV: https://link.springer.com/chapter/10.1007/978-3-031-73001-6_19
- FANeRV / high-frequency enhanced hybrid representation: https://www.sciencedirect.com/science/article/pii/S0957417425011741
- Modal pricing checked 2026-05-13: https://modal.com/pricing
- Lambda pricing checked 2026-05-13: https://lambda.ai/service/gpu-cloud/pricing
- Cool-Chic: https://arxiv.org/abs/2307.12706 and https://github.com/Orange-OpenSource/Cool-Chic
- Ballé hyperprior: https://arxiv.org/abs/1802.01436
- SIREN: https://www.computationalimaging.org/publications/siren/
