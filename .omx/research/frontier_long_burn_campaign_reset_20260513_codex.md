# Frontier Long-Burn Campaign Reset

Date: 2026-05-13
Author: Codex
Status: campaign planning / no dispatch performed
Score claim: false
Promotion eligible: false

## Reason

The operator correctly pointed out that NeRV, SIREN, RAFT/ego-motion, foveation, and related representation families were visible well before the endgame, and that a 50 GPU-hour burn over that time horizon was not a serious budget obstacle.

The failure mode was not technical impossibility. It was failure to convert high-upside horizon ideas into managed long-burn campaigns. A campaign is different from a lane:

- lane: scaffold, memo, smoke, or one exact-eval packet;
- campaign: timing smoke, claimed provider job, resumable checkpoints, cost telemetry, staged gates, exact-eval harvest, and clear stop/continue criteria.

Going forward, high-EV representation ideas should not sit as memos. They either launch a timing smoke or receive a written blocker.

## Campaign matrix

### Campaign 1: PR95 + PR101 HNeRV source-faithful campaign

Goal: recover PR95 training truth, then export with PR101-grade microcodec discipline.

Existing evidence:
- `public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/train.py` claims ~50 hours on one GPU.
- PR95 contains the 8-stage curriculum.
- PR101 contains the better codec/microcodec path and reports local CPU score 0.19284 at 178,258 bytes.

First gate:

```bash
cd /Users/adpena/Projects/pact
python -m submissions.hnerv_muon.src.train
```

The source command is not yet canonicalized into `tac`; first action is a 10-50 epoch timing fork/port, not an untracked monolithic run.

Expected full burn:
- about 50 GPU-hours if PR95 source claim holds;
- around $30-$200 depending provider/GPU at 2026-05-13 listed rates.

Blockers before paid launch:
- port source to canonical `tac`/`experiments`;
- lane claim;
- operator-authorized recipe;
- checkpoint and optimizer-state custody;
- final archive exact CUDA/CPU eval plan.

### Campaign 2: Lane 12-v2 NeRV-as-renderer

Goal: train a full RGB NeRV renderer, not a mask slot replacement, as a real HNeRV alternative.

Existing command surface was verified from argparse:

```bash
.venv/bin/python experiments/train_lane_12_v2_nerv_as_renderer.py \
  --output-dir experiments/results/lane_12_v2_nerv_campaign_<UTC> \
  --device cuda \
  --epochs 200 \
  --batch-size 8 \
  --eval-every-epochs 25
```

First gate:
- CUDA timing smoke with `--max-pairs` and a small epoch count;
- no auth eval until a committed Phase-B auth memo and lane claim exist.

Expected full burn:
- previous code comment estimates $30-$50 / 5-10h on T4 for one serious run;
- larger configs could be 20-100 GPU-hours.

Why it matters:
- this was visible from the 2026-04-30 Lane 12 council and should have run as a long-burn campaign.

### Campaign 3: RAFT / ego-motion pose campaign

Goal: exploit dashcam physics and pose marginal value with compress-time RAFT/radial priors, without inflate-time compliance risk.

Existing command surface:

```bash
.venv/bin/python experiments/derive_poses_from_raft.py \
  --video upstream/videos/0.mkv \
  --baseline-poses <baseline_poses.pt> \
  --output experiments/results/raft_pose_campaign_<UTC>/raft_poses.pt \
  --manifest-output experiments/results/raft_pose_campaign_<UTC>/manifest.json \
  --device cuda \
  --n-frames 1200
```

First gate:
- locate/canonicalize baseline pose tensor path;
- run RAFT disagreement audit against PoseNet-derived targets;
- decide whether it is a prior, sidecar, or dead for this scorer.

Expected burn:
- 0.5-5 GPU-hours for audit;
- 10-50 GPU-hours if trained into a representation/sidecar.

Important compliance boundary:
- compress-time RAFT is allowed as training/prior machinery;
- inflate-time RAFT is compliance-gated because it relies on a large learned network at runtime.

### Campaign 4: SIREN / FINER / WIRE / BACON INR campaign

Goal: avoid full-frame canonical SIREN local minimum by training activation-family INRs as hard-region residual atoms or domain-prior renderers.

Existing command surface:

```bash
.venv/bin/python experiments/train_substrate_siren.py \
  --output-dir experiments/results/siren_activation_campaign_<UTC> \
  --epochs 2000 \
  --device cuda \
  --activation-family siren \
  --dispatch-contract naked_siren_replacement
```

First gate:
- residual/hybrid contracts currently fail closed until archive builders exist;
- timing smoke activation family variants;
- decide whether full replacement is worth a full burn or residual atoms only.

Expected burn:
- 2-20 GPU-hours per serious residual atom config;
- 50-150 GPU-hours for full replacement sweep.

### Campaign 5: LA-pose / Telescope foveation campaign

Goal: use ego-motion and foveated scorer sensitivity as representation priors, not as isolated payload artifacts.

Existing evidence:
- LA-pose/foveation tools and LFV1 candidate exist;
- Telescope distinction memo exists;
- runtime-consumption and exact-eval blockers are explicit.

First gate:
- build a runtime-consumed foveation packet that mutates output in a proven way;
- no score claim until exact CUDA eval.

Expected burn:
- 0-5 GPU-hours for packet/proof work;
- 20-100 GPU-hours for learned foveal representation.

### Campaign 6: Ballé / Cool-Chic / C3 learned-codec campaign

Goal: move beyond generic HNeRV weight compression into true entropy-model / overfitted-codec substrates.

First gate:
- byte-closed residual path with consumed archive payload;
- exact output mutation/no-op proof;
- small CUDA training smoke.

Expected burn:
- 20-300 GPU-hours depending on model size and export/runtime closure.

## Immediate correction

Do not retire or defer a high-upside representation because it needs long training. The correct first step is a timing smoke. Once the timing smoke exists, the launch decision is factual:

- seconds/epoch;
- GPU-hours for full run;
- provider rate;
- checkpoint storage;
- exact-eval closure cost;
- expected score lever;
- stop gates.

The long-burn campaigns should run concurrently with short byte-level packet work. Short packet work cannot be allowed to monopolize the whole contest strategy again.
