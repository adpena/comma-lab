# CPU/CUDA Drift Adversarial Review

date: `2026-05-08`
owner: `codex`
evidence_grade: `mixed; contest-CPU numeric anchor plus diagnostic mechanisms`
score_claim: `false`
mechanism_claim_proven: `false`

## Numeric anchors

PR107 now has a Linux x86_64 `[contest-CPU]` replay:

- archive bytes: `178392`
- archive SHA-256:
  `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- score: `0.1966358879`
- seg: `0.00058931`
- pose: `0.00003580`
- rate: `0.00475136`
- workflow: `25556454358`, GitHub Actions `ubuntu-24.04`, x86_64
- artifact:
  `experiments/results/pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/contest_auth_eval.adjudicated.json`

The M5 Max CPU replay was `0.19664189`, only `6e-6` above the Linux CPU row.
This supports macOS CPU as a fast development proxy for HNeRV-class CPU-axis
sweeps, but not as a promotion tag.

Public PR-comment pairs remain useful external calibration:
`reports/public_pr100_108_cpu_cuda_drift_analysis_20260508.json` reports five
paired HNeRV rows with median pose ratio `4.995`, median seg ratio `1.173`,
and median CPU pose `3.443e-05`.

## False or over-specific claims rejected

- FastViT-T12 attention compounding is false for this scorer. The local
  module inventory records zero attention/softmax-named modules and twelve
  `RepMixerBlock` modules.
- TF32-on-T4 is false. T4 is Turing; TF32 is an Ampere-and-later feature.
- A fixed 25/75 decoder/network attribution is not proven. It is a prior from
  earlier theory notes, not an isolated measurement.
- A lossy byte-rate win on PR107 is not automatically a leaderboard win. It
  must survive paired CPU and CUDA exact eval because past lossy coarsening
  produced large score cliffs on other substrates.

## Mechanism hypotheses still live

H1, decoder-dominant:
`evaluate.py` switches the ground-truth loader: DALI/NVDEC on CUDA and
PyAV/libav on CPU, while the compressed raw tensor path does not switch in the
same way. A 1 to 2 LSB decoded-RGB difference is large enough under plausible
PoseNet Lipschitz constants to explain the observed pose gap.

H2, network-kernel dominant:
PyTorch warns that CPU and GPU may differ even for bitwise-identical inputs.
PoseNet regression can amplify small numeric differences more than SegNet
argmax classification.

H3, mixed:
Decoder and network effects both contribute, and the split varies by
substrate class.

## Landed tooling

- `tools/analyze_cpu_cuda_eval_drift.py`: public-comment ratio analyzer.
- `tools/probe_eval_loader_drift.py`: DALI-vs-PyAV decoded-RGB diagnostic.
- `tools/probe_posenet_layer_drift.py`: shared-input PoseNet activation
  tracer. Forward-hook tensors are cloned so in-place modules cannot mutate
  captured activations.
- `tools/cathedral_autopilot.py`: evidence summary now refuses CPU-vs-CUDA
  priority unless both contest axes share archive SHA and runtime-tree SHA.
  macOS CPU rows are tracked as research proxies requiring Linux promotion.
- `tools/all_lanes_preflight.py`: Gate #22 validates that the loader-drift
  probe emits only non-promotable diagnostic artifacts.

## Next tests

1. Run `tools/probe_eval_loader_drift.py` on a T4 with DALI available and
   record per-channel LSB differences.
2. Force PyAV ground-truth decode while running CUDA scorer kernels as a
   mechanism-only fork. This isolates loader from network.
3. Inject uniform integer noise of amplitude 1, 2, and 3 LSB into a PyAV
   decoded tensor and run PoseNet on T4. If amplitude about 1.5 LSB induces
   pose drift near `1.4e-4`, H1 becomes dominant.
4. Run paired `[contest-CUDA]` and `[contest-CPU]` on the exact PR107
   lossy-coarsening candidate before making any medal/gold claim.

## External references

- PyTorch numerical accuracy:
  <https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html>
- NVIDIA TF32 on Ampere:
  <https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/>
- NVIDIA T4/Turing:
  <https://www.nvidia.com/en-gb/data-center/tesla-t4/>
- NVIDIA DALI video reader:
  <https://docs.nvidia.com/deeplearning/dali/archives/dali_2_1_0/user-guide/operations/nvidia.dali.fn.experimental.readers.video.html>
- FastViT/RepMixer:
  <https://arxiv.org/abs/2303.14189>
