# Latest Research Grounding - 2026-05-06 - Halley

Scope: latest paper/source grounding for LA-Pose, telescopic/foveated
compression, Fridrich/Yousfi categorical and steganographic priors, and
openpilot/comma priors. This is a research/control-plane ledger only.

Constraints:

- No GPU or remote dispatch was performed.
- No score claim is made here.
- Paper ideas remain planning-only until a byte-closed archive, deterministic
  inflate path, archive/runtime custody, and exact CUDA auth eval exist.
- Openpilot/comma priors may rank compression-time atoms for planning, but any
  inflate-time consumer, label remap, model, codebook, or side information must
  be charged in `archive.zip` or fixed contest code.

## Evidence Grade Used Here

- `strong-primary`: arXiv, CVF, DOI/publisher page, official author/project
  page, official comma/openpilot GitHub/blog/source.
- `supporting-primary`: primary but not directly contest-matched, or focused on
  human/PSNR perception rather than SegNet/PoseNet.
- `canonical-stale`: older source that is still an engineering invariant but
  not a latest 2025/2026 frontier item.
- `discovery-only`: useful search lead, not sufficient for local implementation
  or score decisions.

## Source Map

### LA-Pose / Driving Camera Motion

Source:

- `strong-primary`: Zhengqing Wang, Saurabh Nair, Prajwal Chidananda, Pujith
  Kachana, Samuel Li, Matthew Brown, Yasutaka Furukawa, "LA-Pose: Latent Action
  Pretraining Meets Pose Estimation", arXiv submitted 2026-04-30,
  `arXiv:2604.27448`, DOI `10.48550/arXiv.2604.27448`.
  <https://arxiv.org/abs/2604.27448>
- `strong-primary`: project page, CVPR 2026, same authors.
  <https://la-pose.github.io/>

Local modules/files this should inform:

- `src/tac/analysis/lapose_paper_contract.py`
- `experiments/optimize_poses.py`
- `src/tac/raft_pose.py`
- `src/tac/pose_gaussian_process.py`
- `src/tac/optimization/meta_lagrangian_allocator.py`
- `src/tac/optimization/field_equation_planner.py`
- `src/tac/optimization/bayesian_experimental_design.py`
- Archive members: `optimized_poses.bin`, `zoom_scalars.bin`

Local variables/contracts:

- `LAPOSE_PAPER_REFERENCE["implementation_alignment"]`
- `missing_paper_components`
- `pose_atom`, `expected_pose_dist_delta`, `d_pose_dist_d_epsilon`
- `pose_score_delta(base_pose_dist, pose_dist_delta)`
- `init_pose`, `frame_indices`, `optimized_poses`, `zoom_scalars`

Engineering mapping:

- Treat latent-action pretraining as a planning prior for pose atoms, not as a
  contest score source.
- The local contract already marks this as
  `inspired_planning_only_not_paper_faithful_model`; that must remain true
  until inverse/forward dynamics pretraining, tokenizer, latent-action encoder,
  and pose-head semantics are implemented and audited.
- For immediate local use, LA-Pose can justify ranking candidate pose streams or
  pose-initialization sources, especially those built from driving-video motion
  features.

Gaps/risks before any score claim:

- No paper-faithful local model is present.
- No custody bridge from LA-Pose weights/features to a charged contest archive.
- Any pose replacement must pass local output parity / pose-safety checks and
  exact CUDA auth eval on the exact archive bytes.
- ArXiv/project evidence is pose-estimation benchmark evidence, not contest
  PoseNet evidence.

### Telescopic / Foveated / ROI Compression

Sources:

- `strong-primary`: Max Schwarz, Sven Behnke, "Foveated Compression for
  Immersive Telepresence Visualization", IEEE TELEPRESENCE 2025,
  `arXiv:2510.19848`, DOI `10.48550/arXiv.2510.19848`.
  <https://arxiv.org/abs/2510.19848>
- `strong-primary`: Jona Balle, Luca Versari, Emilien Dupont, Hyunjik Kim,
  Matthias Bauer, "Good, Cheap, and Fast: Overfitted Image Compression with
  Wasserstein Distortion", CVPR 2025, pp. 23259-23268,
  `arXiv:2412.00505`, DOI `10.48550/arXiv.2412.00505`.
  <https://openaccess.thecvf.com/content/CVPR2025/html/Balle_Good_Cheap_and_Fast_Overfitted_Image_Compression_with_Wasserstein_Distortion_CVPR_2025_paper.html>
  <https://arxiv.org/abs/2412.00505>
- `supporting-primary`: I. Dror, O. Hadar, "Optimizing traffic signs and lights
  visibility for the teleoperation of autonomous vehicles through ROI
  compression", 2024 preprint, DOI `10.48550/arXiv.2404.02481`.
  <https://arxiv.org/abs/2404.02481>
- `supporting-primary`: Itai Dror, Ofer Hadar, "Improved Perceptual Quality of
  Traffic Signs and Lights for the Teleoperation of Autonomous Vehicle Remote
  Driving via Multi-Category Region of Interest Video Compression", Entropy
  27(7):674, 2025, DOI `10.3390/e27070674`.
  <https://doi.org/10.3390/e27070674>
- `supporting-primary`: Vadivel Shanmugam, B. Uma Maheswari, "A Semantic-Aware
  Compression Strategy for Intelligent Vehicles", Procedia Computer Science
  258:2544-2553, 2025, DOI `10.1016/j.procs.2025.04.516`.
  <https://www.sciencedirect.com/science/article/pii/S1877050925016205>

Local modules/files this should inform:

- `src/tac/hyperbolic_foveation.py`
- `src/tac/balle_sensitivity_weighted.py`
- `src/tac/component_sensitivity_artifact.py`
- `src/tac/sensitivity_map.py`
- `src/tac/submission_archive.py`
- `src/tac/preflight.py`
- `src/tac/optimization/field_equation_planner.py`
- Archive members: `foveation_params.bin`, foveation/ROI selectors if any

Local variables/contracts:

- `_FOVEATION_MAGIC = b"HFV1"`
- `alpha`, `radius`, `power`, `origin`, `frame_indices`
- `expected_seg_dist_delta`, `expected_pose_dist_delta`, `byte_delta`
- `interaction_assumptions`, `byte_allocator_lambdas`
- `foveation_field_manifest`

Engineering mapping:

- The foveated telepresence source supports spatial rate allocation and
  block-wise quality modulation, but the contest does not have eye tracking.
  The contest-safe analog is a deterministic foveation/ROI field derived from
  fixed inputs, or a charged `foveation_params.bin`.
- Balle et al. supports perceptual transport / Wasserstein-style loss ideas,
  but local use must be scorer/component-aware, not human-rating-aware.
- Autonomous ROI compression papers support class-tiered quality allocation,
  especially road, lane markings, movable actors, traffic signs/lights analogs,
  and background/undrivable regions.

Gaps/risks before any score claim:

- Human perception, PSNR, and telepresence immersion metrics do not transfer to
  SegNet/PoseNet.
- A foveation field can easily preserve perceived RGB while changing scorer
  features; exact CUDA component gates are mandatory.
- Any foveation/ROI parameters used at inflate must be archive members or fixed
  deterministic runtime code.
- `src/tac/hyperbolic_foveation.py` is a local operator; paper citations do not
  prove its numerical invertibility, scorer safety, or archive readiness.

### Learned Video / Neural Representation Compression

Sources:

- `strong-primary`: Chun Zhang, Heming Sun, Jiro Katto, "FLAVC: Learned Video
  Compression with Feature Level Attention", CVPR 2025, pp. 28019-28028.
  <https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_FLAVC_Learned_Video_Compression_with_Feature_Level_Attention_CVPR_2025_paper.html>
- `strong-primary`: Jun Zhu, Xinfeng Zhang, Lv Tang, JunHao Jiang, "MSNeRV:
  Neural Video Representation with Multi-Scale Feature Fusion", arXiv submitted
  2025-06-18, DOI `10.48550/arXiv.2506.15276`.
  <https://arxiv.org/abs/2506.15276>
- `strong-primary`: Jiajun He, Zongyu Guo, Zhaoyang Jia, Xiaoyi Zhang, Jiahao
  Li, Xiao Li, Bin Li, Jose Miguel Hernandez-Lobato, Yan Lu, "Compression as
  Adaptation: Implicit Visual Representation with Diffusion Foundation Models",
  arXiv v2 2026-05-01, DOI `10.48550/arXiv.2603.07615`.
  <https://arxiv.org/abs/2603.07615>

Local modules/files this should inform:

- `src/tac/hnerv_wavelet_apply_transform.py`
- `src/tac/self_compressing_nn.py`
- `src/tac/neural_weight_codec_sensitivity.py`
- `src/tac/mdl_bayesian_codec.py`
- `src/tac/mask_entropy_coder.py`
- `src/tac/arithmetic_qint_codec.py`
- `src/tac/submission_archive.py`
- `src/tac/optimization/entropy_rate_decomposition.py`
- `src/tac/optimization/bayesian_experimental_design.py`

Local variables/contracts:

- `decoder_entrypoint`, `bytes_charged`, `model_byte_delta`
- `conditional_entropy_floor_bits`, `best_conditional_model_label`
- `description_length_delta_bytes`, `expected_improvement_minimize`
- Archive members for learned weights, latents, entropy tables, side info

Engineering mapping:

- FLAVC supports testing feature-level attention and global-context entropy
  modeling, but local implementation must preserve deterministic decode and
  scorer/runtime constraints.
- MSNeRV supports multi-scale feature fusion and temporal-window/GoP structure
  for NeRV/HNeRV-family lanes.
- Compression-as-adaptation supports overfit/adaptation and MDL/Bayesian
  framing, but a frozen foundation model is a severe contest-compliance risk
  unless it is fixed contest code or fully charged.

Gaps/risks before any score claim:

- Published RD curves on HEVC/UVG/image benchmarks are not contest evidence.
- Foundation-model sidecars or network downloads are non-compliant.
- Any learned entropy model, feature table, LoRA/adaptation vector, or decoder
  must be deterministic, charged, manifestable, and runnable within inflate
  budget.

### Fridrich / Yousfi / Stego Priors

Sources:

- `strong-primary`: Yassine Yousfi publication page. Latest listed Yousfi
  publication there is 2023; the 2022 detector-informed batch steganography
  paper is with Yousfi, Eli Dworetzky, and Jessica Fridrich.
  <https://yassineyousfi.github.io/publications/>
- `strong-primary`: Binghamton DDE page lists current members Jessica Fridrich,
  Eli Dworetzky, Edgar Kaziakhmedov, affiliates Patrick Bas and Remi Cogranne,
  and former affiliate Yassine Yousfi.
  <https://dde.binghamton.edu/yousfi/index.php>
- `strong-primary`: Eli Dworetzky, Jessica Fridrich, "Secure Payload Scaling in
  Detector-Informed Batch Steganography: The Mismatched Detectors Case",
  IH&MMSec 2025, DOI `10.1145/3733102.3733134`.
  <https://ws2.binghamton.edu/fridrich/Research/mismatched_scaling_IHMMSEC_final.pdf>
  <https://doi.org/10.1145/3733102.3733134>
- `strong-primary`: Edgar Kaziakhmedov, Jessica Fridrich, Patrick Bas, "Effect
  of Acquisition Noise Outliers on Steganalysis", IH&MMSec 2025, pp. 164-173,
  DOI `10.1145/3733102.3733131`.
  <https://ws2.binghamton.edu/fridrich/Research/LIEs_final_v2_pdfa.pdf>
  <https://doi.org/10.1145/3733102.3733131>
- `canonical-stale`: Tomas Filler, Jan Judas, Jessica Fridrich, "Minimizing
  Additive Distortion in Steganography Using Syndrome-Trellis Codes", IEEE
  TIFS 2011, DOI `10.1109/TIFS.2011.2134094`.
  <https://doi.org/10.1109/TIFS.2011.2134094>
- `canonical-stale`: Vojtech Holub, Jessica Fridrich, Tomas Denemark,
  "Universal distortion function for steganography in an arbitrary domain",
  EURASIP Journal on Information Security 2014, DOI
  `10.1186/1687-417X-2014-1`.
  <https://link.springer.com/article/10.1186/1687-417X-2014-1>
- `canonical-stale`: Jessica Fridrich, Miroslav Goljan, David Soukal,
  "Perturbed Quantization Steganography with Wet Paper Codes", MM&Sec 2004,
  DOI `10.1145/1022431.1022435`.
  <https://doi.org/10.1145/1022431.1022435>

Local modules/files this should inform:

- `src/tac/fridrich.py`
- `src/tac/stc_boundary_codec.py`
- `src/tac/mask_entropy_coder.py`
- `src/tac/sensitivity_map.py`
- `src/tac/component_sensitivity_artifact.py`
- `src/tac/optimization/meta_lagrangian_allocator.py`
- `src/tac/optimization/field_equation_planner.py`
- `src/tac/optimization/research_basis.py`

Local variables/contracts:

- `cost_map`, `compute_pixel_cost_map`, `fridrich_constrained_optimize`
- `selector_logits`, `conditional_groups`, `symbol_counts`
- `expected_total_score_delta`, `expected_seg_dist_delta`,
  `expected_pose_dist_delta`, `byte_delta`
- `confidence`, `frechet_derivatives`, `interaction_kernel`
- `NON_RANKABLE_EVIDENCE_GRADES`

Engineering mapping:

- Detector-informed batch steganography maps cleanly to scorer-aware atom
  allocation: spend bytes where the detector/scorer is least sensitive, but
  treat this as planning until archive evidence exists.
- LIEs support searching for locally influential cover/scorer elements and
  false-alarm style vulnerabilities. In contest terms, this should inform
  sensitivity maps and wet/dry atom selectors, not direct score claims.
- STC, UNIWARD, and wet-paper sources remain the canonical structure for
  additive distortion, syndrome-style rounding, and selection-channel custody.

Gaps/risks before any score claim:

- Yousfi's own current publication page did not show a 2025/2026 Yousfi paper;
  use 2025 Dworetzky/Kaziakhmedov/Fridrich work as lineage continuation, not as
  Yousfi authorship.
- Image/JPEG steganography assumptions are not video/SegNet/PoseNet evidence.
- Additive distortion can fail under non-additive stacked interactions; every
  stacked atom needs exact archive replay.
- Any selection channel not derivable from fixed contest inputs or charged
  bytes is a sidecar risk.

### Categorical Masks / Openpilot / Comma Priors

Sources:

- `strong-primary`: comma10k README, official commaai GitHub. It defines 10,000
  PNGs from the comma fleet and the internal SegNet category colors/IDs:
  road, lane markings, undrivable, movable, my car, and an interior-only sixth
  class for `imgsd`.
  <https://raw.githubusercontent.com/commaai/comma10k/master/README.md>
- `strong-primary`: comma.ai "Crowdsourced Segnet" blog, 2020. It states the
  openpilot/comma SegNet has five labels grouped as road/lane/undrivable,
  movable, and my car, and that road/lane labels matter for car tracks.
  <https://blog.comma.ai/crowdsourced-segnet-you-can-help/>
- `strong-primary`: comma2k19 official GitHub and paper. It provides driving
  video, CAN, IMU, GNSS, and camera pose data for pose/mapping development.
  <https://github.com/commaai/comma2k19>
  <https://arxiv.org/abs/1812.05752>
- `strong-primary`: Nam Nguyen, Thuan Nguyen, Thinh Nguyen, Bella Bose,
  "Universal Rate-Distortion-Classification Representations for Lossy
  Compression", arXiv v2 2025-04-22, DOI `10.48550/arXiv.2504.09025`.
  <https://arxiv.org/abs/2504.09025>
- `strong-primary`: Nam Nguyen, Thinh Nguyen, Bella Bose,
  "Rate-Distortion-Classification Representation Theory for Bernoulli Sources",
  arXiv submitted 2026-01-17, DOI `10.48550/arXiv.2601.11919`.
  <https://arxiv.org/abs/2601.11919>

Local modules/files this should inform:

- `src/tac/semantic_label_contract.py`
- `src/tac/categorical_compression_contract.py`
- `src/tac/learnable_class_targets.py`
- `src/tac/mask_grayscale_lut.py`
- `src/tac/mask_entropy_coder.py`
- `src/tac/stc_boundary_codec.py`
- `src/tac/submission_archive.py`
- `src/tac/optimization/field_equation_planner.py`

Local variables/contracts:

- `CONTEST_SEGNET_CLASSES`
- `CONTEST_SEGNET_CLASS_NAMES`
- `CONTEST_SEGNET_COMMA10K_COLORS`
- `SELFCOMP_CLASS_TO_GRAY`
- `SEMANTIC_QUANTIZATION_DEFAULT_BITS`
- `build_categorical_compression_contract()`
- `class_targets.fp16`
- `conditioning_families["openpilot_priors"]`

Engineering mapping:

- comma10k/openpilot labels are the authoritative semantic contract for
  category-aware compression. Keep one-based comma10k IDs and zero-based
  contest tensors explicit.
- RDC/Bernoulli sources support class-task-aware compression math for masks:
  Hamming/categorical distortion, classification-preserving representations,
  and universal encoder tradeoffs.
- Openpilot/comma priors should first be used for proposal ranking and atom
  selection; inflate-time use requires charged payload and no-op controls.

Gaps/risks before any score claim:

- comma/openpilot labels are source-of-truth semantics, not a scorer oracle.
- Any label remap, class target, or conditioning weight must be carried in the
  archive or fixed in audited contest runtime.
- RDC theory is binary/Gaussian/MNIST/SVHN-style in the cited papers; local
  five-class 1200-frame mask streams need their own byte/roundtrip evidence.
- Label permutation, decode/reencode identity, and runtime-consumption controls
  must fail closed before exact eval dispatch.

## Sources Too Weak, Stale, Or Speculative

- DeepWiki, Moonlight, Reddit, ResearchGate, and generic AI paper summaries are
  discovery-only. Do not use them as implementation authority.
- Yousfi 2020-2023 sources are highly relevant lineage but are not "latest"
  2025/2026 research. Use them to preserve detector/categorical priors, not to
  claim new frontier evidence.
- STC, wet-paper, and UNIWARD sources are canonical-stale. They justify local
  contracts for additive distortion, selection-channel custody, and
  entropy/rounding structure; they do not validate any current archive.
- Human foveation, telepresence immersion, PSNR, LPIPS, Wasserstein perception,
  and standard RD curves are non-promotable for the contest until translated
  into SegNet/PoseNet component evidence on exact archive bytes.
- Foundation-model adaptation papers are high-upside concept sources but are
  contest-risky until dependency closure, charged bytes, runtime budget, and
  deterministic decode are proven.

## Hard Evidence Blockers Before Promotion

Every source above is blocked from score/promotion until all of the following
are true:

1. Exact archive bytes exist with SHA-256 and deterministic manifest.
2. Inflate uses no uncharged sidecars, network installs, or scorer loads.
3. Any new mask, pose, foveation, entropy, codebook, or openpilot-prior payload
   is either fixed contest code or charged archive bytes.
4. Local no-op controls prove the intended payload is consumed and that
   decode/reencode or label-permutation controls fail closed.
5. Component distances are recomputed from `contest_auth_eval.json`, not human
   logs.
6. CUDA auth eval is run through the canonical archive path before any score
   claim.
7. Stacked interactions are evaluated as a stacked archive; additive atom math
   alone is planning-only.
