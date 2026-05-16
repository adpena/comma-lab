# L5/Cathedral paper-fidelity review - Codex 2026-05-16

Tag: `research_only=true`. No score claim. No dispatch. This ledger preserves
the read-only source-fidelity review from subagent `019e2e4e-7637-7893-a931-47e6ecdc5dd1`
and converts it into codebase guardrails.

## Standing rule

Every cited paper-backed lane must record two separate facts:

1. `source_supports`: what the paper, OSS project, or article actually supports.
2. `pact_must_prove`: the contest-specific evidence needed before promotion.

For Pact, `pact_must_prove` always includes consumed bytes, scorer-free inflate,
exact archive SHA/bytes, runtime custody, and paired contest-axis eval before a
promotion claim. Paper PSNR, BD-rate, or visual-quality results do not become
Pact score movement without byte-closed archive evidence.

## Source map

| Area | Primary sources | Source supports | Pact must prove |
|---|---|---|---|
| Learned hyperprior codecs | Ballé et al. scale hyperprior: https://research.google/pubs/variational-image-compression-with-a-scale-hyperprior/ ; Minnen, Ballé, Toderici joint autoregressive/hierarchical priors: https://papers.nips.cc/paper_files/paper/2018/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html ; TensorFlow Compression: https://tensorflow.github.io/compression/ ; CompressAI: https://interdigitalinc.github.io/CompressAI/zoo.html | Learned entropy models, hyperprior side information, joint priors, range/ANS-backed coding. | Side-info must replace or condition consumed bytes, not append silently. Require CDF/update/roundtrip/runtime-consumption checks before any Z3/BRV2 claim. |
| NeRV/HNeRV family | NeRV: https://arxiv.org/abs/2110.13903 ; HNeRV CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html ; HNeRV repo: https://github.com/haochen-rye/HNeRV ; Fourier features: https://github.com/tancik/fourier-feature-networks | Full-frame neural video representation and content-adaptive embeddings. | PR95/PR101 remain control/export baselines. Any improvement needs score-aware training, eval-roundtrip, archive grammar, and full-frame inflate proof. |
| Coordinate/hierarchical image codecs | Cool-Chic ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Ladune_COOL-CHIC_Coordinate-based_Low_Complexity_Hierarchical_Image_Codec_ICCV_2023_paper.html ; Cool-Chic repo: https://github.com/Orange-OpenSource/Cool-Chic ; C3 repo: https://github.com/google-deepmind/c3_neural_compression | Overfitted low-complexity coordinate/hierarchical codecs and C3 improvements for still images/videos. | Use clean-room ideas only. Require scorer-aware loss and byte-closed Pact export before claims. |
| Entropy coders | Duda ANS: https://arxiv.org/abs/0902.0271 ; ANS speed/rate: https://arxiv.org/abs/1311.2540 ; constriction: https://github.com/bamler-lab/constriction ; rANS in practice: https://fgiesen.wordpress.com/2015/12/21/rans-in-practice/ ; MacKay ITILA: https://www.fon.hum.uva.nl/rob/Courses/InformationInSpeech/CDROM/Literature/LOTwinterschool2006/www.inference.phy.cam.ac.uk/mackay/itila/book.html | Real entropy coding primitives and rate/throughput tradeoffs. | PMF/CDF custody, exact roundtrip, byte delta, runtime decoder cost, and consumed-stream mutation proof. Entropy coding alone is not a score claim. |
| World models and predictive coding | DreamerV3: https://arxiv.org/abs/2301.04104 ; PlaNet/RSSM: https://planetrl.github.io/ ; PredNet: https://arxiv.org/abs/1605.08104 ; Rao-Ballard context: https://www.nature.com/articles/nn0199_9 | Latent dynamics, RSSM-style posterior/prior learning, and predictive residual modeling. | L5/Time-Traveler needs identity-predictor and capacity-matched ablations so gains are attributed to prediction rather than extra parameters or schedule changes. |
| Motion, foveation, semantic priors | RAFT: https://arxiv.org/abs/2003.12039 ; LA-Pose: https://arxiv.org/abs/2604.27448 ; Geisler-Perry foveation: https://svi.cps.utexas.edu/spie1998.pdf ; DeepFovea: https://github.com/facebookresearch/DeepFovea ; DSSLIC: https://arxiv.org/abs/1806.03348 | Motion/pose/foveation/semantic priors as compression guides. | Treat as compress-time priors unless every side-info byte and decoder dependency is charged and scorer-free at inflate. |
| Efficient coding, IB, side information | Atick-Redlich efficient coding: https://cir.nii.ac.jp/crid/1361981470594781568 ; Information Bottleneck: https://research.google/pubs/the-information-bottleneck-method/ ; Wyner-Ziv: https://cir.nii.ac.jp/crid/1360564063947537280 | Efficient coding, relevance-preserving bottlenecks, and decoder side-information analogies. | Pact cooperative-receiver language remains analogy/objective unless locally re-derived and empirically closed. |

## Overclaim risks to fix or downgrade

- `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md`
  uses `0.150-0.170`, `5-10x`, and `-0.020 to -0.040` as internal predictions.
  These are not paper-supported estimates. Tag as internal hypothesis or attach
  measured Pact ablations.
- `.omx/research/campaign_lane_c1_z6_world_model_foveation_20260514.md`
  includes a `[0.06, 0.10]` cumulative band. Treat as a hypothesis only until
  a byte-closed staged campaign produces exact contest-axis artifacts.
- `src/tac/xray/foveation_ego_motion.py` says "70% of usable visual information
  in 25% of pixels". This needs a Pact video/scorer measurement or citation
  downgrade.
- `src/tac/xray/predictive_coding_hierarchy.py` currently reports a byte budget
  from raw residual tensor bytes, not entropy-coded archive bytes.
- `experiments/train_ffnerv_as_renderer.py` cites Tancik/Fourier-feature support
  but the `1.3x spectral coverage` scalar is not sourced.
- `.omx/research/stc_dasher_arithmetic_maximalism_v1_design_20260515.md`
  has an initial `[-0.010, -0.030]` band that conflicts with the later
  no-score-band correction.
- `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md`
  must not imply Atick-Redlich derives a known-contest-scorer compressor
  theorem. Keep it as analogy unless locally re-derived.
- `.omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md` should
  attach measured histograms and actual packet bytes before repeating the
  `13.4 KB` savings estimate.

## Highest-value guardrail

Add per-lane metadata:

```json
{
  "source_supports": "...",
  "pact_must_prove": [
    "consumed_bytes_mutation_changes_output",
    "scorer_free_inflate",
    "exact_archive_sha256_and_bytes",
    "runtime_tree_custody",
    "paired_contest_cpu_cuda_eval_before_promotion"
  ],
  "paper_claim_scope": "analogy|derivation|empirical_external|pact_empirical"
}
```

This should live beside cathedral/autopilot evidence rows and be checked before
the ranker treats a literature anchor as score authority.

