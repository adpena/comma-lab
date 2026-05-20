# Frontier‑Lowering Staircase

The following roadmap decomposes the challenge of reducing the contest score into
concrete, bite‑sized tasks.  It treats CPU, CUDA, local macOS, and cloud
resources as separate axes and emphasizes reproducibility, parallelism and
evidence gates.  The staircase is organized by time horizon and prioritised by
expected value per wall‑clock hour.

## Short‑term (24–72 hours)

These tasks require little or no GPU and can be run on a local Mac or a modest
CPU cloud instance.  They focus on validating the current submission,
establishing reproducible baselines, and preparing the ground for more costly
experiments.

| Task | Why it matters | Actions | Expected outputs | Dependencies | Promotion gate |
|---|---|---|---|---|---|
| **S1. Recompute PR #110 archive and scores** | Establish a local baseline before exploring new models.  Ensures that the reported score is reproducible. | (1) Download `archive.zip`; verify SHA‑256; list members; (2) run `inflate.sh` and `upstream/evaluate.py` on CPU and (if available) MPS; record scores; (3) compare to PR numbers【294212394795766†L354-L357】. | Confirmed scores; `contest_auth_eval.json` for CPU and MPS; a mini report summarising any drift. | Upstream evaluator commit; `torch`, `numpy`, `brotli`. | Match scores within 0.001 on CPU; if drift >0.001, investigate before moving on. |
| **S2. Construct deterministic packet compiler** | Many future experiments will emit new archives.  A deterministic compiler ensures reproducible bytes and simplifies SHA custody【58280996536521†L200-L206】. | Build a script that takes a candidate payload, sidecar and metadata and emits a zip with exact member ordering and compression flags.  Use it to reproduce PR #110’s archive. | `packet_compiler.py`; reproducible `archive.zip` identical to the official one; CLI documentation. | S1. | Passes SHA match; accepts deterministic inputs; writes manifest. |
| **S3. Audit existing HNeRV variants** | The HNeRV family is the control arm.  Before exploring other directions, collect an exact scoreboard of PR #95, #98, #100, #101, #102, #103 and #110. | For each PR: reproduce the archive (via `packet_compiler`), run CPU and CUDA evaluations on the same hardware, and compute deltas. | Table of HNeRV variants with exact scores and archive sizes; update to the candidate inventory. | Access to PR archives or ability to rebuild them; upstream evaluator. | Table complete; CPU and CUDA axes computed on same machine; differences accounted for. |
| **S4. Byte‑profile PR #110** | Identify which bytes in the selector payload and source payload contribute most to the score; informs later compression attempts. | Instrument the evaluator to record per‑frame distortions; compute gradients via finite differences on the selector bits; produce a heatmap. | Heatmap of per‑byte sensitivity; list of highly sensitive bits; a note on which transforms contribute most. | S1; ability to run local evaluation and mutate bytes. | Clear identification of sensitive bits; cost–benefit analysis of selector complexity. |
| **S5. Static audit of `comma‑lab` source map and inventory** | Understand which research avenues have scaffolded tooling vs. design notes【58280996536521†L18-L32】.  Guides mid‑term planning. | Read `comma‑lab/docs/full_stack_source_map.md` and the candidate inventory; classify each row as EMPIRICAL, SCAFFOLDED, LANDED TOOLING or DESIGN‑ONLY; extract local surfaces and missing pieces. | Annotated spreadsheet summarising each candidate; list of local modules to explore; open issues in `comma‑lab` repository for missing scaffolds. | None. | Completed table with classification and pointers. |
| **S6. Prep local compute environment** | Many experiments can run on local CPU or MPS.  A reproducible environment ensures consistent results. | Set up conda/virtualenv; install exact versions of `torch`, `numpy`, `brotli`; clone `comma_video_compression_challenge`, `comma‑lab` and `tac`; run all unit tests in `tac`【635462165059268†L14-L24】. | Verified local environment; baseline runtime for `tac` examples; list of failed tests (if any). | Internet access to clone repos; Mac hardware. | All tests pass; environment documented in a `requirements.txt` or `conda` file. |

## Mid‑term (1–3 weeks)

These tasks involve running small GPU smokes and implementing new features in
`comma‑lab` or `tac`.  They should be pursued in parallel where possible.  Each
lane includes a “kill criteria” to abort if early results are negative.

| Task | Hypothesis | Local surface & missing implementation | Cheap local smoke | Required GPU smoke | Expected artifact | Promotion gate | Kill criteria |
|---|---|---|---|---|---|---|---|
| **M1. HNeRV variant sweep (E‑NeRV/HiNeRV)** | Changing the network architecture may change the representation and reduce residual entropy. | `comma‑lab` has scaffolds for HiNeRV/E‑NeRV; need export wrappers compatible with `packet_compiler`. | Train a tiny E‑NeRV on a single video using CPU/MPS to ensure the code runs; verify export. | Train on 1‑2 videos on a T4; export archive; run evaluator. | Scores for each variant; delta vs. HNeRV baseline; runtime logs. | If any variant yields ≥0.0005 CPU improvement at similar byte budget, move to full training on all videos. | Kill if no improvement after smoke; if training diverges or export fails. |
| **M2. FEC7/FEC8 selectors** | Extending the per‑frame transform palette (K>16) may capture more frame variations at the cost of a larger sidecar. | Use `frame_selector.py` to support 63‑mode and 127‑mode palettes; implement offline search; extend fixed‑Huffman tables. | Build new selector tables; run local search on a single video; measure per‑frame distortion improvements. | Train FEC7 on 3 videos; export archive; compute CPU score delta vs. FEC6; evaluate sidecar size increase. | Table of K vs. score and bytes; recommendation. | Promote if CPU score improves ≥0.0003 net of byte cost. | Kill if improvements are <0.0001 or sidecar size overwhelms benefits. |
| **M3. SIREN/coordinate INR codec** | Periodic activation functions can encode high‑frequency details more efficiently than ReLU networks【58280996536521†L151-L158】. | `SIRENVideoCodec` scaffold exists; need training scripts and export wrapper. | Train a SIREN model on a 16‑frame segment using CPU; verify that the INR produces recognisable frames. | Train on a single video on a 4090 or A100; export; measure CPU/CUDA scores. | Archive with INR representation; comparison table vs. HNeRV; training logs. | Promote if INR archive yields ≥0.001 CPU improvement or similar at lower byte budget. | Kill if training cost is prohibitive or archive bytes explode. |
| **M4. VQ‑VAE discrete latent codec** | Discrete tokens facilitate entropy coding and may lower byte counts【58280996536521†L159-L163】. | `comma‑lab` has VQ‑VAE scaffolds; missing exact export and sidecar grammar. | Train a tiny VQ‑VAE on low‑resolution frames using CPU; ensure token indices can be exported. | Train on a subset of videos on a T4; export tokens; run arithmetic/Huffman coder; evaluate CPU score. | Archive with discrete latent representation; RD curve for token vocab sizes. | Promote if CPU score drops ≥0.002 at comparable bytes or if tokens compress significantly better. | Kill if token complexity leads to worse scores or training fails to converge. |
| **M5. Compression‑aware pretraining (MAE/VideoMAE)** | Pretraining on domain‑specific data can provide a better starting point and reduce training cost【58280996536521†L65-L79】. | Build scripts to load MAE or VideoMAE pretrained weights into HNeRV or other networks; freeze some layers. | Use local CPU to verify weight loading and that the network trains without NaNs. | Train on a subset of videos on a 4090; compare training speed and final score vs. random init. | Training logs; exported archives; table of performance vs. baseline. | Promote if pretraining reduces GPU hours or improves final scores. | Kill if pretraining overhead outweighs benefits or if export cost increases. |
| **M6. Foveation mask and wavelet allocation** | Non‑uniform spatial allocation may save bytes in less important regions【58280996536521†L116-L125】. | Implement `foveation_field` and `wavelet_residual` in `tac` or `comma‑lab`; create masks around vanishing point. | Run a CPU‑only smoke: apply a down‑sampling mask to the residual; ensure training and export pipelines handle variable resolution. | Train with a foveated decoder on a GPU; export archive; measure SegNet and PoseNet distortion differences. | Archive with foveated frames; distortion difference vs. baseline. | Promote if byte savings exceed the resulting distortion penalty. | Kill if segmentation and pose distortions increase too much or if complexity is high. |
| **M7. Predictor calibration and refusal analysis** | The `tac` predictor with refusal modes helps prioritise candidates and avoid wasted GPU spend【635462165059268†L118-L142】. | Use the predictor on the HNeRV variant sweep results; identify calibration gaps; tune refusal thresholds. | Evaluate predictor on CPU using existing anchors; record false accept/reject rates. | None required (predictor is CPU). | Updated predictor thresholds; improved ranking accuracy. | Promote if false accept/reject rates drop; integrate into search loops. | Kill if predictor continues to misclassify or over‑refuse candidates. |
| **M8. RAFT/LA‑Pose pose prior** | Better optical‑flow and pose priors may reduce PoseNet error【58280996536521†L104-L115】. | Integrate RAFT or LA‑Pose into training as an auxiliary loss; scaffolds exist but need export wrappers. | Run CPU‑only smoke with a tiny RAFT network to ensure integration; no GPU yet. | Train on 1–2 videos on a 4090; export; measure pose distortion improvement and byte cost. | Archive with pose‑prior‑trained decoder; table of pose error vs. baseline. | Promote if pose distortion decreases enough to lower total score by ≥0.001 at similar byte count. | Kill if RAFT integration slows training too much or if improvements are negligible. |

## Long‑term (1–3 months)

These tasks involve significant engineering, multiple GPU runs or new research
directions.  They should be pursued only after the mid‑term tasks identify
promising directions.

| Task | Hypothesis | Key workstreams | Estimated compute/cost | Main risks | Reactivation criteria |
|---|---|---|---|---|---|
| **L1. Neural codec integration (C3, Cool‑Chic, DCVC‑FM, Balle hyperprior)** | Modern learned video codecs may encode dashcam content more efficiently than HNeRV【58280996536521†L52-L63】. | Integrate C3 or Cool‑Chic models into `comma‑lab`; build export wrappers; implement sidecar grammar; train on entire dataset; evaluate. | 1–2 weeks of GPU time on A100/4090 per codec; cost $100–$200. | Large training cost; may require memory beyond local resources; risk of non‑convergent training. | Proceed only if mid‑term tasks show that new representations (e.g. SIREN or VQ‑VAE) beat HNeRV. |
| **L2. World‑model priors (DreamerV3, V‑JEPA 2, Mamba‑2)** | Predictive coding may reduce residual entropy by modelling temporal dynamics【58280996536521†L80-L88】. | Implement latent world models (RSSM, V‑JEPA, Mamba) that predict future frames; use them as priors during compression; build side‑information proofs. | Multi‑month research; requires GPUs with large memory; integration complexity high. | High compute and algorithmic complexity; untested for small dashcam segments; risk of mis‑specifying the prior. | Revisit only if other representation changes saturate and compute budget is available. |
| **L3. Semantic conditioning (SPADE, CLADE)** | Conditioning on segmentation masks may allow the decoder to allocate bytes to important classes【58280996536521†L126-L133】. | Implement SPADE or CLADE layers in the decoder; train on semantic labels; modify scorer; build export grammar. | Weeks of GPU training; additional time preparing ground‑truth masks. | Scorer may not reward better semantics directly; complexity of semantic labs. | Pursue if foveation and RAFT priors show strong benefits; otherwise low priority. |
| **L4. Dense feature priors (SAM3, Falcon, DINOv3)** | Dense features from large vision models may act as teachers or priors【58280996536521†L134-L148】. | Distill features from SAM3 or Falcon into the decoder; build compact teacher tokens; integrate into training. | Very high compute and memory; licensing and weight‑download issues. | Models may be too large to include in archive; risk of violating contest rules by requiring side‑information at inflate time. | Only pursue if contest rules evolve to allow large side‑information or if distillation can be done entirely at train time. |
| **L5. Steganography and STC/UNIWARD** | Treat the contest as an inverse steganography problem; design codecs that allocate bits in texture‑adaptive ways【58280996536521†L166-L175】. | Implement UNIWARD cost functions; integrate syndrome‑trellis coding; combine with master‑gradient to allocate bits; build deterministic exporter. | Requires development of binary linear codes and heavy optimisation; run multiple simulations. | Risk of overfitting to scorer; high algorithmic complexity. | Pursue if foveation and RAFT provide minimal gains and if research bandwidth exists. |
| **L6. Composition via ADMM and Dykstra** | Combining multiple proven layers may yield larger gains than any single layer【58280996536521†L207-L213】. | Implement a stack‑of‑stacks recipe framework; coordinate multiple modules (e.g. SIREN + foveation + VQ‑VAE) via ADMM; tune Lagrange multipliers. | High complexity; requires careful tuning; heavy compute. | Risk of incompatible interfaces or training collapse; may require full rewrites. | Attempt only after at least two individual innovations have been proven on the contest axis. |

## Priority order by expected value per wall‑clock hour

1. **Recompute baseline and build deterministic packet compiler (S1–S2).**  These
   tasks pay off immediately for all future work.
2. **Audit and byte‑profile (S3–S4).**  Understanding where the HNeRV family
   spends bytes informs whether to pursue bigger selectors (M2) or different
   architectures (M1, M3, M4).
3. **Prepare environment and read the source map (S5–S6).**  Knowing which
   candidates are scaffolded avoids duplicating work.
4. **Selector extensions and small architecture changes (M2, M1).**  These
   require modest GPU time and directly attack the representation and sidecar.
5. **VQ‑VAE and SIREN experiments (M3, M4).**  They have higher risk but
   potentially higher reward; run in parallel on different GPUs.
6. **Foveation and RAFT priors (M6, M8).**  These may provide complementary
   gains and can be tested concurrently.
7. **Compression‑aware pretraining (M5) and predictor tuning (M7).**  These
   support all other lanes and can run opportunistically when GPUs are idle.
8. **Long‑term research (L1–L6).**  Reserve only if mid‑term efforts plateau
   and compute budget permits.

## Local vs. cloud resource recommendations

* **Local CPU/MPS:** Use for recomputing baselines, running unit tests, small
  smokes for new selectors or quantization schemes, predictor calibration,
  environment setup and static analysis.
* **Cloud T4/4090:** Use for mid‑term experiments requiring moderate GPU time
  (e.g. small HNeRV variants, VQ‑VAE smokes, foveation prototypes).  Use
  spot instances or reserved hours to control cost.
* **Cloud A100:** Use sparingly for full training runs once a candidate has
  passed local and T4 smokes.  Document GPU hours, memory usage and cost
  carefully.
