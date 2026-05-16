# Cathedral Literature Anchor Scope Hardening - 2026-05-16

## Scope

Read-only paper/source-fidelity review found that `literature_anchor` rows in
the substrate composition matrix were too easy to treat as ranking authority.
This landing adds machine-readable scope fields to every long-burn campaign row
that carries a literature anchor:

- `source_supports`
- `paper_claim_scope`
- `pact_must_prove`
- `decode_complexity_evidence`

These fields make the ranker carry the distinction between "paper supports a
principle" and "Pact has proved this packet lowers contest score."

## Rows Hardened

- `z3_balle_hyperprior_bolton`
- `z4_cooperative_receiver_loss`
- `z5_predictive_coding_world_model`
- `time_traveler_l5_autonomy`
- `c1_world_model_foveation`
- `c6_e4_mdl_ibps`

## Source-Fidelity Findings

- Balle-style hyperpriors support side-information priors in learned image
  compression, but not frozen-A1 latent replacement without byte-closed paired
  contest eval.
- Predictive coding and world-model papers support temporal/residual modeling
  hypotheses, but not TT5L/C1/Z5 score bands, foveation multipliers, or byte
  budgets without identity/capacity-matched ablations.
- INR-video sources such as TeCoNeRV, PNVC, and SNeRV argue for co-designed
  representation, entropy model, and decoder complexity. That supports the
  unique-and-complete-per-method directive and argues against blindly forcing
  L5/Z3G2 through A1/Z3 helper assumptions.
- Cool-Chic 5.0 and related overfitted-codec work support measured
  entropy-modeling and decoder-complexity discipline, not unmeasured
  planning-only rank promotion.

## Citation Anchors

- Balle, J., Minnen, D., Singh, S., Hwang, S. J., & Johnston, N. (2018).
  *Variational Image Compression with a Scale Hyperprior*. ICLR.
  https://research.google/pubs/variational-image-compression-with-a-scale-hyperprior/
- Minnen, D., Balle, J., & Toderici, G. D. (2018). *Joint Autoregressive and
  Hierarchical Priors for Learned Image Compression*. NeurIPS 31.
  https://papers.nips.cc/paper_files/paper/2018/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html
- Ladune, T., Philippe, P., Jaffuer, P., Blard, T., Kervadec, S., Henry, F.,
  & Clare, G. (2026). *Cool-Chic 5.0: Faster Encoding and Inter-Feature
  Entropy Modeling for Overfitted Image Compression*. arXiv:2605.02726.
  https://arxiv.org/abs/2605.02726
- Padmanabhan, N., Gwilliam, M., & Shrivastava, A. (2026). *TeCoNeRV:
  Leveraging Temporal Coherence for Compressible Neural Representations for
  Videos*. arXiv:2602.16711. https://arxiv.org/abs/2602.16711
- Gao, G., Kwan, H. M., Zhang, F., & Bull, D. (2025). *PNVC: Towards
  Practical INR-based Video Compression*. AAAI 39(3), 3068-3076.
  https://ojs.aaai.org/index.php/AAAI/article/view/32315
- Kim, J., Lee, J., & Kang, J.-W. (2025). *SNeRV: Spectra-preserving Neural
  Representation for Video*. arXiv:2501.01681.
  https://arxiv.org/abs/2501.01681
- Rao, R. P. N., & Ballard, D. H. (1999). *Predictive Coding in the Visual
  Cortex*. Nature Neuroscience, 2(1), 79-87. doi:10.1038/4580.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_substrate_composition_matrix.py -q`
  -> `51 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py src/tac/tests/test_build_composition_ranking_json.py -q`
  -> `47 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/substrate_composition_matrix.py src/tac/tests/test_substrate_composition_matrix.py`
  -> `All checks passed`

## Remaining Work

- Add a strict preflight gate once all Cathedral ranker input surfaces, not
  just the canonical composition matrix, carry these source-fidelity fields.
- Extend autonomous ranking logic to downweight or block rows where
  `pact_must_prove` remains unsatisfied by byte-closed exact evidence.
