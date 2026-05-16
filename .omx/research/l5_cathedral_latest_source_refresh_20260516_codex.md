# L5/Cathedral latest source refresh - Codex 2026-05-16

Tag: `research_only=true`. No score claim. No dispatch. This ledger records
current paper/OSS signal found during the L5/Cathedral hardening pass and binds
it to concrete Pact actions. It does not supersede
`l5_cathedral_paper_fidelity_review_20260516_codex.md`; it extends it with
newer or under-used source anchors.

## Newer source anchors

| Source | Source signal | Pact action |
|---|---|---|
| PNVC, AAAI 2025, `https://ojs.aaai.org/index.php/AAAI/article/view/32315` | Practical INR video codec that combines autoencoder-style and overfitted INR approaches, with structural reparameterization, hierarchical quality control, modulation-based entropy modeling, and scale-aware positional embeddings. | Treat as a direct L5/C2 design prompt: a floor-breaking Pact lane should combine representation, entropy model, and decode-speed discipline instead of appending a score-aware loss to the old scaffold. Gate on T4 decode time and byte-closed archive grammar. |
| SNeRV, ECCV 2024/arXiv 2025, `https://arxiv.org/abs/2501.01681` | Uses wavelet/frequency decomposition to handle NeRV spectral bias, encoding low-frequency components while reconstructing high-frequency detail with specialized modules. | Reopen wavelet residual / spectral split lanes as candidate L5 stack components. Required proof: the HF stream must be charged, consumed by inflate, and compared against PR95/PR101 on the same contest axis. |
| MetaNeRV, AAAI 2025, `https://arxiv.org/abs/2501.02427` | Meta-learned initialization plus spatial-temporal guidance to accelerate fitting many videos; progressive temporal fitting is the key transferable idea. | Use comma10k19/pretraining only as an initialization/curriculum experiment, not a score claim. Required proof: paired same-architecture from-scratch vs pretrained timing/score smoke, then byte-closed export. |
| C3 official OSS, `https://github.com/google-deepmind/c3_neural_compression` | Low-complexity overfitted image/video compression with explicit bpp/distortion logging, per-video training, and Apache-2.0 code. | Use C3 as an OSS/DX and low-complexity decoder benchmark. Do not import code blindly; clean-room candidate must report MACs or measured decode wall-clock plus Pact scorer-aware objective. |

## Immediate design consequences

1. **Break the old shared scaffold when it suppresses the method.** PNVC and C3
   both support the operator's unique-and-complete-per-method directive:
   representation, entropy model, archive grammar, and decoder contract should
   be co-designed rather than forced through the A1/Z3 helper path.
2. **Frequency split is not optional for L5.** SNeRV makes the spectral-bias
   failure mode explicit. TT5L, D4, and PR95-family extensions should carry an
   explicit LF/HF or wavelet/spectral ablation before declaring the basin
   saturated.
3. **Pretraining is a speed/curriculum hypothesis first.** MetaNeRV-style
   initialization can reduce wall-clock and possibly avoid local minima, but it
   is not promotable until the final packet is a normal scorer-free contest
   archive with no hidden training data dependency at inflate time.
4. **Low-complexity decode is a first-class score-lowering constraint.** Rich
   autoregressive or hierarchical models are attractive only if they survive a
   measured T4 decode budget. Every Cathedral-ranked literature anchor should
   carry a `decode_complexity_evidence` field before dispatch.

## Guardrail for Cathedral/autopilot rows

Add or preserve these metadata fields whenever a literature anchor drives a
lane ranking:

```json
{
  "source_supports": "paper/OSS claim in its native metric",
  "pact_transfer_hypothesis": "why it should affect SegNet/PoseNet/archive bytes",
  "pact_must_prove": [
    "consumed_bytes_mutation_changes_output",
    "scorer_free_inflate",
    "exact_archive_sha256_and_bytes",
    "runtime_tree_custody",
    "paired_contest_cpu_cuda_eval_before_promotion",
    "decode_complexity_evidence"
  ],
  "paper_claim_scope": "analogy|derivation|empirical_external|pact_empirical"
}
```

Missing `pact_transfer_hypothesis` or `pact_must_prove` should make a source
useful for brainstorming but not for autonomous dispatch authority.
