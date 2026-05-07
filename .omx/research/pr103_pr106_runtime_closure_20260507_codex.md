# PR103 -> PR106 Runtime Closure Proof (2026-05-07)

Scope: PR103 arithmetic-coded decoder bytes spliced into the PR106
`0xff | u24(decoder_len) | decoder | fixed_latents` envelope.

Proof tool:

```bash
.venv/bin/python tools/prove_pr103_pr106_runtime_closure.py
```

Current artifact:

- `experiments/results/pr103_repack_pr106_standalone_20260507/runtime_closure.json`
- score claim: false
- evidence grade: `empirical_runtime_closure`
- candidate archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- decoder section SHA-256:
  `854278d7bb049a59b44a0fa85cbb849752ba84f02fbd7d91480c1a1ffcac42e5`
- decoder section bytes: `169617`
- section lengths:
  `br=7192`, `hists=989`, `merged_ac=161380`, `hi_hist=0`,
  `ac_fallback=0`
- `ac_fallback_set=[]`

The closure helper fails closed on decoder SHA mismatch, latents SHA mismatch,
section-length sum mismatch, fallback-section / `ac_fallback_set` mismatch, and
fallback indices outside PR103 `AC_TENSOR_INDICES`.

Remaining blockers before exact CUDA:

1. Copy or generate the runtime adapter into the final submission runtime tree.
2. Close `brotli` / `constriction` dependency custody inside that runtime.
3. Run `scripts/pre_submission_compliance_check.py --contest-final --strict`.
4. Claim the dispatch lane before exact CUDA auth eval.
5. Run `archive.zip -> inflate.sh -> upstream/evaluate.py` on CUDA.
