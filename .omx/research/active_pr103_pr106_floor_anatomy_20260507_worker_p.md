# Active PR103-on-PR106 Floor Anatomy - Worker P

This ledger records byte/custody anatomy only. It does not claim score from logs or eval JSON.

## Active Archive

- path: `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip`
- bytes: `185578`
- sha256: `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- zip file members: `1`
- zip overhead vs stored payload bytes: `108`

## Charged Byte Proof

- charged_archive_bytes: `185578`
- charged_archive_sha256: `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- identity_sources_match: `True`
- score_claim: `False`

## Packed Payload

- `packed_header_ff_len24` payload[0:4] 4B sha256 `1e1042fb1a764966cc07425de7fbabc966e578afef8516521540c9115945e87d`
- `decoder_pr103_ac_bytes` payload[4:169621] 169617B sha256 `854278d7bb049a59b44a0fa85cbb849752ba84f02fbd7d91480c1a1ffcac42e5`
- `latents_pr106_fixed_brotli` payload[169621:185470] 15849B sha256 `94257b33cf3083c5daa0f3b1e127cb7c51bee42a6416b19763eea7bf9ecc3c32`

## Nested Decoder Sections

- `decoder.scales_fp16` payload[4:60] 56B sha256 `0152d84bd39477ad9a6394107b9cc0c85027a5e33ac50e6ac0023e0f45cd14e8`
- `decoder.br` payload[60:7252] 7192B sha256 `96c6fb01d0fb5f9f7116fb27e6fd1e831bf772cc272ca562ea61b68d6d4f877c`
- `decoder.hists` payload[7252:8241] 989B sha256 `3499b2305afc5fd5e4d048560f5cd2052ee5db0068cca73dbbd98a5122eee5bd`
- `decoder.merged_ac` payload[8241:169621] 161380B sha256 `1ad4e883ac8fef97af7f07d24aff790e5fd2d52cb05f5a4ee3a58e4e4b7b315c`
- `decoder.hi_hist` payload[169621:169621] 0B sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `decoder.ac_fallback` payload[169621:169621] 0B sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Exact Eval Identity

- auth eval artifact: `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`
- exact_eval_blockers: `[]`
- score_fields_suppressed: `['avg_posenet_dist', 'avg_segnet_dist', 'canonical_score', 'final_score', 'rate_unscaled', 'reported_final_score_display_rounded', 'score_pose_contribution', 'score_rate_contribution', 'score_recomputed_from_components', 'score_reported_rounded_differs_from_canonical', 'score_rounding_abs_delta', 'score_seg_contribution']`
- runtime_tree_sha256: `54db9e5ddee85ae7f486fae900ff3907932efb1c8d3062bc264b0e5c7456d8f6`

## PR100-107 Intake Coverage

- PR100: archive_present=`True` bytes=`178981` sha256=`afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641` title=`hnerv_lc_v2 submission (0.1954)`
- PR101: archive_present=`True` bytes=`178258` sha256=`b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` title=`add hnerv ft microcodec submission`
- PR102: archive_present=`True` bytes=`276481` sha256=`03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746` title=`hnerv_lc_v2_scale095_rplus1 submission (0.19538 CPU)`
- PR103: archive_present=`True` bytes=`178223` sha256=`31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30` title=`hnerv_lc_ac submission (0.19)`
- PR104: archive_present=`True` bytes=`178637` sha256=`6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8` title=`qhnerv_ft_best`
- PR105: archive_present=`True` bytes=`177857` sha256=`597ba0732810eba08cdae619b679d211d398bc0249b8831898f7096d5beece1d` title=`kitchen_sink (0.19797)`
- PR106: archive_present=`True` bytes=`186239` sha256=`3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58` title=`belt_and_suspenders (0.20946)`
- PR107: archive_present=`True` bytes=`178392` sha256=`7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb` title=`apogee submission (0.2293)`
