# PFP16 A++ Evidence Bundle

This bundle is a deploy/paper custody packet for the exact T4 CUDA auth eval.
Score authority is `eval/contest_auth_eval.json`; MPS/CPU/proxy/log-only outputs are not score evidence.

- Recomputed score: `1.043987524793892`
- Archive SHA-256: `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`
- Archive bytes: `686635`
- GPU: `Tesla T4` (`gpu_t4_match=true`)
- Upstream commit: `c5e1274e54e47f81b121bc3bf75eaa9a432b1837`
- Custody manifest: `custody/custody_manifest.json`
- Paper custody note: `custody/PFP16_A_PLUS_PLUS_CUSTODY_NOTE.md`

Legacy remote parser fields are superseded: any quarantined `contest_cuda_score`, `score_delta_vs_lane_g_v3`, `hard_kill_triggered`, or `lane_status=HARD_KILL_REGRESSION` values in build provenance are historical parser output only and are invalid for claims.

Known gap: the remote Lightning staged tree was non-git, so this bundle includes local git/diff state and records the missing remote staged-tree manifest explicitly.
