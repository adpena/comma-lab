# L5 v2 TT5L side-info effect curve Lightning alternate-provider plan

Generated: 2026-05-17T05:54:42Z

This memo preserves the current five-variant Lightning exact-CUDA dry-run signal for
the TT5L side-info effect curve after Modal failed before app creation on the
workspace billing limit.

## Status

- Classification: CUDA dry-run plan only, no score claim.
- Source plan: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json`.
- Modal blocker: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_modal_billing_blocker_20260517_codex.json`.
- Covered axis: `contest_cuda`.
- Missing paired axis: `contest_cpu`; Lightning does not close the paired CPU/CUDA effect-curve requirement by itself.
- Runtime: `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir`.
- Runtime content tree SHA-256: `bf857cb79055e914b64e5cd5788a5a15a37824e01b32c871b84715c153c0ab32`.

The older `.omx/research/l5_v2_tt5l_lightning_alt_provider_plan_20260517_codex.json`
is superseded for this effect curve. It names one stale `random_lsb` archive SHA
`b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7` and the old
recovered exact-eval runtime. Do not use it for the current five-variant packet set.

## Dry-run records

All five local Lightning exact-CUDA dry-runs returned `DRY_RUN` with submit
readiness OK and no submit-readiness blockers. These are ignored result-tree
artifacts, so this ledger records their custody hashes instead of committing the
raw provider files.

| variant | bytes | archive SHA-256 | command SHA-256 | dry-run stdout SHA-256 | state SHA-256 |
| --- | ---: | --- | --- | --- | --- |
| zero | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `37a254d18f2b713f688b04f30f338bbdbc74ca42e35f5af2bcdc9034861145d0` | `c42cbfd798e9a0070f0b4810a025bca9968b43cbb76dd20724cd7490f52b5ed0` | `48b97d743d90bc40137d26bccbcba2ca74138fc1a307a34aa7e526c70f0e846f` |
| random_lsb | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `f588d56de5780c64682e26d56a08538e0a163d11cb69936e8bebaddea6b2674e` | `fa3ecacbe24926175a3705650ee0b8b7eaf8f4b26000d83b7173b1ea06d3b91b` | `14b1b932a08e3106f0e18b09bf8a81484c6c407fe02fdcc36f797ccf80020879` |
| shuffled | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `68f9eac3ef4ccec02a47113d9a0736e74732f5378b7e18cf2bf7f6462f831bf5` | `aeea7910e7315de27473ad6e7d6538e07a671e332b609ef3021df99726f0a89b` | `706481ea12d245ba807081ca4d5f11d95ab00ad8d7af6164f4dd26acba7af7ee` |
| trained | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `d2f4e03790953dc7c631dad155ed01a79e76f200cbd130aa959aabf29d539a63` | `d45d4c96346def7706ec0eeac103fb8586aa8c10e930eca3b186981bbed2a6c9` | `010c8939ea95d81659942bb08ca437f9619aefbceac46d7e1fe4ef03d557a57f` |
| ablated | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `4eef7e4d4ea88ded98ee79b086cf350bd82aa25998482afce7906a7299b51d7e` | `1a4492516b3ad7b083bea7aabf7580979c329eab40419fe3dde1d5a7f1cecd7e` | `685b87737336e8bbc6665b5d12d018937c72696f0418a47f69d3088b4510c9d8` |

## Blockers

- `LIGHTNING_SSH_TARGET` is not configured.
- `LIGHTNING_TEAMSPACE` is not configured.
- Source manifest has not been staged to a Lightning workspace.
- Remote CUDA runtime has not been probed.
- Non-dry-run launch still requires lane claims before provider execution.

## Reactivation criteria

Use the current five archive SHA-256 values above. Reject stale provider plans
that name `b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7`.
After Lightning identity and workspace preflights pass, claim each
`contest_cuda` lane before non-dry-run launch. For actual TT5L side-info effect
claims, pair these CUDA cells with matching `contest_cpu` cells, or fix Modal
billing and use the paired Modal dispatch plan.
