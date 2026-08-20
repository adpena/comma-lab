# ddm_fo2h GENERATION 2 — RECEIPT

UTC 2026-08-19T12:04:59Z · git `3b5824337b894ecb4c2c96a25eba95b12e5fc276` · host Primary

Axis `[macOS-CPU advisory]` · `score_claim=false` · `promotable=false` · pointer UNMOVED.

**GT lineage is explicit per artifact.** `null_shard*/` and `null_retain12/ETA_GATE_ROWS.jsonl`
are **PyAV**-lineage. `FO2H_POSE_LINEAGE_RESCORE.json` and
`FO2H_BEFORE_SIDE_LINEAGE_FACTOR.json` carry **DALI** re-scores against
`gt_cache_dali.pt`, and each records its own lineage in-band.

Gen-1's 256 files are preserved untouched and receipted in `RECEIPT.md`; gen-1's verdict JSON
was copied to `FO2H_ETA_ADJUDICATION.gen1_n48.json` before any gen-2 rerun could overwrite it.

`null_shardC/`, `null_shardD/` and `null_retain12/` were still filling at close and are
`--resume`-capable; re-running the adjudicator over the four shards picks up every new row.

| artifact | bytes | sha256 |
|---|---:|---|
| `FO2H_SAMPLE_GEN2.json` | 1,145 | `7c749162eff39cf5300383f377c48ec29525c6e6072ce24643391f986e3ff0bd` |
| `FO2H_ETA_ADJUDICATION.json` | 64,833 | `596c21926e1c0f7f97b661faefbbe6eaf99f012d2e55c1492a3e2e4800a40dae` |
| `FO2H_ETA_ADJUDICATION.gen1_n48.json` | 46,891 | `7e154b2c7a4cda3c2d1e472954bfc4ff0597a86f8025585c47bad1be3e2da5cc` |
| `FO2H_BEFORE_SIDE_LINEAGE_FACTOR.json` | 7,729 | `ae893fad0cf0182bfaeffec3a19f2d90454ec3b57b1af4bb5badcf98362c79d0` |
| `gen1_n48_rows.jsonl` | 26,652 | `fb080d80bd1cb28f0273a5f7a4ac9c694a597d1fa3773f62bea1d9355da7531c` |
| `PROGRESS.jsonl` | 117,553 | `51a0a51f206fb497b12b3be3aa7d959ebbde8393380caa6056107d76f88726ab` |
| `null_shardC/.launch_start_gate` | 237 | `08ccd63679e92a30453b31a231b80c7eecfacdea80bf70ab5a708007bcabf398` |
| `null_shardC/.watchers_start_gate` | 246 | `42ec2592d780fa8cbc08faeb0f76749aad26ef89296655f626cd7a3021c9e4b9` |
| `null_shardC/ETA_GATE_ROWS.jsonl` | 6,109 | `11c434593145f922225757b660645dcf5cc202676e03affa4109e94e6b1ba3ff` |
| `null_shardC/launch_manifest.json` | 3,644 | `d91a61086636a9ea3f24b24d6d266b773fff68532951c5dfd39ac6bcafbe667f` |
| `null_shardC/resource_safe_run_child.pid` | 6 | `6bc876d3d9fde8463142c783fca2c79f86f5513003804044a2ef74acf6086d92` |
| `null_shardC/resource_safe_run_status.json` | 1,398 | `4c7c31657cf6e6353bbfdc0e576d880e551ea99f2d9cd24fb2cbad3d1016fce4` |
| `null_shardC/run.log` | 1,272 | `d34de4e4823dbb04145a1c8e26df0826bf9bb3581b622d529f4097195c118388` |
| `null_shardC/run.pid` | 6 | `4f295b6167b60d010ebc8a88da0be1a5069a006ac15c47b018d36186a41279a4` |
| `null_shardD/.launch_start_gate` | 237 | `69ef8e1f976126c7a8edaac311d2bf0f7076bb54ad57be816c57121e8ec93ef1` |
| `null_shardD/.watchers_start_gate` | 246 | `8b1833ecdba0691a05b82d1bb38cc5f736ffffb686d4fb3f72dcb353ccdfe806` |
| `null_shardD/ETA_GATE_ROWS.jsonl` | 6,114 | `3b566eae03a0ac7ea3a6757c795e2b22a1c98d43dac868b8f95365ceb7c8e3fd` |
| `null_shardD/launch_manifest.json` | 3,648 | `1e2c73c694041f8df5825c054eb3cb16abdca6052893b32faef9bcd7bc7d58b3` |
| `null_shardD/resource_safe_run_child.pid` | 6 | `b8cd4522adc0d1d56f05273cdf081bf516c5f5d8e51b5b42eb89d1d97a154c65` |
| `null_shardD/resource_safe_run_status.json` | 1,401 | `a6897ab45003ace245c871f28d7cd9666dc754e07c633bca72e4f013f36fb57c` |
| `null_shardD/run.log` | 1,274 | `a3cbb19939c733178c3ce807c7907239d90df82ef340f09678fbc0d0232d15ac` |
| `null_shardD/run.pid` | 6 | `b513d152b4aaf16a3b98fe8e6229c6736e6807162a249cbc6396f1d2da8b52d4` |
| `null_retain12/.launch_start_gate` | 239 | `7aaa5bd8c84b5bc109697459eb7f7676e987570bb922031e584f100353b6b0f1` |
| `null_retain12/.watchers_start_gate` | 248 | `1b9c806e912bfa9357e8e252eec203d051c259df61ab67d1eace437e83babe5c` |
| `null_retain12/ETA_GATE_ROWS.jsonl` | 2,762 | `0f781d32a4a3b561595f022d13e3990bea74f22239737fb88bd13c3612ef872d` |
| `null_retain12/FO2H_POSE_LINEAGE_RESCORE.json` | 3,070 | `71148c79526b23f933fd6f7f1ef1383d8da32f5ac402617e621a2f34e666dc74` |
| `null_retain12/PROGRESS.jsonl` | 1,243 | `72f6428f51b88d57331571aed655a6dc3b73939f5c09d50172725c8d4a9bd205` |
| `null_retain12/cam_edit_pair0015.npy` | 3,052,136 | `308d99f970c7993b3ca96c5d056e93429fc6db23610ef2f5cbb150f2bba59005` |
| `null_retain12/cam_edit_pair0022.npy` | 3,052,136 | `4ddfef0a82f4a0cdee31c482a4d1ab04c44d412353e8d9b8ba4495dd4b2618bd` |
| `null_retain12/cam_edit_pair0027.npy` | 3,052,136 | `1d3879b6aebbbbf6955735f5a1a97790a35413e3490f9b59634a20e0efcf3ad8` |
| `null_retain12/cam_edit_pair0035.npy` | 3,052,136 | `f280108b8b1e5713abb78ea6d13c2999a9905fb992d244abac50fab61bb41d88` |
| `null_retain12/cam_edit_pair0052.npy` | 3,052,136 | `98f5a562079c46400c3b2862821d318c61c0e054b20d7201d0fa54beeffe97fb` |
| `null_retain12/launch_manifest.json` | 3,690 | `8d3d7ba8dfa42621a5a00f116a734c8e92977ce67f7233e47da2acec24d488c0` |
| `null_retain12/resource_safe_run_child.pid` | 6 | `74a48c9d27fbb1e33c2de01b9edb3b0fe4235d9b5224846a2bfbee70055d2614` |
| `null_retain12/resource_safe_run_status.json` | 1,380 | `890020c2ac05712d80ef06f5f44642fc7478fe6d488f5e1b76c9c1beeea18243` |
| `null_retain12/run.log` | 916 | `f5b27f8cc1f7810d65c22be3cf5433611ac3e59dfb1dafa0b86db9db2b500002` |
| `null_retain12/run.pid` | 6 | `9f04e2425000d0414db0d4d2b063f9bbd434121bb8400610d888dc45c18a8433` |
| `identity_control/FO2H_POSE_LINEAGE_RESCORE.json` | 1,590 | `31ca1085d360856efe1fcaed736aaa449391d457878a3bf6d638c294d3b4e2a8` |
| `identity_control/PROGRESS.jsonl` | 723 | `f86f9d943cc1a8b8ea0c7c80c93e29f776a47d903353f490e8891929d34020be` |
| `identity_control/cam_edit_pair0015.npy` | 3,052,136 | `d6a8bf39fbf15048209a5541b4a1f8d0314f1bba9d22aa18c2a6118d7f198c4d` |
| `identity_control/cam_edit_pair0022.npy` | 3,052,136 | `e45c6de0abd7fbf9bb0d1b27700756b3304c637b318a346094d207aa79ae86cc` |

**41 gen-2 files, 21,671,478 bytes.**
