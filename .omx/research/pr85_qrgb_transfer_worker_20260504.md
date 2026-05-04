# PR85 QRGB Transfer Worker

## Contract

- tool: `experiments/plan_pr85_qrgb_transfer_actions.py`
- score_claim: false
- dispatch_performed: false
- remote_jobs_dispatched: false
- gpu_required: false
- dispatch_unlocked: false
- no exact eval dispatch was attempted

## Sources

- PR85 archive: 236328 bytes, sha256 `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- PR85 exact T4 score context: 0.25806611029397786
- PR90 qrepro archive: 218080 bytes, sha256 `608ea0355e60faad97b046c27644205d05120ac85ab3e8a99543a75a4ab2dd2d`
- PR90 QRGB residual: 4106 compressed bytes, 4307 nonzero int8 edits
- pair planning source: `experiments/results/pr85_pair_atom_candidates_20260504_orchestrator/planning.json`

## Design Choice

PR90's raw QRGB stream is not transplanted into PR85. The planner lowers the idea into PR85-native stream/value actions on `bias`, `region`, and selected `randmulti` groups because those are the existing archive-consuming PR85 runtime controls. The best current candidate is still action evidence only, not an archive.

## Ranked Candidates

- rank 1: `pr85_qrgb_f2_randglobal_pair_0192` pair=192 stream=randmulti 0->20 proxy_bytes=7 break_even_margin=509.6922114573065 dispatch_unlocked=false
- rank 2: `pr85_qrgb_f1_bias_pair_0060` pair=60 stream=bias 24->5 proxy_bytes=7 break_even_margin=477.2862579445188 dispatch_unlocked=false
- rank 3: `pr85_qrgb_f1_region_pair_0060` pair=60 stream=region 84->40 proxy_bytes=7 break_even_margin=477.2862579445188 dispatch_unlocked=false
- rank 4: `pr85_qrgb_f2_randglobal_pair_0060` pair=60 stream=randmulti 12->4 proxy_bytes=7 break_even_margin=477.2862579445188 dispatch_unlocked=false
- rank 5: `pr85_qrgb_f1_bias_pair_0164` pair=164 stream=bias 3->21 proxy_bytes=7 break_even_margin=458.22756469211794 dispatch_unlocked=false
- rank 6: `pr85_qrgb_f1_region_pair_0164` pair=164 stream=region 37->45 proxy_bytes=7 break_even_margin=458.22756469211794 dispatch_unlocked=false
- rank 7: `pr85_qrgb_f2_randglobal_pair_0164` pair=164 stream=randmulti 18->16 proxy_bytes=7 break_even_margin=458.22756469211794 dispatch_unlocked=false
- rank 8: `pr85_qrgb_f1_region_pair_0197` pair=197 stream=region 60->45 proxy_bytes=7 break_even_margin=445.8814701277624 dispatch_unlocked=false

## Blockers

- no_archive_changing_path: no byte-closed PR85 archive candidate or non-noop custody exists yet
- no_pr85_component_response_for_direction: PR90 QRGB signs are transfer priors, not measured PR85 scorer response
- direct_pr90_qrgb_stream_transplant is blocked because PR85 has no matching QRGB runtime consumer

## Artifacts

- ranking JSON: `experiments/results/pr85_qrgb_transfer_actions_20260504_worker/planning.json`
- pair-action evidence JSON: `experiments/results/pr85_qrgb_transfer_actions_20260504_worker/pair_action_evidence.json`
- stable plan digest: `05f618d3db808131f83a78b509db0e7b834b3326370e518e129edf96c7563528`

## Exact Next Implementation

Feed pair_action_evidence.json into build_pr85_pair_action_candidates.py, then build a local PR85 candidate archive with build_pr85_pair_atom_candidates.py and require fixed-runtime non-noop custody before any lane claim or exact CUDA eval.
