# NEXT_IF_RESUMED

BL1 code, tests, review marks, and receipt were completed. If resumed:

1. Verify serializer result first:
   - `git log --oneline -3`
   - `git status --porcelain=v1 -- tools/launch_detached_process.py tools/tests/test_sigurg_kill_class_guard.py src/tac/pr103_lc_ac_runtime_adapter.py src/tac/tests/test_pr103_lc_ac_runtime_adapter.py src/tac/preflight.py src/tac/tests/test_check_bl1_background_launcher_rc.py .omx/research/ddm_bl1_20260805`

2. If BL1 was not committed, run the serializer with the post-edit sha256 for every file listed in the receipt/manifest. Do not hand-roll `git commit`.

3. Follow-on fire order:
   - `BL1-FOLLOW-BPY`: patch the three remaining si1 bare-python emitters (`tools/witness_byte_close_and_eval.py`, `src/tac/v2_compose/archive_grammar.py`, `src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py`) to the same fail-closed shell block and test them.
   - `BL1-FOLLOW-HINERV-RC`: patch the HiNeRV backend stale-score-on-nonzero-rc path before any HiNeRV backend row drives a decision.

No scorer or paid work is owed by BL1.
