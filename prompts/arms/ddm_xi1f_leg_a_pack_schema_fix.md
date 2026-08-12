You are ddm_xi1f, a codex arm on the pact repo (cwd: repo root).

READ FIRST (binding): CLAUDE.md + AGENTS.md; the xi1 receipt
.omx/research/ddm_xi1_screw_conditioned_learned_prior_20260812.md; the crash
log /Volumes/APDataStore/pact/ddm_xi1_20260812/run/leg_a_main_fire.log.

THE BUG (MAIN-diagnosed, fix it properly): xi1 Leg A TRAINED clean on MPS
(20 epochs, spatial cell bpp 0.1201) but crashed at pack time —
tools/run_ddm_xi1_screw_conditioned_learned_prior.py:992
`source.load_state_dict(terminal["ema_shadow"])` raises Missing key(s)
"*.bit_depth" for IntegerHPAC. Telemetry shows bit_depth_histogram {} all
epochs — the Leg-A training path never registered/saved the QAT bit-depth
buffers that the pack-side IntegerHPAC expects. Compare against the PROVEN
cl1 chain (tools/fit_ddm_cl1_hpac_capacity.py — its train→pack round-trip
passed byte-equal today): find where cl1 registers/saves bit_depth and where
xi1's adapted copy dropped it. Fix at the ROOT (trainer-side registration so
the shadow carries real bit depths), NOT strict=False suppression (NO-FAKE:
bit_depth drives pack quantization; silently defaulting it is a fake pack).

DELIVER: (1) the root-cause fix, 2 review passes; (2) a CPU unit test that
round-trips a synthetic checkpoint through pack_and_encode_cell's load path
(the bug reproduces device-independently); (3) verify resume compatibility —
the fixed trainer must RESUME from the existing Leg-A checkpoint if its
buffers can be reconstructed, else document that a fresh 20-epoch rerun is
required (~4 min/cell measured); (4) the exact re-fire command for MAIN
(arms cannot reach Metal). DO NOT run MPS.

OPERATIONAL: serializer --no-co-author, post-edit --expected-content-sha256,
tags [no-triality] [p0-ledger-ok]; 2 review passes per .py; payload law P0
(state under /Volumes/APDataStore/pact/ddm_xi1_20260812/fix/); blocked-git →
commit_intent. Final message: root cause + fix commit + test result +
resume-vs-rerun verdict + the pinned MAIN re-fire command.
