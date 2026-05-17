# L5 v2 TT5L dispatch custody materialization

Date: 2026-05-17

## Verdict

The TT5L side-info effect-curve custody blocker is resolved at the planning and
operator-review layer. This is not a provider dispatch, score claim, promotion
claim, or architecture-lock clearance.

## Materialized local custody

- Runtime report:
  `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir/report.txt`
  - SHA-256:
    `a39bafeb144166a72d026e44e1ac1ec1e2b1254ccbc4e1304f2a2277fa790724`
- Variant archive manifests:
  - `zero`:
    `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/zero/archive_manifest.json`
    - SHA-256:
      `ae733a159d4270836d1519aa12c5e3425679f9cdbc440949491539835a84ec9e`
  - `random_lsb`:
    `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/random_lsb/archive_manifest.json`
    - SHA-256:
      `0cf581d53bc4310a6dde57c4c1fba7d14c43e6f23b89b4be8f4dc676dd88ba07`
  - `shuffled`:
    `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/shuffled/archive_manifest.json`
    - SHA-256:
      `8d375acde35fea4db1c7e32b22871f056992e4b9983917b8b325b59b9d93a796`
  - `trained`:
    `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/trained/archive_manifest.json`
    - SHA-256:
      `45764dc0ffc16c5236abf107a4f7da4b3ce534e872202f99b25fad14a70633e4`
  - `ablated`:
    `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/ablated/archive_manifest.json`
    - SHA-256:
      `dffd322cebc3eb77170dbf289077fa36951798315f71ca5093ed06a38620a918`

These files live under ignored `experiments/results/` custody roots. The
reusable materialization path is committed in
`src/tac/optimization/tt5l_sideinfo_variant_packets.py`; rerun
`tools/build_tt5l_sideinfo_variant_packets.py` to regenerate them.

## Refreshed authority artifacts

- Variant manifest:
  `.omx/research/l5_v2_tt5l_current_code_fullshape_sideinfo_variant_packets_20260517_codex.json`
  - SHA-256:
    `53520df3292bcf9a1f4dce23f4da0ea65f7e54de5d07b42ee80aee4b2a9966ec`
  - runtime file count: `11`
  - runtime tree SHA-256:
    `4d7d6aeb733e3cb08307a2a9abe9fe8b4d164d33218f3967108324f9f29b0cf7`
- Modal paired dispatch plan:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json`
  - SHA-256:
    `89f589fb0ade79f92184275eb2cf34417ae0fe2e4637847c22f69e22dbba941f`
  - `work_unit_count=5`
  - `ready_work_unit_count=5`
  - per-variant `dispatch_blockers=[]`
- Lightning paired-axis plan:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
  - SHA-256:
    `935c465917a3791d15140c591e7a82814f83c2b6fbb773d67ee5fc3e0ce7d616`
  - `source_commit=730b52ee3118b3ec6331a600d760f4fe44897d4e`
  - `cell_count=10`
  - `all_cells_dry_run_ready=true`
- Lightning execution preflight:
  `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_preflight_20260517_codex.json`
  - SHA-256:
    `d99e2c57d14c98946206ed55f36d7f6f8afcff85d7a6a29a13a95e2244462153`
  - `ready_cell_count=10`
  - `ready_for_operator_claiming=true`
- Lightning execution bundle:
  `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json`
  - SHA-256:
    `3d5a493a59b5707b803e98fe5dcfa0cf67fc3a15e531533a7c495662e24fcde8`
  - `ready_dry_run_cell_count=10`
  - `ready_for_dry_run_submit=true`
- Lightning dry-run verification:
  `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification_20260517_codex.json`
  - SHA-256:
    `91c2d3000c67799a564bfb63ea71db0a0fe3eab95b8d76746cfe70cda7ce2ed4`
  - `passed_cell_count=10`
  - `all_dry_runs_passed=true`
- Lightning route-unblock packet:
  `.omx/research/l5_v2_tt5l_lightning_route_unblock_packet_20260517_codex.json`
  - SHA-256:
    `1adbfcc02028e214d58afad21dfeecd83898ca0d592b58797e64f93d12fdbd44`
  - `artifact_blocker_count=0`
  - `source_relevant_diff_paths=[]`
- Lightning required-doctor plan:
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.json`
  - SHA-256:
    `64daa9a46787ab78042055b70797dd9cc9adc6d4f06cf72e7f584fcedca3b19f`
  - `ready_for_operator_doctor=true`
- Architecture-lock packet:
  `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.json`
  - SHA-256:
    `536b457dc00b0ba0adc425c885a8f0a75f542d9317d039cc9b5fa762d9ac8aa3`
  - `architecture_lock_allowed=false`
  - remaining blockers:
    `requires_all_l5_v2_gate_evidence_valid`,
    `requires_c1_z5_tt5l_probe_gate_evidence`,
    `requires_paired_cpu_cuda_sideinfo_effect_curve`

## Hardening landed

- `build_tt5l_sideinfo_variant_packets` now emits per-variant archive manifests
  and a runtime `report.txt` before runtime custody is computed.
- The TT5L route-unblock packet now fails closed when source-relevant dirty
  files exist, even if `source_commit == current_head_commit`.
- The TT5L route-unblock packet no longer hashes the architecture-lock packet
  as an upstream source artifact. Architecture lock is downstream of route and
  doctor status; hashing it inside the route packet creates circular custody
  drift. The regenerated architecture-lock packet records
  `source_route_packet_sha256_matches=true`.
- Focused tests cover archive-manifest emission, runtime-report emission, and
  dirty source-relevant route blocking.

## Next action

Run the Lightning doctor, stage per-cell source manifests, claim each per-axis
lane, and execute the 10 paired CPU/CUDA cells only after the doctor result and
source-manifest custody are green. No score movement, method status change, or
architecture lock is allowed until those harvested exact-eval cells exist.
