# RECOVERY-1 NVDEC forensic anchor directive 2026-05-14

**Active subagent**: `a155dcdabb5e1595d` (RECOVERY-1 D1 L2 fix + Catalog #220)
**Source**: RECOVERY-2 (`ae42f94dcb44b1d43`) just harvested D1 R4 Modal call_id `fc-01KRKBF28G2M3N73FS7PDCB6AZ` with **critical new finding**: rc=1 at 1022s due to NVDEC `nvml error (999)` infrastructure failure inside DALI pipeline — **NOT method failure**. The D1 substrate L1 SCAFFOLD itself trained + archive built + auth-eval reached; the FAILURE was at NVDEC driver level during eval.

**Operator directive**: continue D1 L2 INTEGRATION + margin-map shrink + Catalog #220 work AS-IS. The NVDEC issue is orthogonal infrastructure noise; the L2 fix + Catalog #220 self-protection still need to land per operator directive *"all of those fixes need to be made and also whatever led to the decision to defer needs to be permanently prevented"*.

**Adjusted operator-routable for D1 dispatch validation post-L2-fix**:

When you re-fire the Modal smoke after L2 INTEGRATION lands, request:
- T4 instance with `cuda_vers>=12.4` (NVDEC stable on driver >=550)
- Add `--probe-nvdec` pre-flight in remote driver (per Catalog #163 sentinel-source pattern) — if NVDEC probe fails, exit early before paying for training-only cost
- Document the NVDEC infra-failure class in your memory file as anchor for future substrate dispatches

**Cross-ref**: RECOVERY-2 landing memo `feedback_recovery_2_c6_finish_and_modal_harvest_landed_20260514.md` documents the rc=1/1022s/NVDEC anchor.

Tagged `research_only=true`. Recovery-1 picks this up on next checkpoint cycle.
