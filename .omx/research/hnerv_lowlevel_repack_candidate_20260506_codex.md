# HNeRV Low-Level Repack Candidate - 2026-05-06

Evidence grade: `byte_profile_only`

Score claim: `false`

This tranche turned the HNeRV payload scorecard follow-up into a concrete
archive candidate against the current PR106x exact frontier. The candidate only
recodes the packed HNeRV decoder brotli section and preserves brotli raw
equivalence for the decoded section payload.

## Result

- Source archive: `experiments/results/public_pr106_belt_and_suspenders_xrepack_20260504_codex/archive.zip`
- Source label: `PR106x`
- Source archive bytes: `186231`
- Candidate archive: `experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip`
- Candidate archive bytes: `186080`
- Archive byte delta: `-151`
- Repacked section: `decoder_packed_brotli`
- Source section bytes: `170278`
- Candidate section bytes: `170127`
- Candidate diff audit blockers: none
- Ready for archive preflight: `true`
- Ready for exact eval dispatch: `false`

The expected rate-only change if SegNet/PoseNet components are bit-identical is
approximately `-0.000100546893` score. This is not a promoted score delta until
the candidate archive passes exact CUDA auth eval through the canonical path.

## Follow-Up

1. Run strict archive manifest/preflight on the candidate archive.
2. Claim the lane before any remote eval dispatch.
3. Run exact CUDA auth eval on T4/equivalent hardware.
4. If exact components are unchanged, promote as a byte-only HNeRV archive
   improvement; if components drift, preserve as a negative payload-custody
   result and investigate runtime/member-name effects.

