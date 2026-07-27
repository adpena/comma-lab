# Premise falsification — self-orient is not an established required G111 cure

UTC: 2026-07-27T12:39:12Z  
Role: Codex adversarial review  
Verdict scope: current G111/G105 architecture and existing repository evidence only  
Pointer: dynamic upstream/display frontier `0.172`; unchanged

## Premise tested

The first G114 coupled-feasibility pass proposed that G111 was a predictable
frontier NO-GO primarily because it disables self-orient, citing the historical
approximately `-48% d_seg` directional-basis result and the fact that the
available `v9c2_defensive_bank` checkpoint carries self-orient.

## Falsification

That premise is not admissible as the required correction:

1. `tools/launch_witness_run.py` records the later governed v7.5.2 decision:
   self-orient OFF was operator-GO after realized transfer of the historical
   `-48%` effect measured approximately zero, while removing a 47 GiB memory
   tax.
2. `.omx/research/codex_findings_v9_cgauge_fake_remediation_20260714_codex.md`
   classifies `-48%` as an unreproduced n96 advisory and explicitly fixes the
   fake of presenting self-orient ON or the percentage as live V9 evidence.
3. `tools/build_v752_isolation_arms.py` records that the owed16 n600 warm-start
   OFF arm beat ON and requires fresh, same-floor, through-R training arms
   before any self-orient promotion.
4. `.omx/research/v75_seal_20260708.md` independently leaves the production
   self-orient ON/OFF effect owed rather than settled.

Therefore the historical direct-partition result cannot be transferred to
current scorer-native G111, and the presence of self-orient in one ancestor
does not establish that disabling it caused the current feasibility gap.

## Surviving P0

The quantization-surface mismatch survives this falsification:

- trainer deployment/verdict selection uses the legacy arbitrary-scale int8
  checkpoint realization;
- G105 ships a different power-of-two wire realization: int8 shared weights
  and int16 biases, palette, and Y1 codes;
- post-G105 pose refit can repair conditional `Y0|Y1`, but cannot repair a
  semantic `d_seg(Y1)` regression caused by selecting on the wrong quantized
  Y1.

No 3000-epoch frontier launch is justified until the parsed G105 wire
quantizer is the semantic verdict and checkpoint-selection surface. After that
change, rederive the exact four-way whole-archive rate envelope for every
preserved stage. Self-orient may return only as a separately typed, fresh,
same-floor n600 A/B whose public decoder implements the same deterministic
fixed-point ABI; it is not a launch prerequisite.

## Triality and no-orphan wire-in

- DSL: current G111 remains self-orient OFF; no silent flag reversal.
- DAG: EMA state -> exact G105 quantize/serialize/parse -> public Y1 render ->
  scorer verdict -> stage selection -> G110 conditional refit/archive.
- Equation: selection must minimize the realized public object
  `100*d_seg(Y1_G105) + min_Y0|Y1[sqrt(10*d_pose) + rate_cond] + rate_G105`,
  not a differently quantized proxy.

Integration owner: G111/G105 wire-quantized semantic verdict and selection
gate. Candidate, score, and pointer claims remain false.
