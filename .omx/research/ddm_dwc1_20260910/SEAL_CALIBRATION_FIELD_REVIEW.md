# Calibration source-field binding correction

The previous validator authenticated the T4 source bytes but accepted an arbitrary caller-selected numeric path. Selecting `evaluate_elapsed_seconds` could relabel 45 seconds of scoring as decode time despite a 1336.668977565-second inflate in the same retained source. This is a FORMULATION-scoped apparatus defect; source hashing alone cannot prove field semantics.

The validator now accepts only completed decode calibration, with the canonical Modal source paths for inflate seconds, archive identity, runtime identity, and hardware identity. `passed` must be the boolean true and `returncode` must be integer zero. `full_eval_upper_bound` is refused because no distinct trustworthy source contract was implemented. The real tc3 calibration already uses the accepted fields.

## Review pass 1

Traced the new checks before numeric extraction and ratio arithmetic. The missing/wrong source path cannot reach either the candidate's direct measured-overrun comparison or projected budget acceptance. Verified exact-type handling excludes numeric true/false substitutions in source status. Read the adapted shared fixture and all tests that mutate its nested source JSON. No further finding.

## Review pass 2

Attacked the correction with an explicit incident-shaped test: actual inflate is 1336.668977565 seconds, evaluation is 45 seconds, every edited source/calibration reference is rehashed, and both ratio and projected time are recomputed from the substituted evaluation field. The test asserts semantic-source refusal rather than stale hash or arithmetic refusal. Separate tests cover relabeling as whole-eval scope, source failure, malformed status, and redirected identity fields. No further finding on this bounded surface.

Assumption challenged: a named receipt field is sufficient provenance. It is insufficient when the name is caller-selected; the consumer must bind the actual worker schema. The correction encodes this binding rather than trusting another descriptive field.

Ruff passed on both edited Python files. Test execution is intentionally deferred until root releases the active tc4 timing; these new controls are written and reviewed, not yet claimed executed. Root owns that test fire order and the final fire-tool strict flip. Existing historical test receipts remain historical.
