# PR103 -16B strict compliance closure (2026-05-11)

## Summary

The PR103 clean-runtime mid32+latent-hi `-16B` packet now passes the strict
pre-submission compliance gate. This closes the earlier blocker as a checker
comparison bug, not a packet/runtime regression.

Strict report:

`.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_candidate/pre_submission_compliance.strict.json`

Result:

- `passed=true`
- `checks=53`
- archive SHA-256:
  `8460014d70855ce9226285f80513d6d743ed23723870a6a38b009cfca40f423e`
- archive bytes: `178207`
- strict formula score: `0.22776740354111893`
- auth-eval raw runtime tree SHA-256:
  `59c6a80f62b6bd8d7fab1b7252898b4dc19fa8736a91e2b7ecac6f8bb2e23ee2`
- local submission raw runtime tree SHA-256 after custody pruning:
  `6eccdffce3db87e77364ae0abd93ae1b8774b708bf0df291c354f19307ea39bc`
- portable executable-runtime SHA-256 after custody pruning:
  `89cd87b144b4441420931947aa1b0e2661d8231ce8b84a534afd87d2c163e9e2`

## Bug fixed

`scripts/pre_submission_compliance_check.py` previously compared the local
submission runtime tree after excluding submission-custody files against the
raw auth-eval runtime tree. That was not apples-to-apples when auth eval
recorded custody files such as `archive_manifest.json` and `report.txt`.

The first fix added auth-side pruned runtime candidates. The live PR103 packet
then exposed a second legitimate mismatch: Modal records the evaluated runtime
under `/tmp/modal_auth_eval/submission_dir`, while the local packet is under
the repo path. The checker now records both:

- raw runtime tree SHA values for exact auth-eval custody; and
- a portable executable-runtime SHA over relative paths, byte counts, file
  hashes, dependency roots, repo-local `tac` import closure, and
  `upstream/evaluate.py`, after excluding submission-custody files.

The strict match is therefore strong enough to reject a real `inflate.sh`
mismatch while not depending on provider-local absolute paths.

## Regression protection

`src/tac/tests/test_pre_submission_compliance_check.py` now covers:

- auth-eval manifests that include custody files;
- remote auth-eval runtime roots that differ from the local packet root; and
- rejection when a non-custody executable runtime file hash differs.

Focused verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
# 25 passed
```

## Score-lowering implication

This does not create a new score result. It upgrades the existing PR103 `-16B`
rate-only `[contest-CUDA]` observation from compliance-blocked to
strict-compliance-closed. Remaining PR103 promotion/adjudication questions are
policy and parity-scope questions, not a strict-compliance failure.
