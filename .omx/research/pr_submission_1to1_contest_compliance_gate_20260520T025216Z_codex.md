# PR Submission 1:1 Contest Compliance Gate - Codex Round

UTC: 2026-05-20T02:52:16Z
Lane: `lane_pr_submission_public_surface_convergence_20260519`
Verdict: `NOT_SAFE_TO_PR`

## Gate Change

`scripts/pre_submission_compliance_check.py --contest-final` now verifies two
additional 1:1 runtime-conformance properties:

- `inflate_sh_uses_canonical_three_arg_contract`: `inflate.sh` must consume
  the upstream-style `archive_dir=$1`, `output_dir=$2`, `file_list=$3` contract
  and read the file list.
- `submission_runtime_loads_no_scorers_or_eval`: runtime `.py` / `.sh` files
  must not import or call the contest evaluator/scorers (`evaluate.py`,
  `contest_auth_eval`, `DistortionNet`, `segnet_sd_path`, `posenet_sd_path`,
  `compute_distortion`, or `modules` scorer imports).

This preserves the contest rule that inflate is a deterministic decompressor,
not a scorer/evaluator path.

## Regression Coverage

```text
test_pre_submission_check_requires_canonical_inflate_signature
test_pre_submission_check_rejects_runtime_scorer_imports
```

Focused checks:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "inflate_signature or scorer_imports or non_executable_inflate"
4 passed in 1.12s

.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "template_placeholder or public_source or source_pin or hosted or axis_labels"
11 passed in 1.61s
```

Full checker file:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
55 passed in 4.37s
```

Hygiene:

```text
git diff --check
clean

.venv/bin/python tools/lane_maturity.py validate
OK - 1035 lane(s) validated cleanly.
```

## Current Strict Gate Result

Command wrote:

```text
/tmp/pr101_fec6_gate_1to1_compliance.json
```

Result: exit code `1`, `passed=false`.

Currently green local 1:1 conformance properties:

```text
required_file_present:archive.zip
required_file_present:inflate.sh
required_file_present:report.txt
inflate_sh_executable
inflate_sh_uses_canonical_three_arg_contract
inflate_sh_loads_no_scorers_or_eval
submission_runtime_loads_no_scorers_or_eval
zip_expected_single_member
```

Current runtime identifiers:

```text
submission_runtime_tree_sha256=cd76c8ac46b04e03ba2a1fab14f4f93e4ed02efe7c56a983a27cd13331167c57
portable_runtime_tree_sha256_without_custody_files=0c70199106957e6140520e879c86fa13e23a5915abf6b15acedccafcb9635de1
scorer_import_hits=[]
```

## Remaining 1:1 Compliance Blockers

The packet is not 1:1 contest-conformant yet. The strict gate still blocks on:

```text
runtime_equivalence_proof_submission_runtime_matches
submission_runtime_tree_matches_auth_eval
hosted_archive_manifest_supplied
hosted_archive_public_text_has_no_placeholder
public_source_pinned_revision_present
public_source_pin_text_has_no_placeholder
public_source_pinned_revision_publicly_visible
public_text_has_no_unresolved_template_placeholders
```

Runtime custody is the critical non-publication blocker: the existing
`pre_submission_runtime_equivalence_proof_v1` proves the earlier submission
runtime hashes:

```text
fd4b36b0114789ffd25c6169f529bca70b20da8f70e4ee1336dad9fd64971a09
ee53871c8766718b9ab289c4a12a59501795ce7fbf71bfbb42728b5669884e8c
```

The current dirty worktree runtime hashes are:

```text
cd76c8ac46b04e03ba2a1fab14f4f93e4ed02efe7c56a983a27cd13331167c57
0c70199106957e6140520e879c86fa13e23a5915abf6b15acedccafcb9635de1
```

Do not treat the old proof as valid for the current runtime. The next compliant
path is one of:

1. Rebuild a full-output byte-identity proof for the current runtime.
2. Restore the exact runtime that the existing proof covered.
3. Rerun paired exact auth eval on the current runtime and refresh custody.

Public release blockers remain operator-gated: replace `<HOSTED_URL_PLACEHOLDER>`
with a real hosted archive URL bound by `hosted_archive_manifest_v1`, replace
`<PINNED_COMMIT>` only after an operator-authorized public push/tag, and refresh
`public_source_ref_manifest_v1` so the pin is publicly visible.

No push, PR, release, hosted archive publication, GPU spend, or dispatch was
performed.

## Round Extension - Runtime Equivalence Proof Shape Drift

UTC: 2026-05-20T03:06:14Z

`scripts/pre_submission_compliance_check.py --contest-final` now also compares
the optional proof-recorded `submission_runtime_shape` file hashes against the
current submission runtime manifest:

```text
runtime_equivalence_proof_submission_runtime_shape_matches
```

This makes stale proof drift file-addressable instead of only tree-hash
addressable. A proof can no longer appear opaque when the current PR runtime has
changed after the proof was generated.

Regression added:

```text
test_pre_submission_check_rejects_stale_runtime_equivalence_shape
```

Focused checks:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "runtime_equivalence"
3 passed, 56 deselected in 0.79s
```

Full checker file:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
59 passed in 4.28s
```

Strict PR gate rerun wrote:

```text
/tmp/pr101_fec6_gate_runtime_shape_drift.json
```

The stale proof now reports the exact runtime-file drift:

```text
changed:
- inflate.py
  proof_sha256=6758c3b518b4a94c7a531f66b542128d47ef8183adf68314b3191aa89c2ca06f
  current_sha256=45722504b03c1a08bfb28d223c6b2f5a73123b6b42b8c878c56940ace378230a
- src/codec.py
  proof_sha256=8a9bc8b23d2eb30815e955deeeea989632ffcb57630f86fc5878f30d31ab3fd9
  current_sha256=79bad598244d2d5afb7b7b3f258a88921b6dffc45a7071a496245989e24f6685
```

The strict PR gate remains `passed=false`. Publication is still blocked by:

```text
runtime_equivalence_proof_submission_runtime_matches
runtime_equivalence_proof_submission_runtime_shape_matches
submission_runtime_tree_matches_auth_eval
hosted_archive_manifest_supplied
hosted_archive_public_text_has_no_placeholder
public_source_pinned_revision_present
public_source_pin_text_has_no_placeholder
public_source_pinned_revision_publicly_visible
public_text_has_no_unresolved_template_placeholders
```

No push, PR, release, hosted archive publication, GPU spend, or dispatch was
performed.

## Round Extension - Runtime Side-Effect Hygiene

UTC: 2026-05-20T02:55:53Z

`scripts/pre_submission_compliance_check.py --contest-final` now also verifies:

```text
submission_runtime_has_no_network_install_or_local_paths
```

This scan covers runtime `.py` and `.sh` files and fails on network fetches,
package-manager installs, shell/network copy tools, local `/Users/...` paths,
or Python network/process launch surfaces such as `urllib.request`,
`requests.get`, `socket.socket`, `subprocess.Popen`, and `os.system`.

Regression added:

```text
test_pre_submission_check_rejects_runtime_network_install_or_local_paths
```

Focused checks:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "network_install or scorer_imports or inflate_signature or non_executable_inflate"
5 passed in 1.03s

.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "template_placeholder or public_source or source_pin or hosted or axis_labels"
11 passed in 1.44s
```

Full checker file:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
56 passed in 3.98s
```

Strict gate rerun wrote:

```text
/tmp/pr101_fec6_gate_runtime_side_effects.json
```

The current runtime passes the local side-effect scan:

```text
scorer_import_hits=[]
forbidden_side_effect_hits=[]
```

The remaining eight blockers are unchanged from the prior compliance pass:

```text
runtime_equivalence_proof_submission_runtime_matches
submission_runtime_tree_matches_auth_eval
hosted_archive_manifest_supplied
hosted_archive_public_text_has_no_placeholder
public_source_pinned_revision_present
public_source_pin_text_has_no_placeholder
public_source_pinned_revision_publicly_visible
public_text_has_no_unresolved_template_placeholders
```

## Round Extension - Runtime Dependency Closure Allowlist

UTC: 2026-05-20T03:02:45Z

`scripts/pre_submission_compliance_check.py --contest-final` now also parses
runtime Python imports with `ast` and verifies:

```text
submission_runtime_import_allowlist_parseable
submission_runtime_imports_within_allowlist
```

The allowlist permits standard-library imports, local submission-runtime modules,
and the explicit contest runtime dependency roots:

```text
brotli
numpy
torch
```

This makes dependency drift fail closed: a new unreviewed runtime dependency
such as `pandas` cannot silently enter the PR packet, and syntax errors in
runtime Python prevent the allowlist scan from passing.

Regression added:

```text
test_pre_submission_check_rejects_runtime_dependency_outside_allowlist
test_pre_submission_check_rejects_unparseable_runtime_imports
```

Focused checks:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "runtime_dependency or unparseable_runtime_imports or network_install or scorer_imports or inflate_signature"
6 passed, 52 deselected in 0.99s
```

Full checker file:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
58 passed in 4.13s
```

Strict PR gate rerun wrote:

```text
/tmp/pr101_fec6_gate_runtime_import_allowlist.json
```

The current PR runtime passes dependency-closure inspection:

```text
disallowed_runtime_imports=[]
runtime_import_parse_errors=[]
scorer_import_hits=[]
forbidden_side_effect_hits=[]
```

Observed runtime import roots are limited to stdlib/local modules plus `brotli`,
`numpy`, and `torch`:

```text
__future__
brotli
codec
codec_sidecar
frame_selector
functools
io
lzma
math
model
numpy
pathlib
struct
sys
torch
```

The strict PR gate remains `passed=false`; the remaining eight blockers are
unchanged:

```text
runtime_equivalence_proof_submission_runtime_matches
submission_runtime_tree_matches_auth_eval
hosted_archive_manifest_supplied
hosted_archive_public_text_has_no_placeholder
public_source_pinned_revision_present
public_source_pin_text_has_no_placeholder
public_source_pinned_revision_publicly_visible
public_text_has_no_unresolved_template_placeholders
```

Current PR runtime identifiers:

```text
submission_runtime_tree_sha256=cd76c8ac46b04e03ba2a1fab14f4f93e4ed02efe7c56a983a27cd13331167c57
portable_runtime_tree_sha256_without_custody_files=0c70199106957e6140520e879c86fa13e23a5915abf6b15acedccafcb9635de1
```

No push, PR, release, hosted archive publication, GPU spend, or dispatch was
performed.
