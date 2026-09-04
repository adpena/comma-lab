# ddm_ps1 — PR #140 update packet PREPARED for the fs1 bytes (2026-09-04) — NOTHING PUBLISHED

Tokens: `[no-triality] [p0-ledger-ok]`

## What this arm did and did not do

**Did:** staged, under `/Volumes/APDataStore/pact/ddm_ps1_pr140_update_prep/`, a complete
PR #140 update packet for the fs1 archive — a sixth lossless `compress.py` stage that
**actually rebuilds** the fs1 bytes from afr1's and refuses on anything else, the two
`inflate.py` pin lines, the README/report/PR-body number deltas, a strict compliance dry-run,
and a release plan.

**Did not:** push, create or upload a release, edit the PR body, post a comment, touch
`submissions/semantic_joint_ctxmix/`, or write under `upstream/`. The p0 swap procedure
holds: publication waits on the operator's one-line confirm.

## The prior-law prediction, and where it was FALSE

The charter predicted that fs1's runtime tree differs from the live PR tree **only** in the
two `inflate.py` pin lines. **MEASURED: false as stated.** `diff -rq` over the two trees:

| file | status |
|---|---|
| `inflate.py` | differs — exactly the two pin lines (prediction holds) |
| `README.md` | differs — fs1's is the **g8v1-era** text, predating the fr2c credit and TODO cures |
| `compress.py` | differs — fs1's is the g8v1-era encoder |
| `MANIFEST.sha256` | differs — the three rows above |
| 35 `runtime/` + `cpr1/` files, `inflate.sh` | **byte-identical** |

fs1's tree was built from the g8v1 pointer-reproof tree, not from the posted PR tree. Copying
it into the packet would have **regressed the public README** — dropping the named credits
(Fesal Fayed, Shreyan Mohanty, @JasonMo123) and the operator-approved fr2c TODO wording. So
the staged packet is the **live PR tree** with fs1's two pin lines applied. The staged
`inflate.py` is then byte-identical to fs1's (`53a2da21…`, 2,476 B), which is what the
evaluated row actually ran.

## Deliverable 1 — the sixth stage REBUILDS the bytes (NO-FAKE proof)

`compress.py` grew one stage: `fold_fs1_stage`. It consumes the pinned afr1 archive
(`cbb8d928…`, 180,002 B), re-encodes the frame-0 selector section, and refuses unless the
result is `50fcaf1a…708cf` at 180,022 B. It was **executed**, not asserted:

| run | rebuilt SHA-256 | bytes | elapsed |
|---|---|---|---|
| run 1 | `50fcaf1ac3c8504abdf3e0daff7c5bce32104f19d8de4a7ba207816f32e708cf` | 180,022 | 0.844 s |
| run 2 | `50fcaf1ac3c8504abdf3e0daff7c5bce32104f19d8de4a7ba207816f32e708cf` | 180,022 | 0.831 s |

Run 1 and run 2 are byte-identical, and both are byte-identical to the sealed fs1
`archive.zip`. Both payloads are retained with their SHA-256 recorded
(`evidence/rebuild/run{1,2}_archive.zip`, receipts `STAGE6_run{1,2}.json`).

The stage refuses on four separate grounds, and a negative control fired all four
(`evidence/rebuild/STAGE6_NEGATIVE_CONTROL.json`, `all_refused: true`):

| perturbation | refusal |
|---|---|
| one changed mode | blob `007ad7cc…` ≠ pinned `c6864efa…` |
| one changed pair position | blob `b888b433…` ≠ pinned |
| one dropped pair | blob `174acf6e…` ≠ pinned |
| wrong pointer archive | pointer SHA-256 mismatch vs `cbb8d928…` |

Inside the stage, in order: the pointer pin · the afr1 selector blob pin (`67d43d90…`, 14 B) ·
a **container identity control** that rebuilds afr1 with its unchanged tail and must reproduce
`cbb8d928…` exactly (the no-op detector for brotli parameters, RX1 header packing and ZIP
framing) · the selector blob pin (`c6864efa…`, 34 B) · an in-memory determinism repeat · a
parse-back through the receiver's own `read_residual_archive` proving every non-selector
section byte-identical to afr1's · the final archive pin.

**Rule 118 boundary.** The 24 `(pair, mode)` choices are carried as `FS1_SELECTOR_CHOICES`, a
data table documented in place as video-derived content — the recorded output of the pose
re-solve, and exactly what the 34-byte selector blob inside `archive.zip` stores and pays for.
The code around it is generic: the colex rank and the 3-bit label packer are the inverse of the
receiver's `decode_selector`, and the encoder decodes its own output through the **shipped**
decoder and refuses bytes whose parse-back differs. Nothing video-derived is disguised as an
algorithm, and `compress.py` is not shipped in `archive.zip`.

## Deliverable 2 — runtime and manifest

Staged `inflate.py` SHA-256 `53a2da21d8600804d38a34be4f98064e3e29a9961edd04e7121bbb08e37aa667`,
2,476 B — byte-identical to fs1's. Its diff against the live PR tree is exactly two lines
(`evidence/diff/inflate_py_two_pin_lines.diff`). MANIFEST rehashed: 39 rows, three changed
(`README.md`, `compress.py`, `inflate.py`), 36 untouched.

Tree digests, MEASURED: staged `ec4c9d19…` (with custody files) / `bc663204…` (without);
evaluated `fbf4aaf4…`. They differ because the evaluated tree carries the older README and
compress.py. Every file `inflate.sh` executes is byte-identical between them.

## Deliverable 3 — documents

README: numbers only (`evidence/diff/readme_numbers_only.diff`) — 180,002 → 180,022,
0.14797617125559104 → 0.14786319521362173, archive SHA-256, the T4 timing sentence, and one
new bullet naming the sixth stage. The disclosure paragraph, the credits and the TODO bullets
are **unchanged**. `report.txt` is the harvested evaluator artifact verbatim (`68ae91de…`,
664 B). `PR_BODY_DELTA.md` lists six number/link edits and states what must not change.
**Title stays `semantic_joint_ctxmix (0.148)`** — 0.14786319521362173 rounds to 0.148 at three
decimals, so no title edit is needed.

## Deliverable 4 — compliance dry-run: 78 GREEN / 7 RED of 85

`scripts/pre_submission_compliance_check.py --contest-final --strict` against the staged
packet with the expected SHA-256/size and the harvested auth-eval JSON. Five runs are retained;
r5 is terminal. Against the frozen pq12 adjudication (80 GREEN / 7 RED of 87):

| pq12 red | now |
|---|---|
| `auth_eval_raw_promotion_policy_blockers_absent` | still red — STRUCTURAL-RECORD |
| `contest_cpu_auth_eval_exists` | still red — RECORD-WITH-REASON |
| `hosted_archive_manifest_supplied` | still red — nothing hosted yet |
| `submission_runtime_tree_matches_auth_eval` | still red — **new cause**: the evaluated tree is g8v1-era |
| `public_scan_has_no_private_surface` | **GREEN** |
| `submission_runtime_has_no_network_install_or_local_paths` | **GREEN** |
| `submission_runtime_imports_within_allowlist` | **GREEN** |

**Three NEW red classes, all reported and none waived:**
`dispatch_claim_terminal_archive_sha_bound`, `dispatch_claim_terminal_runtime_tree_sha_bound`,
`dispatch_claim_prior_active_row`. The terminal claim row for lane
`ddm_fs1_t4_frame0_selector_20260904` records the archive SHA by 8-character prefix and no
runtime tree SHA; the checker wants both in full. This is ledger shape, not a defect in the
bytes, and it is the same class pq12 cured by appending a canonical terminal row. Appending to
the shared claims ledger belongs to MAIN, not to this preparation arm.

Two reds the run caught on **my own** staged text, both fixed before r5: the identity report
first carried a raw Modal call id (`public_scan_has_no_private_surface`), and the
evaluator-verbatim `report.txt` backticks its SHA-256 so the checker's binding regex — which
wants `Archive SHA-256: <64 hex>` unquoted — never matched. The cure for the second was the
pq12-shaped packet identity report (`PACKET_REPORT_fs1.txt`), which clears three reds at once.

I also refused to invent a `portable_runtime_content_tree_sha256` for the generated
`archive_manifest.json`: the pq12 stager produced that digest and this arm has no canonical
producer for it, so the key is absent with its reason recorded rather than filled with a
number that would read as authoritative.

## Deliverable 5 — `RELEASE_PLAN.md`, not executed

Six steps: operator gate → branch update → push → **new** fork release tag
`semantic_joint_ctxmix-fs1` (a new tag, not an asset swap, so the afr1 link keeps resolving to
the bytes it names) → fetchback verification that must show `50fcaf1a…`, `180022` and
`HOSTED_BYTES_IDENTICAL` before the body is edited → `gh pr edit 140` with the operator's own
text. Every command is written out; none was run.

## Custody

| artifact | SHA-256 | bytes |
|---|---|---|
| `STAGING_REPORT.json` | `7e9f1c4fc346468f5afece461e59ec373043e68b59897e0b38c9724b656bd571` | — |
| `evidence/rebuild/run1_archive.zip` | `50fcaf1ac3c8504abdf3e0daff7c5bce32104f19d8de4a7ba207816f32e708cf` | 180,022 |
| `evidence/rebuild/run2_archive.zip` | same | 180,022 |
| `report.txt` (evaluator, verbatim) | `68ae91deab96d60c08b411e0360651da6b4fa9ba04ccf494d6489b12eb69c08f` | 664 |
| `evidence/inputs/contest_auth_eval_fs1_t4_20260904.json` | `b93efb62e536a39bf7ee3acc9c303861a4fa2828a6742cb6d3ded697f67a2443` | 51,375 |
| harvest `MODAL_REMOTE_RESULT.json` | `4bcecd01f3481b29c57415a934b5e3fab5f71e823d1a8802637a986093fa54a9` | — |

Consumed, not re-measured: the T4 score row and its component values. Measured here: archive
and packet bytes and hashes, the stage-6 rebuild and its repeat, the negative controls, the
tree diffs, and the compliance result.

## Equations leg (`tac.canonical_equations`)

No new equation and no re-fit. The stage-6 rebuild is a byte identity, not a law: it produces
no ΔS of its own, and the fs1 row's own equation anchor
(`exchange_ratio_noise_floor_v1`, from the pointer memo) is untouched by a packaging arm. If
the operator fires a fresh T4 row on the final staged tree to cure the tree-digest red, that
row is where an equations-leg update would belong.

## Hot-state line proposal

`PR #140 carries the afr1 bytes (180,002 B). The fs1 update packet is PREPARED at
/Volumes/APDataStore/pact/ddm_ps1_pr140_update_prep/ (stage-6 rebuild proved, 78 GREEN / 7 RED,
3 new reds are dispatch-claim ledger shape). BLOCKED-ON-OPERATOR-CONFIRM: nothing pushed,
hosted or edited.`

Own-vehicle frontier: **fs1 — S 0.14786319521362173 @ 180,022 B [contest-CUDA T4 n600]**,
archive sha `50fcaf1a…708cf`. This arm did not move it.
