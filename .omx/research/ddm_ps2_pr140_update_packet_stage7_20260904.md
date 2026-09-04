# ddm_ps2 — PR #140 update packet re-pointed at the fs2 bytes, with a SEVENTH stage that rebuilds them (2026-09-04) — NOTHING PUBLISHED

Tokens: `[no-triality] [p0-ledger-ok]`

## What this arm did and did not do

**Did:** extended `ddm_ps1`'s packet into
`/Volumes/APDataStore/pact/ddm_ps2_pr140_update_prep/` — a seventh lossless `compress.py`
stage that **actually rebuilds** the fs2 bytes from the fs1 bytes and refuses on anything
else, the two `inflate.py` pin lines re-pinned to fs2, the README/report/PR-body number
deltas, a strict compliance dry-run, a `SEAL_VALID` custody seal, and a release plan.

**Did not:** push, create or upload a release, edit the PR body, post a comment, touch
`submissions/semantic_joint_ctxmix/`, write under `upstream/`, or fire Modal. The p0 swap
procedure holds: publication waits on the operator's one-line confirm, and MAIN fires the seal.

## The prior-law prediction, and it HELD

The charter predicted a seventh stage consuming the fs1 archive and the retained fs2
coordinate table would rebuild
`a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6` (180,023 B) bit-exactly,
with the identity control reproducing `50fcaf1a…708cf`. **MEASURED: both held, three times.**

| run | rebuilt SHA-256 | bytes | elapsed |
|---|---|---|---|
| run 1 | `a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6` | 180,023 | 2.18 s |
| run 2 | same | 180,023 | 2.18 s |
| run 3, clean state | same | 180,023 | 2.19 s |

The local work store was deleted before run 3, so the stage depends on no surviving scratch.
All three payloads are byte-identical to each other and to the sealed fs2 `archive.zip` —
`cmp` against `retained/candidate_D_alternation/archive.zip` returns clean. Every payload is
retained with its SHA-256 (`evidence/rebuild/run{1,2,3_cleanstate}_archive.zip`, receipts
`STAGE7_run*.json`).

## Deliverable 1 — the seventh stage REBUILDS the bytes (NO-FAKE proof)

`fold_fs2_stage` consumes the pinned fs1 archive, overlays fifteen pairs' twelve-coefficient
rows onto the body's own decoded code table, moves one frame-0 selector label, and re-encodes
the CAP1 carrier through the shipped receiver's decode chain run backwards. Six refusal
grounds fire in order:

1. the pointer pin (`50fcaf1a…`, 180,022 B);
2. the fs1 code-table pin — the body must decode to `1a5b7a46…`, the table the re-solve was
   defined against;
3. a **container identity control** that re-encodes the fs1 body's OWN codes and must reproduce
   `50fcaf1a…` exactly, with `packed_metadata_identical` and `rice_payload_identical`. This is
   the no-op detector for the whole encode path — Rice coding, packed CAP1 metadata, the
   RR5/DX2 riders, brotli, the RX1 header, ZIP framing. Without it no byte delta would be
   attributable to the carrier;
4. the fs2 code-table pin (`5f7ca86e…`) plus the scale facts: 15 pairs, 67 coordinates;
5. the selector blob pin (`15c256dc…`, 34 B) **and a measured length check** — the label move
   must cost zero bytes, not be assumed to;
6. an in-memory determinism repeat, a parse-back through the receiver's own
   `read_residual_archive` proving eleven other sections byte-identical, and the final archive
   pin.

Four negative controls fired (`STAGE7_NEGATIVE_CONTROL.json`, `all_refused: true`):

| perturbation | refusal |
|---|---|
| one changed int12 coordinate | code table `28f7aa4c…` ≠ pinned `5f7ca86e…` |
| one dropped pair | code table `c91d61d4…` ≠ pinned |
| selector label put back at stage 6's mode | blob `c6864efa…` ≠ pinned `15c256dc…` |
| wrong pointer archive | pointer SHA-256 mismatch vs `50fcaf1a…` |

Measured by the stage itself, independently of the fs2 arm's own build receipt and agreeing
with it row for row: Rice bits 78,628 → 78,634 (+6 bits = the whole +1 byte), selector blob
34 B → 34 B, changed pairs `[5, 71, 77, 95, 161, 173, 221, 259, 436, 488, 504, 518, 547, 555,
586]`, changed selector pair `[259]`, container `(ck2=false, q=9, lgwin=16)`.

**Rule 118 boundary.** Two tables are carried as data: `FS2_CARRIER_ROWS` (15 pairs × 12
signed-int12 coordinates) and `FS2_SELECTOR_CHOICES` (24 `(pair, mode)` tuples). Both are
documented in place as video-derived content — the recorded output of the damped Gauss–Newton
re-solve, read out of `codes_fs2_D_alt.npy` (`7861538e…`) and `choices_D_alternation.npy`
(`0f701d15…`) in the fs2 arm's retention manifest, never re-derived. They are exactly what the
CAP1 Rice payload and the 34-byte selector blob inside `archive.zip` store and pay for. The
code around them is generic: the vendored carrier-splice encoder runs the shipped receiver's
own modules backwards and refuses bytes whose parse-back differs. `compress.py` is not shipped
in `archive.zip`.

**The vendored encoder.** `ddm_up3_carrier_splice.py` (39,701 B, `b7140cf9…`) joins the
embedded-source set rather than being re-implemented. It is adapted at materialization time
exactly the way JG2 already was: its two lab-absolute `Path` defaults become `None`
(`53a599e5…`), which removes the private paths from the public tree without changing the
executed mechanism, because every stage-7 call site passes `runtime_dir` explicitly. The
match counts fail closed. One bug surfaced and was fixed at the same time: `dataclasses`
resolves a frozen class's annotations through `sys.modules[cls.__module__]`, so the module
must be registered before it executes.

## Deliverable 2 — runtime and manifest

Staged `inflate.py` SHA-256 `7333e8830ffc1a0b33f8ee9fdfd321f95b9b12f8f425bff09f0209a1860c4f17`,
2,476 B — byte-identical to the tree the fs2 row actually ran. Its diff against the **live PR
tree** is exactly the two pin lines (`evidence/diff/inflate_py_two_pin_lines.diff`), which is
the charter's proof obligation, discharged. `diff -rq` over the whole tree returns exactly
four files: `README.md`, `compress.py`, `inflate.py`, `MANIFEST.sha256`. Nothing under
`runtime/` or `cpr1/` moves. MANIFEST rehashed: 39 rows, three changed.

ps1's MEASURED correction still binds and is worth restating, because it is why the packet is
built from the live PR tree rather than copied from the fired tree: the fs2 fire tree carries
the **g8v1-era** `README.md` and the generation-8 `compress.py`. Copying it would regress the
public README's credits and the operator-approved fr2c TODO wording.

Tree digests, MEASURED by the compliance checker's own producer: staged
`18041600…` (with custody files) / `963955e8…` (without); evaluated `915d25f9…` / content
`739d14ce…`.

## Deliverable 3 — documents

README: numbers only (`evidence/diff/readme_numbers_only.diff`) — 180,022 → 180,023,
0.14786319521362173 → 0.14784474152757654, the archive SHA-256, the timing sentence
(587.3 s inflation + 49.4 s evaluation), and bullet 4 rewritten to name **both** stages:
"a per-pair frame-0 selector re-selection and a pose-carrier re-solve on those pairs, +21 bytes
total, PoseNet 0.00000637 → 0.00000614, segmentation output unchanged". The disclosure
paragraph, the credits and the three TODO bullets are **unchanged**. `report.txt` is the
harvested evaluator artifact verbatim (`e9ab4206…`, 664 B). `PR_BODY_DELTA.md` lists six
number/link edits and states what must not change. **Title stays
`semantic_joint_ctxmix (0.148)`** — 0.14784474152757654 rounds to 0.148 at three decimals, so
no title edit is needed.

## Deliverable 4 — custody seal, not fired

`SEAL_VALID` at
`/Volumes/VertigoDataTier/pact/ddm_ps2_pr140_update_prep/SEAL_ps2_staged_fs2_packet_tree_contest_cuda.json`,
seal sha `60b8d3db5cd5f9755f08146008b5a6570c7769fcb559eb7ddf93ab799e24035b`. Candidate
`ps2_staged_fs2_packet_tree`, axis `contest_cuda`, runtime 43 files / 937,350 B digest
`eb2db2cf…`, archive `a8f3a379…` / 180,023 B verified against the measured bytes, admit bar
`net dS < 0.0` against the pointer's own 0.14784474 with zero tolerance. Three falsifiers are
pre-registered, the first of which is the whole point: the staged tree must reproduce the fs2
row exactly (pose 6.14e-06, seg 0.00020139, 180,023 B, S 0.14784474152757654), and any
component drift means the staged tree is not the evaluated object. **MAIN fires; this arm did
not.** Firing it is what would cure red 4 below.

## Deliverable 5 — compliance dry-run: 83 GREEN / 4 RED of 87, and ZERO new red classes

`scripts/pre_submission_compliance_check.py --contest-final --strict` against the staged packet
with the fs2 SHA-256/size, the harvested auth-eval JSON, and the evaluated runtime tree SHA-256.
Four runs are retained; **r4 is terminal**.

| | ps1 (fs1) | pq12 (frozen) | **ps2 (fs2)** |
|---|---:|---:|---:|
| green | 78 | 80 | **83** |
| red | 7 | 7 | **4** |
| total | 85 | 87 | **87** |

**The three `dispatch_claim_*` reds ps1 reported are now GREEN.** MAIN appended a canonical
terminal row for lane `ddm_fs2_t4_carrier_resolve_20260904` at 2026-09-04T23:28:25Z that binds
the full archive SHA-256 **and** the full runtime tree SHA-256 and carries the status prefix
the checker recognises (`completed_contest_cuda_exact_eval_harvested`). That is exactly the
ledger-shape cure ps1 asked for, landed. Verified: `dispatch_claim_terminal_row`,
`dispatch_claim_successful_exact_eval_terminal_row`, `dispatch_claim_terminal_archive_sha_bound`,
`dispatch_claim_terminal_runtime_tree_sha_bound` and `dispatch_claim_prior_active_row` all pass.

**The ledger moved underneath this arm and I am recording that rather than smoothing it.** Run
r1 read the pre-23:28:25Z ledger and saw 3 matching rows with the older
`completed_modal_auth_eval_harvested_S_…` status and a 12-character runtime-tree prefix; r2–r4
read 4 rows. The dispatch-claim differences between r1 and r2 are MAIN's append, not an edit of
mine. r3 is a deliberate variant that binds `--expected-job-id` to the Modal call id instead of
the `_r2` job id: it scores 79/7 and shows the complementary red set, which is why the `_r2`
job id is the right binding.

The four survivors, all pq12-frozen, **none waived**:

| red | disposition |
|---|---|
| `auth_eval_raw_promotion_policy_blockers_absent` | STRUCTURAL-RECORD — the raw emitter stamps its pre-adjudication blockers |
| `contest_cpu_auth_eval_exists` | RECORD-WITH-REASON — no CPU score for these bytes; none inherited; the prior same-lineage CPU attempt timed out at the 1,800 s inflation limit |
| `hosted_archive_manifest_supplied` | BLOCKED-ON-OPERATOR — nothing hosted yet; release-plan steps 3 and 4 clear it |
| `submission_runtime_tree_matches_auth_eval` | ENUMERATED-TREE-PROVED — the evaluated tree is the g8v1-era one; every file `inflate.sh` executes is byte-identical; curing the digest needs the custody row MAIN fires |

**No new red class. No red relabelled green.**

## Deliverable 6 — `RELEASE_PLAN.md`, not executed

Six steps: operator gate → branch update → push → **new** fork release tag
`semantic_joint_ctxmix-fs2` (a new tag, not an asset swap, so every older link keeps resolving
to the bytes it names) → fetchback verification that must show `a8f3a379…`, `180023` and
`HOSTED_BYTES_IDENTICAL` before the body is edited → `gh pr edit 140` with the operator's own
text. Every command is written out; none was run.

## Blockers and things I did NOT change

1. **BLOCKED-ON-OPERATOR-CONFIRM.** Nothing is pushed, hosted or edited. That is the p0 gate,
   not a defect.
2. **The custody row is unfired.** The seal is valid and waiting; MAIN fires it. Until it
   lands, `submission_runtime_tree_matches_auth_eval` stays red for a reason the packet states
   plainly.
3. **The README's "evaluated commit `1c9fbbf58716eb0f26bcdf2a91e3c89d0e4efdde`" is UNCHANGED,
   and a reader should know why.** It pins the public research-receipt repository
   (`adpena/comma-lab`), not the pact commit — the fs2 row's own pact commit is
   `7c65f118dd21d9f3b9e77078ba27d880d64f4c50`. My scope was numbers, and I have no verified
   newer public receipts-repo commit to point at, so I left the pin and am naming it here
   rather than silently carrying it forward. If the operator publishes a newer receipts commit
   before the PR body is edited, that line and the packet identity report both need it.
4. **I did not invent a `portable_runtime_content_tree_sha256`** for the generated
   `archive_manifest.json`, for ps1's reason: the pq12 stager produces that digest and this arm
   has no canonical producer for it. The key is absent with its reason recorded.

## Custody

| artifact | SHA-256 | bytes |
|---|---|---|
| `STAGING_REPORT.json` | `3e28730dd9a4f8da9732060a8996d38f15500914a8f8285ab16f9c12fc6fe097` | 14,704 |
| `RETENTION_MANIFEST.json` (151 artifacts, 3,834,882 B) | — | — |
| `evidence/rebuild/run{1,2,3_cleanstate}_archive.zip` | `a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6` | 180,023 |
| `packet/compress.py` | `605fdc1af4b1aec50d62798eecb49b0525ba7ebab42392825969a00cba03d8d8` | 238,604 |
| `packet/inflate.py` | `7333e8830ffc1a0b33f8ee9fdfd321f95b9b12f8f425bff09f0209a1860c4f17` | 2,476 |
| `packet/MANIFEST.sha256` | `451dd4b72964c3068ce17ad3d5feb2f5bfa99f5de9c8f92e5f229836da6d736d` | 3,605 |
| `report.txt` (evaluator, verbatim) | `e9ab42066d2bb6e23fa4375a01f6abb99b72e7cbd485b14943e6b6d7ac52a93b` | 664 |
| `PACKET_REPORT_fs2.txt` | `1e4b234b2a06f010db328095189f7b819c1399aa732faefffd4e97da67825f4c` | 4,614 |
| `PR_BODY_DELTA.md` | `0fcf685e7d227b66ad2f608e907c526bf87950c5d52c2a9acc5dbe4a5f608268` | 5,770 |
| `RELEASE_PLAN.md` | `323275378517f12a5033544b78e6acf0392708ea7751cddad82098b31c66b4d3` | 6,517 |
| `evidence/inputs/contest_auth_eval_fs2_t4_20260904.json` | `0c8f3390d96d19d4c10c37c1b5eee5450c777131a6bd0870729d92dff441d99f` | 51,385 |
| harvest `MODAL_REMOTE_RESULT.json` | `ebd7f808ce57adcd6bd2f6072b38bc0e24b014644da0c43b7b331461e306cee5` | — |
| seal | `60b8d3db5cd5f9755f08146008b5a6570c7769fcb559eb7ddf93ab799e24035b` | — |

Consumed, not re-measured: the T4 score row and its component values. Measured here: archive
and packet bytes and hashes, the stage-7 rebuild and its two repeats, the four negative
controls, the container identity control, the tree diffs and digests, and the compliance result.

## Equations leg (`tac.canonical_equations`)

No new equation and no re-fit. The stage-7 rebuild is a byte identity, not a law: it produces
no ΔS of its own, and the fs2 row's own equation anchor (`exchange_ratio_noise_floor_v1`, the
admissibility rule the fs2 arm's seal cites) is untouched by a packaging arm. If MAIN fires the
custody seal, that row measures the same bytes on a different tree and is still not an
equations-leg event — it is a custody binding. An equations-leg update would belong to the next
arm that moves a component.

## Hot-state line proposal

`PR #140 carries the afr1 bytes (180,002 B). The fs2 update packet is PREPARED at
/Volumes/APDataStore/pact/ddm_ps2_pr140_update_prep/ (stage-7 rebuild proved 3x, four negative
controls, 83 GREEN / 4 RED of 87 with ZERO new red classes — MAIN's canonical terminal row
cured all three of ps1's dispatch-claim reds). Custody seal SEAL_VALID and UNFIRED.
BLOCKED-ON-OPERATOR-CONFIRM: nothing pushed, hosted or edited.`

Own-vehicle frontier: **fs2 — S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]**,
archive sha `a8f3a379…27bb6`. This arm did not move it.
