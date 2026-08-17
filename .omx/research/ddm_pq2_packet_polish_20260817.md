# ddm_pq2 — the packet is re-targeted onto rr4, the end-to-end compression script exists and rebuilds the archive byte-for-byte, and four honest reds remain

**Date:** 2026-08-17
**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]` for everything this arm
measured. `score_claim: false`, `promotable: false`. **No Modal, no dispatch, no exact eval.
Spend: $0.00.** No submission, no publication, no GitHub write.

---

## ANSWER

**The packet is READY-except-OPERATOR-GO on four named blockers, none of which an arm can
honestly clear.** Compliance went from pq1's 78/86 to **81/85** on a candidate that is 2,341
bytes smaller and 0.00156 lower in score. The four survivors are a checker/schema naming drift,
a policy adjudication MAIN owns, the missing contest-CPU row, and the fail-closed dependency
bootstrap. I cleared the lane-ledger reds and the runtime-tree red by fixing the real defects
underneath them, not by relabelling.

**The end-to-end compression script is built, sanitized, and PROVEN.** `--stage build` rebuilt
the archive into a clean store and the sha assertion **passed**: `35ac2b9b…`, 181,161 B,
determinism repeat byte-identical, 7 of 8 sections byte-identical to the base. `--stage encode
--resume` replayed from the retained checkpoint in **1.55 s** and returned token stream
`6c3757bd…` — the pinned value. This is a genuinely stronger artifact than the field norm:
**PR138's `compress.sh` is 27 lines that copy a frozen archive and verify its hash.** It never
reads the video, never trains, never runs its encoder, and never reproduces bytes from source.
Ours does.

**A charter premise is FALSIFIED and I am reporting it rather than working around it.** The
charter states the rr4 runtime carries "the rr3 constriction declared-dep." **It does not.** An
AST scan of all 32 runtime files returns third-party imports `brotli`, `numpy`, `torch` — and
`constriction` appears nowhere in the tree. The only self-installed dependency is
`Brotli==1.2.0`. I smoked that one instead, end to end, and it passes.

**The custody gap MAIN pre-cleared was real, and larger than described.** rv2 called the
inherited `GENERATION_RECEIPT.json` / `RECEIVER_PARSEBACK.json` a labeling gap; that is correct.
But the measured file set shows those two stale receipts **are inside the hashed runtime tree**,
so rv2's own prescribed cure — writing `CUSTODY_SUPERSEDED.json` beside them — would have
changed the pinned tree hash. I wrote it, measured the consequence, **moved it out to the store
root**, and put the correction in `README.md` instead, which the evaluated manifest excludes.
The pinned hash `7acedb07…` is intact and now reproduces from a clean staging directory.

---

## 1. STORES CONSULTED

* `.omx/research/charters/ddm_pq2_packet_polish_and_e2e_script_20260817.md` — the charter.
* `.omx/research/ddm_pq1_submission_packet_prep_20260815/` — the generation-0 packet, its
  `SWAP_PROCEDURE.md`, compliance table, PR draft, and review scaffold.
* `.omx/research/ddm_rv2_frontier_adversarial_review_r1_20260817.md` — findings 1 and 2, the
  custody adjudication, and the F5 economics correction.
* `.omx/research/ddm_rr4_cuda_prob_reencode_20260817.md` + `ddm_rr4_t4_verdict_pointer_move_20260817.md`
  — the mechanism, the byte targets, and the explicit warning that the two inherited receipts
  "must not be cited."
* `/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/RESULT_{build,encode,parseback_v2,receiver_build}.json`
  — the three authoritative binding receipts.
* `experiments/results/ddm_rr4_cuda_exact_contest_cuda_20260817_r1/returned_artifacts/` — the
  recovered exact-authority payload (8 artifacts) MAIN restored via FO-1.
* `.omx/state/active_lane_dispatch_claims.md` — the real lane and job ids.
* Source read directly: `experiments/ddm_rr2_encoder_byteclose.py`,
  `experiments/ddm_rr2_receiver_close.py`, `scripts/pre_submission_compliance_check.py`,
  `tools/fire_modal_auth_eval.py`, `tools/claim_lane_dispatch.py`,
  `experiments/modal_auth_eval_cpu.py`.

## 2. The candidate custody: VERIFIED, and the falsifier branch did not fire

Hashed the bytes myself: `35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956`,
**181,161 B**, single stored member `p`, 181,061 B, CRC32 885609521. I re-derived the score
independently from the reported components:

```
100·0.00029611 + √(10·0.00000688) + 25·181161/37545489
= 0.029611 + 0.008294576541331089 + 0.12062767380656568
= 0.15853325034789678
```

All 17 digits, matching `canonical_score` in the recovered receipt.

Per MAIN's update, rv2 independently verified the row custody sound, so the charter's
custody-gap branch does **not** fire and I proceeded on rr4 rather than preparing both candidates.

## 3. The end-to-end compression script (the operator-named deliverable)

`experiments/ddm_pq2_compress_e2e.py`. It **orchestrates the two instruments that actually
produced the candidate** rather than reimplementing them, so what it proves is what the shipped
pipeline does.

| Stage | Status | Receipt |
|---|---|---|
| A `provenance` | DOCUMENTED, NOT RE-RUN, and labelled as such | reproducing the checkpoint from raw video is multi-day GPU compute; the script says so instead of pretending |
| B `encode --resume` | **PASSED**, 1.55 s | token stream `6c3757bd52a18d3c…`, 110,512 B, code length 110511.27763690146, 9,613 warm contexts |
| B `build` + `verify` | **PASSED** | archive `35ac2b9beb7e6fa8…`, 181,161 B, determinism repeat byte-identical, 7/8 sections byte-identical |
| C `decode` | available; honest limitation documented in the docstring | it is a repository proof, not judge-runnable; a judge uses the shipped `inflate.sh`, which needs no private input |

**No private paths.** Every input root arrives through `--inputs-json` or an environment
variable and is sha256-verified before any stage runs; `--emit-inputs-template` prints the
schema. Achieving this required a four-line change to `ddm_rr2_encoder_byteclose.py` replacing
four hardcoded roots with `_env_path(variable, default)`. **The defaults are unchanged, and the
change is proved inert rather than asserted inert**: the rebuild through the overridden path
returns the same archive sha. That is the control.

## 4. Compliance: 81/85, and how the reds moved

Ran the real `scripts/pre_submission_compliance_check.py --contest-final` against a staged
submission directory. Receipt sha256 `ba596648e1e1f6a5b8c3eb4e5af28bca8eccc9597f5598a8119de3bbc3c43656`.

Trajectory, all by fixing causes: **21 reds → 11 → 8 → 6 → 4.**

| Red | Cause found | Cure |
|---|---|---|
| 9 doc/report/label reds | I pointed the checker at the runtime tree, which has no `report.txt` or `README` | staged a real submission directory |
| checker **crashed** mid-run | ExFAT creates AppleDouble `._*` files on copy; the checker's AST reader hit non-UTF-8 bytes | `COPYFILE_DISABLE=1` + strip; **this would have ridden along into a submission** |
| `submission_runtime_tree_matches_auth_eval` | the evaluated tree is exactly 32 files, excluding `archive.zip`/`README`/`report` and `__pycache__`, and **including** the two stale receipts | stage exactly that set; the pinned `7acedb07…` now reproduces |
| 4 `dispatch_claim_*` reds | I had guessed the lane ids; the real ones are `ddm_rr4_platform_exact_reencode_exact_contest_cuda_20260817` / `ddm_rr4_cuda_exact_r1`, and the existing terminal row carried **empty notes**, binding neither sha | appended a terminal row binding archive sha, runtime-tree sha, and the score, in the canonical status vocabulary |
| `public_source_repo_link_present` | my README rewrite dropped the repo URL | restored repo + pinned-revision links |

### The four that remain, and why an arm cannot honestly clear them

1. **`auth_eval_schema_metric_consistency`** — the receipt records
   `canonical_score_source = report_8dp_components_plus_exact_archive_bytes`; the checker
   demands `score_recomputed_from_components`, which is present only as
   `legacy_canonical_score_source_alias`. This is **checker-versus-schema drift, not a defect in
   the row**. Editing the receipt to satisfy a checker would be forging evidence. MAIN owns
   whether the checker learns the newer source name.
2. **`auth_eval_raw_promotion_policy_blockers_absent`** — blockers include
   `pre_submission_compliance_check_not_recorded`, which is circular (it clears when this run is
   recorded), plus `cpu_leaderboard_reproduction_not_adjudicated`. MAIN adjudication.
3. **`contest_cpu_auth_eval_exists`** — the CPU row. Fire order prepared, not fired. §6.
4. **`submission_runtime_has_no_network_install_or_local_paths`** — `inflate.sh:27`, the
   fail-closed dependency bootstrap. Verified working (§5). Needs MAIN's recorded policy
   acceptance, which is the e4 precedent; a WAIVER is a policy act, not an arm act.

## 5. Dependency closure — the charter premise is falsified, and the real dep passes

AST scan over all 32 runtime files: third-party imports are **`brotli`, `numpy`, `torch`**.
**`constriction` does not appear anywhere in the tree.** The charter's "rr3 constriction
declared-dep" is not a property of this runtime. I smoked the dependency that actually exists.

Bare-venv bootstrap smoke, all four legs green:

1. fresh `uv venv`, Brotli absent → the guard fails → the bootstrap path fires, as designed;
2. the exact `inflate.sh:27` command installs `brotli==1.2.0`;
3. the guard then passes and a compress/decompress round trip returns the input;
4. with `uv` absent the script exits **69** — fail-closed, no silent degradation.

The bootstrap fetches a pinned wheel only. No model, table, or video-derived payload crosses the
network, so rule 118 is not implicated.

## 6. CPU fire-order: PREPARED, NOT FIRED — with the corrected economics and a blocker

`CPU_AXIS_SEALED_FIRE_ORDER.json`, every flag verified against the real `local_entrypoint`
signature of `experiments/modal_auth_eval_cpu.py` (line 1163). Nothing invented.

**Economics, corrected per rv2 F5:** Modal headroom is **$1.38**, not ~$13. At the ~$0.40 CPU
precedent this row consumes **~29% of everything remaining**. I present both facts and take no
position: PR135 and PR138 both declare GPU-required evaluation, so the field norm makes a CPU
row plausibly optional; our packet's "requires GPU" answer is honest without it. **MAIN
adjudicates.**

**A blocker I found while preparing it:** `tools/fire_modal_auth_eval.py`, the ONE canonical
deterministic firing path, targets `experiments/modal_auth_eval.py::main` and is **CUDA-only**.
There is no CPU equivalent, so a CPU row must currently be hand-assembled — precisely the hazard
the hand-assembled-dispatch law names as an error factory. **Cure the tool before firing the
row.**

## 7. Field intake — PR 137 and PR 138

Detached custody at `/Volumes/APDataStore/pact/ddm_pq2/intake/`. Repo untouched, nothing
published.

* **PR 138 `opal_v1`** (ccastillo1043), archive `bd9a4714…`, **182,040 B** — both the expected
  sha prefix and the expected size **confirmed**. Claims `0.1591495384`.
* **Token-delta claim CONFIRMED exactly: 114,706 → 110,022 B, Δ −4,684**, and every non-token
  section is byte-identical to PR135 at sha256 level. The accounting closes to the byte.
* **Mechanism:** a lossless entropy transcode of PR135 with **no retrain**. It splits PR135's
  five-class law by its rank-one maximal projector into the argmax class versus the other four
  and learns only probability transport between those sectors, online from the decoded prefix
  over 55 causal context families. **Zero learned bytes added**; ~49.4 MB of state regenerated at
  decode. Storage traded for runtime.
* **Its compression script does not reproduce its archive** — 27 lines that copy a frozen
  `archive.zip` and check its hash. It also ships two merge blockers: `verify_submission.py` is
  called but absent from the PR, and `archive.zip` is not in the tree while `inflate.py`
  hard-fails without it.
* **Neither PR 137 nor PR 138 carries a maintainer eval-bot score.** Both figures are
  author-claimed. So is ours until this PR is run. The packet says exactly that.
* **PR 137 `metric_shift_av1`**: 2.04 claimed, ~12.8× worse; pose alone is 0.888. Not
  competitive. One reusable observation: AV1 film-grain synthesis as cheap SegNet texture.

**Structurally, PR138 is the same move as ours** — re-encode a frozen PR13x token stream with a
learned decode-side prior, zero added bytes. They took 4,684 B off a 186,724 B base; we took
1,598 B off a 182,759 B base. Their base was larger.

## 8. Red burn-down performed

* 3 lane-ledger binding defects → **closed** by appending a terminal row that binds archive sha,
  runtime-tree sha, and the score (the prior terminal row's notes were empty).
* GitHub comment census → **retried and completed**; the finding is that neither new PR has a
  maintainer eval comment.
* Hosted-manifest prep → the exact artifact list and shas are in `ARCHIVE_MANIFEST.json` and
  `PACKET_TARGET.json`. **Nothing published**; hosting remains blocked on operator authority.
* AppleDouble contamination → **new red found and closed** before it could reach a packet.

## 9. What I did NOT do

No submission, no PR opened, no hosting, no publication, no Modal or paid dispatch, no scorer
run, no n600 eval, no edits to `upstream/`, no writes to VertigoDataTier, no checkout of a public
PR into the shared worktree.

## 10. NEXT

| # | Item | State | Owner |
|---|---|---|---|
| 1 | Adjudicate the 4 remaining reds (checker drift, promotion policy, CPU row, dep-bootstrap waiver) | **OWED** | MAIN |
| 2 | Cure the CUDA-only firing tool before any CPU row | **QUEUED** | runtime owner |
| 3 | Review scaffold passes 2–5 on the frozen generation-2 packet | **QUEUED** | reviewers |
| 4 | Operator decision on hosting + source-pin visibility | **BLOCKED on operator** | operator |
| 5 | The opal mechanism deep-read (sector split, 55 context families) | routed | ddm_me1 |

**Retracted / not claimed:** no score claim, no promotion, no pointer move, no assertion that we
beat PR138's unverified number, and no claim that the packet is submittable — it is
READY-except-GO on four named blockers.

---

## Artifacts (ALWAYS KEEP THE PAYLOAD)

* `/Volumes/APDataStore/pact/ddm_pq2/e2e_smoke/` — the rebuild store, `retained/archive.zip`
  181,161 B `35ac2b9b…`, `RESULT_pq2_e2e.json`, `RESULT_pq2_encode_resume.json`, encode log
* `/Volumes/APDataStore/pact/ddm_pq2/submission_staging/` — the exact 32-file evaluated tree plus
  archive, README, report
* `/Volumes/APDataStore/pact/ddm_pq2/compliance_rr4_gen2.json` — the 81/85 receipt,
  sha `ba596648…`
* `/Volumes/APDataStore/pact/ddm_pq2/depclosure_smoke/` — the bare-venv bootstrap smoke
* `/Volumes/APDataStore/pact/ddm_pq2/intake/` — PR137/PR138 archives, diffs, metadata,
  `INTAKE_ANATOMY.md`, `section_anatomy.json`
* `/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/CUSTODY_SUPERSEDED.json` — at the store
  root, deliberately outside the hashed runtime tree
