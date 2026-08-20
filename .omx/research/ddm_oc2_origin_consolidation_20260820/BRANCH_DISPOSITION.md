# ddm_oc2 — branch and worktree disposition

`date_utc: 2026-08-20` · `owner: ddm_oc2` · authority: operator consolidation directive 2026-08-20
(push authorization for **our own** repos only; the contest PR is untouched and still gates on the
operator's one-line confirm, #1111).

## 0. The finding that frames everything

**`origin` for this repo is `git@github.com:adpena/comma-lab.git`, and that repository is PUBLIC.**
The pact research repo and the public comma-lab repo are the same repository. There is no second
clone to prepare. Everything committed to `main` here is published on the next push.

Baseline hygiene state, measured on tracked content before this arm committed anything:

| Check | Tracked baseline | Verdict |
|---|---:|---|
| Files containing `/Users/adpena` | 4,226 | pre-existing; not created by this arm |
| Files containing `/Volumes/` | 5,242 | pre-existing; not created by this arm |
| Tailscale / `100.64.0.0/10` IPs | **0** | clean, and stayed 0 |
| Credentials / API tokens / private keys | **0** | clean, and stayed 0 |

The absolute-path exposure is a real standing gap against CLAUDE.md's Public Disclosure Hygiene
rule ("local absolute paths ... out of GitHub/docs/site/public surfaces"). It is a **pre-existing
condition at scale**, not something this consolidation introduced, and it is owed to MAIN as a
separate decision. This arm did not widen the class knowingly: every file it committed was scanned
for secrets and fleet IPs and came back clean.

## 1. Branch disposition

Seven branches were unmerged against `main`. Verdicts below are from `git merge-tree` dry runs plus
a content check of whether `main` already carries the branch's files.

| Branch | Head | Ahead | Disposition | Why |
|---|---|---:|---|---|
| `ddm/sh1_integration_20260727` | `6a77427ca1` | 10 | **MERGED** | Clean merge, 39 files / 7,637 insertions. It is an integration branch that had already merged the three siblings below, so one merge discharges four branches. |
| `codexwt/ddm_cb1_perclass_carrier_byteclose_20260725T203310Z` | `2721704ab2` | 1 | **MERGED via sh1** | sh1 commit `3963938cc8` merged it cleanly. |
| `codexwt/ddm_wf7_seven_home_stream_waterfill_20260725T203257Z` | `e3c2140d3a` | 2 | **MERGED via sh1** | sh1 commit `8dd643c03a` merged it cleanly. |
| `codexwt/ddm_pf3b_52probe_joint_improving_hunt_20260725T202800Z` | `074955c6ad` | 2 | **MERGED via sh1** | sh1 commit `0bb5b9579c` merged it. |
| `codexwt/ddm_de1_20260803T112347Z` | `7a0d6f0abc` | 2 | **MERGED** | Clean; a 783-line description-efficiency derivation absent from `main`. See the supersession note below. |
| `ddm/dy1_scope_law_resolver` | `a9eac92166` | 1 | **CERTIFIED — SUPERSEDED, do not merge** | Content is already on `main` and `main` is a strict superset. Merging would **delete 93 lines** from `src/tac/witness_dsl/scope_laws.py` (branch 2 insertions / 93 deletions vs main). Add/add conflicts on its research md confirm main landed the work by another path. |
| `codexwt/einstein_kolmogorov_crux_20260719T212159Z` | `81dfb0ee68` | 1 | **CERTIFIED — SUPERSEDED, do not merge** | Same shape, larger stakes. Merging would **delete 3,253 lines** from `src/tac/preflight.py`. `main` already carries `src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py` (68,061 B) and `.omx/research/ddm_ek2_worktree_harvest_20260810/DISPOSITION.md` (11,813 B). The branch's own commit message predicted this: it recorded merge-to-main as owed because "main drifted 2+ wks on preflight.py/CLAUDE.md". The drift resolved in main's favour. |

### de1 supersession note

The de1 derivation is merged for its research value. One claim inside it is **stale**: at
`.omx/research/ddm_de1_description_efficiency_derivation_20260803.md:611,705,773` it argues that
`src/tac/boundary_math/contour_codec.py` carries a name/mechanism mismatch. That specific complaint
was addressed by the #913 rename after de1 was written. The derivation is merged as a dated
historical artifact; the contour_codec paragraphs are not current and must not be cited as a live
defect.

### Recommended cleanup for MAIN

`dy1` and `einstein_kolmogorov` should be **deleted or archived as refs**, not left unmerged. Left
alone they read as "owed work" forever, and the next consolidation arm will re-derive this same
superseded verdict. That is the rediscovery cost the anti-forgetfulness rule exists to prevent.

## 2. Worktrees

54 worktrees were swept by `ddm_wk2r` on 2026-08-03
(`.omx/research/ddm_wk2r_worktree_custody_20260803.md`, denominator 54/54). This arm re-enumerated
them: the branch heads are unchanged and every non-main worktree is still `CLEAN` or carries only
the residue wk2r already recorded. **No live-arm worktree was touched.**
`.omx/state/active_lane_dispatch_claims.md` shows the newest claims all in terminal
`completed_*` states, so no worktree was in active use during this pass.

The one worktree that mattered is `agent-ae029aa6d20642139`, which holds
`ddm/sh1_integration_20260727` — merged above, so it can now be pruned.

## 3. The residue wk2r flagged as `RESIDUE-needs-owner` on main

wk2r counted 167 uncommitted entries in the main worktree on 2026-08-03. By this arm's pass that
had grown to 280 porcelain entries / 1,049 files. **764 documents and 181 sources were committed**;
the remainder is bulk that belongs on the SSD tier by the disk rules. See
`WORKING_TREE_DISPOSITION.md` for the itemized split and the certify records.
