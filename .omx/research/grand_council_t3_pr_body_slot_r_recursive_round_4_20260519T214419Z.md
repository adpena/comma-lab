---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author, Rudin, Daubechies, Karpathy, Carmack, Hassabis, Filler, TimeTraveler, JackFromSkunkworks]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Yousfi
    verbatim: "CRITICAL: README L99 + L102 cite `python upstream/evaluate.py --device cpu /tmp/inflate_out` and `--device cuda /tmp/inflate_out`. Upstream evaluate.py argparse does NOT accept a positional argument; the canonical CLI per experiments/contest_auth_eval.py L1147-1154 is `python upstream/evaluate.py --submission-dir <dir> --uncompressed-dir <videos> --video-names-file <file> --device <cpu|cuda> --report <path>`. A maintainer running the README's command verbatim would get `evaluate.py: error: unrecognized arguments: /tmp/inflate_out`. This is the canonical Yousfi-emulator finding — broken instructions in the most important reviewer-facing surface."
  - member: Karpathy
    verbatim: "Sister CRITICAL: README L96 `bash /tmp/archive_dir/inflate.sh /tmp/archive_dir /tmp/inflate_out videos/0.mkv` passes `videos/0.mkv` as 3rd arg. Per the canonical inflate.sh signature (cat inflate.sh L7: `FILE_LIST=\"$3\"`) and the loop body (`while IFS= read -r line; do ... done < \"$FILE_LIST\"`), the 3rd arg is a FILE LIST text file (newline-separated video names), NOT a video name. Same bug class as the L99 CLI break — wrong arg semantics in reviewer-facing instructions. The 60-second smoke at L70-75 correctly uses `echo \"0.mkv\" > /tmp/list.txt && ... /tmp/list.txt`."
  - member: Hassabis
    verbatim: "Strategic: these 2 README findings are BLOCKING for the hire-worthy posture standard. A reviewer running the verbatim commands gets immediate errors. The 60-second smoke (L66-78) is correct because Slot Q-Q wrote it from canonical evidence; the 'full' instructions at L80-103 were sketched without empirical validation."
  - member: Contrarian
    verbatim: "Round 4 Yousfi-emulator lens finds 2 BLOCKING defects the prior 3 rounds did not surface. Lens rotation is empirically validated yet again as the correct discipline."
  - member: Selfcomp
    verbatim: "Fix in scope: rewrite the full-eval instructions at README L80-103 to match the canonical CLI per experiments/contest_auth_eval.py. The 60-second smoke at L66-78 is correct; preserve verbatim."
  - member: Rudin
    verbatim: "Convergence trend: Round 1 found 1, Round 2 found 3, Round 3 found 2, Round 4 found 2. Counter stays at 0. Round 5 expected to surface 0-1 finding if convergence holds; 5-round safety cap may bind before counter reaches 3 unless Round 5+6+7 are clean."
  - member: PR95Author
    verbatim: "Confirming Yousfi's CLI canonical: PR101 GOLD README similarly uses the canonical --submission-dir form. Medal-class precedent matches."
council_assumption_adversary_verdict:
  - assumption: "The full-evaluation instructions at README L80-103 were validated end-to-end against upstream/evaluate.py."
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "The CLI `python upstream/evaluate.py --device cpu /tmp/inflate_out` is invalid argparse for upstream/evaluate.py (canonical takes --submission-dir / --uncompressed-dir / --video-names-file / --device / --report; verified via upstream/evaluate.py --help). Instructions are factually broken; a reviewer running verbatim gets argparse error."
  - assumption: "inflate.sh 3rd argument is a single video filename."
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "inflate.sh L7 declares FILE_LIST=\"$3\"; L22+ loop is `while IFS= read -r line; do ... done < \"$FILE_LIST\"`; the 3rd arg is a text file path containing newline-separated video names. README L96 passes 'videos/0.mkv' as 3rd arg which is treated as the file list path itself; if the file doesn't exist, inflate.sh fails on `< $FILE_LIST` redirect."
  - assumption: "The Yousfi-emulator lens converges to zero defects."
    classification: HARD-EARNED-EMPIRICALLY-PARTIALLY-CONFIRMED
    rationale: "Round 4 finding count (2) less than Round 2 (3), more than Round 3 (2). Trend is decreasing but not zero. The lens is doing its job of catching reviewer-facing defects."
  - assumption: "The 60-second smoke at L66-78 is correct because Slot Q-Q sourced it from canonical evidence."
    classification: HARD-EARNED-EMPIRICALLY-CONFIRMED
    rationale: "Smoke uses `echo \"0.mkv\" > /tmp/list.txt && ... bash inflate.sh /tmp/data /tmp/out /tmp/list.txt` — correct file-list semantics. The expected sha `d1afc583b01ff4a7...` matches the codex byte-identity verification ledger. Smoke is the canonical surface."
council_decisions_recorded:
  - "op-routable #1 (REVISION #1 BINDING — Yousfi + Karpathy + Hassabis + Selfcomp): rewrite README full-eval instructions at L80-103 to match the canonical upstream/evaluate.py CLI per experiments/contest_auth_eval.py L1147-1154. Required args: --submission-dir, --uncompressed-dir, --video-names-file, --device, --report. Also fix the inflate.sh step 3 to pass a FILE LIST (path to a text file containing video names), not a video filename directly."
  - "op-routable #2 ADVISORY: the 60-second smoke at L66-78 is correct; preserve verbatim and use as canonical pattern for the full-eval rewrite."
  - "op-routable #3 (Slot F D5 unblock signal): VERDICT=PROCEED_WITH_REVISIONS; counter resets to 0; Round 5 follows after revision #1 lands."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
finding_action_class: pursue
finding_followup_dispatch_envelope_usd: 0
finding_canonical_path: gates_pr_submission_d5_via_recursive_review_iteration_round_4
related_deliberation_ids:
  - t3_council_pr_body_slot_r_recursive_round_3_20260519T213942Z
  - t3_council_pr_body_slot_r_recursive_round_2_20260519T213502Z
  - t3_council_pr_body_slot_r_recursive_round_1_20260519T212810Z
---

# T3 Grand Council Symposium — SLOT R'' Recursive Round 4 of N

## Round 4 perspective rotation per CLAUDE.md "Recursive adversarial review protocol" item 1

**Round 4 lens**: Yousfi-emulator — "would the maintainer's actual paired auth-eval, running the README's instructions verbatim, reproduce the claimed scores?"

## Section 1: Yousfi-emulator findings

### Finding A: README full-eval CLI is broken (CRITICAL)

README L99 + L102:

```bash
python upstream/evaluate.py --device cpu /tmp/inflate_out  # expect 0.192051
python upstream/evaluate.py --device cuda /tmp/inflate_out  # expect 0.226210 on T4
```

Canonical CLI per `upstream/evaluate.py --help` AND per `experiments/contest_auth_eval.py` L1147-1154:

```bash
python upstream/evaluate.py \
  --submission-dir <inflate_output_dir> \
  --uncompressed-dir <videos_dir> \
  --video-names-file <list.txt> \
  --device <cpu|cuda> \
  --report <report.txt>
```

A reviewer running the README verbatim would get `evaluate.py: error: unrecognized arguments: /tmp/inflate_out`. The score never gets reproduced. The hire-worthy posture is structurally broken at this surface.

### Finding B: README inflate.sh step 3 has wrong arg type (CRITICAL)

README L96: `bash /tmp/archive_dir/inflate.sh /tmp/archive_dir /tmp/inflate_out videos/0.mkv`

Per `inflate.sh` L7: `FILE_LIST="$3"`. The 3rd arg is a TEXT FILE PATH containing newline-separated video names. The L22-35 loop is `while IFS= read -r line; do ... done < "$FILE_LIST"`. Passing `videos/0.mkv` as 3rd arg means inflate.sh tries to read line-by-line from a file named `videos/0.mkv` which is a binary video, not a text list — causes `< $FILE_LIST` redirect to fail OR garbage output.

Compare to the 60-second smoke at L70-75 which correctly does:
```bash
echo "0.mkv" > /tmp/list.txt && ... bash inflate.sh /tmp/data /tmp/out /tmp/list.txt
```

Same bug class as Finding A — instructions sketched without empirical validation.

## Section 2: Convergence assessment

| Round | Lens | Findings | Counter after |
|---|---|---|---|
| 1 | Default | 1 critical (CUDA fabrication) | 0 (reset) |
| 2 | External-adversary | 3 (permalink staleness / line 19 inconsistency / gitignored path) | 0 (reset) |
| 3 | Mechanism-extrapolation | 2 (rate precision / K=8 misattribution; 1 in-archive deferred) | 0 (reset) |
| 4 | Yousfi-emulator | 2 (README full-eval CLI broken / README inflate.sh arg-type wrong) | 0 (reset) |

Findings count is stable at 2; convergence not yet achieved. **5-round safety cap may bind unless Rounds 5+6+7 are clean.**

**Counter**: 0 (Round 4 produced 2 findings; resets).

**Next**: SLOT R'' coordinator applies revision #1 (rewrite README full-eval) via canonical serializer; emits Round 5 with rotated lens. Candidate Round 5 lens: "deterministic-replay byte-pedantic adversary" (would the maintainer's byte-identity verification reproduce all SHA-256s cited?).

## Section 3: Assumption-challenge axis (item 8 mandatory)

**Shared assumption Round 4 operates within**: that README full-eval instructions were validated end-to-end before being landed in Slot Q-Q's submission_dir/README.md commit.

**Empirically falsified**: the L99 + L102 CLI invocations are factually broken. They were sketched from intent (cite expected scores per device) without invoking `python upstream/evaluate.py --help` to verify the actual argparse contract. Same for L96 inflate.sh invocation.

**Op-routable extension** (NON-BLOCKING): future README generation pipelines should auto-validate every shell command in reviewer-facing docs by running `<cmd> --help` and pattern-matching the args. Catalog #287 docstring-overstatement sister gate at README CLI invocation level would extinct this class structurally.

## Section 4: Continual-learning anchor

This deliberation will be appended to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor` per Catalog #300 v2 frontmatter.

## Section 5: Cross-references

- Rounds 1-3: `.omx/research/grand_council_t3_pr_body_slot_r_recursive_round_{1,2,3}_*.md`
- Canonical upstream/evaluate.py CLI: `experiments/contest_auth_eval.py` L1147-1154
- Canonical inflate.sh FILE_LIST contract: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.sh` L7 + L22-35
- CLAUDE.md "Apples-to-apples evidence discipline" + "Beauty, simplicity, and developer experience" + "Recursive adversarial review protocol"


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:grand-council-T3-PR-body-slot-R-recursive-round-4-trigger-tokens-in-recursive-review-not-new-equation -->
