# Contribution-etiquette harvest

This is a read-only harvest of the contest's actual contribution surface. The
current repository README and pull-request template were checked locally. The
comment evidence comes from the retained 2026-05-20 GitHub API harvest of 113
issue comments, 17 inline review comments, and 5 review bodies across the 15
highest-activity contest PRs, supplemented by the retained PR #108 and PR #110
maintainer-comment receipts. A live GitHub API refresh was attempted on
2026-08-15 but the execution environment could not reach `api.github.com`; the
current public leaderboard was still visible through web search. Therefore the
quotes below are authentic retained receipts, but the claim is bounded to the
named PRs and snapshots rather than represented as a fresh complete census.

| Do | Receipt | Why it belongs in this packet |
|---|---|---|
| Follow the template literally and answer every maintainer question in plain language. | PR #110: the maintainer asked for the new template and an “easy to understand response” to the competitive-or-innovative question. | The draft keeps the required archive, report, host, build-cost, GPU, compression-source, changes, competitive/innovative, and comments surfaces explicit. |
| Explain whether the entry is competitive or technically new. | PR #108 was closed because its techniques were already established; the maintainer's rubric was better than the top entry or a new idea with potential. | The draft makes a narrow competitive claim and does not inflate the lossless composition into a whole-vehicle originality claim. |
| Host `archive.zip` outside the code tree and prove the downloaded bytes. | PRs #67, #71, #86, and #102 received variants of “host the zip file outside of the repo”; PR #102 specifically recommended drag-and-drop attachment. | The packet remains HOLD until an operator-authorized URL is fetched back and hash-verified. |
| Publish useful compression code and a readable write-up when it is real. | PR #95: “we are going to reward folks publishing their code even if not in top 3”; it also received the best-write-up prize. | The draft credits the lineage and exposes the exact compression-side script inventory, while refusing to call the current path-saturated scripts publication-ready. |
| Credit every inherited mechanism at the level a reviewer can verify. | PRs #98, #100, #101, #102, #103, #130, and #135 all state their upstream lineage; PR #100's addition was accepted as potentially novel because others adopted it. | The packet includes a section-by-section byte and SHA accounting instead of a whole-system originality claim. |
| Keep CPU and CUDA evidence separate and disclose the exact hardware. | PR #103's public discussion established that small pixel differences can magnify as scores tighten and that hardware axes need explicit treatment. | The exact CUDA row is reported; the exact-byte CPU axis is ADJUDICATED MEASURED-INFEASIBLE (inflation 3,422.7 s vs the 1,800 s budget on 4-thread x86_64; no CPU score exists or is claimed — a measured boundary, not a pending item). |
| Keep the body concise and the code reviewable. | PR #101's prize-winning body was about 15 lines; PR #95 used the template with a compact technical summary and linked a longer write-up. | Detailed custody stays in the packet; the eventual public body should retain only decision-relevant evidence. |

| Don't | Receipt | Packet consequence |
|---|---|---|
| Do not check the large archive into the PR. | Repeated maintainer comments on PRs #67, #71, #86, and #102; the template says to use a curl-compatible upload link. | Hosting remains an operator-authority blocker, not a placeholder URL. |
| Do not mutate repository-wide dependency files for one submission. | PR #74 received the direct review comment “can you revert this?” on `pyproject.toml`. | Dependency closure must stay inside the submission/runtime contract and must be reconciled with the strict no-network-install check. |
| Do not resubmit established tricks without a competitive result or a clear new mechanism. | PR #108 closure. | The draft explicitly answers the rubric and keeps the borrowing table adjacent to the claim. |
| Do not move evaluation disputes to a private channel. | PR #103: “trying to influence things privately is not the way to do so”. | Any future host-axis discrepancy belongs in the public PR discussion after operator approval. |
| Do not overclaim borrowed work, hide an axis, leak internal paths, or include provider transcripts. | PR #95/#100 lineage discussions plus the public template and retained pre-submission audits. | Public-hygiene scans are a hard gate and the packet carries no machine attribution. |
| Do not open serial PRs while the packet is incomplete. | The maintainer's PR #108 rubric and the charter's one-PR rule. | Strict compliance and five consecutive clean reviews must finish before a single operator-authorized PR. |

## Source receipts

- Repository rules and format: `upstream/README.md` and
  `upstream/.github/pull_request_template.md`.
- Comment-corpus receipt:
  `.omx/research/pr_comments_mining_top_15_prs_for_actionable_signal_20260520T060250Z.md`.
- PR #95 full body/comment receipt:
  `.omx/research/pr_95_full_artifact_deep_research_against_our_submission_20260519T192300Z.md`.
- PR #108 closure receipt:
  `.omx/research/pr_submission_yousfi_non_merge_response_template_20260519T182635Z.md`.
- PR #110 template-request receipt:
  `.omx/research/pr_110_body_audit_yousfi_compliance_innovation_framing_review_20260526T134734Z.md`.
- Later accepted-source convention: retained PR #130 submission source at
  `experiments/results/public_pr130_intake_20260725_fable/source/` and PR #135
  intake memo `.omx/research/ddm_pi135_pr135_intake_20260810.md`.
