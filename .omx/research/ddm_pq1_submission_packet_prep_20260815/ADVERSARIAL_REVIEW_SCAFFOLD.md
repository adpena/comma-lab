# Submission PR adversarial review scaffold

Status: `HOLD`, consecutive clean passes: `0/5`.

Any finding, however small, resets the counter to `0/5` after the finding is
fixed. A pass cannot be counted while the strict compliance chain is red.

| Round | State | Reviewer | Candidate archive SHA-256 | Compliance receipt SHA-256 | Findings | Counter after round |
|---:|---|---|---|---|---|---:|
| 1 | NOT_RUN | unassigned | pending exact active generation | pending | not reviewed | 0/5 |
| 2 | NOT_RUN | unassigned | pending exact active generation | pending | not reviewed | 0/5 |
| 3 | NOT_RUN | unassigned | pending exact active generation | pending | not reviewed | 0/5 |
| 4 | NOT_RUN | unassigned | pending exact active generation | pending | not reviewed | 0/5 |
| 5 | NOT_RUN | unassigned | pending exact active generation | pending | not reviewed | 0/5 |

Each round must independently cover:

- exact archive/member SHA, size, grammar, parse-back, and deterministic repeat;
- executable runtime closure, dependency behavior, 30-minute budget, and
  absence of hidden or uncharged score-bearing payloads;
- both exact score axes on the same archive bytes, recomputation from
  components, hardware labels, upstream snapshot, and runtime tree binding;
- borrowed-substrate accounting against the shipped sections and mechanisms;
- public source pin, hosted archive URL and manifest, report linkage, and
  anonymous availability;
- public-text hygiene, including paths, infrastructure, credentials, provider
  records, unsupported originality claims, and machine attribution;
- PR template conformity, contest status statement, operator attribution, and
  explicit scope of every pending or advisory number;
- swap delta versus the prior generation and confirmation that no stale receipt
  crossed the byte boundary.

The fifth clean pass authorizes only a recommendation to MAIN. It does not
authorize submission, push, or hosting.
