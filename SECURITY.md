# Security policy

## Reporting a vulnerability

If you discover a security issue in `comma-lab` or the `tac` library, please
report it privately rather than opening a public issue.

Contact: `adpena@users.noreply.github.com` (GitHub-routed alias).

When reporting, please include:

- a brief description of the issue and its impact;
- a minimal reproduction (if possible);
- the affected file, function, or commit SHA;
- any suggested remediation.

We aim to acknowledge security reports within 7 days and to issue a patch or
mitigation guidance within 30 days for confirmed vulnerabilities. Public
disclosure should be coordinated to give downstream consumers time to upgrade.

## Scope

In scope:

- the `tac` Python package (`src/tac/`);
- the `comma_lab` Python package (`src/comma_lab/`);
- repository tooling under `tools/` and `scripts/`;
- CI workflows under `.github/workflows/`.

Out of scope:

- the pinned upstream snapshot under `upstream/` (report upstream issues to
  [commaai/comma_video_compression_challenge](https://github.com/commaai/comma_video_compression_challenge));
- third-party dependencies (report to their respective maintainers);
- public-PR intake clones under `experiments/results/public_pr*_intake_*/`
  (these are pristine bytes-identical mirrors of upstream PRs).

## Supply-chain considerations

This repository pins all hard runtime dependencies to major-version ranges and
documents copyleft obligations in `THIRD_PARTY_NOTICES.md`. The optional
`[pr86_replay]` extra contains an LGPL-2.1-or-later dependency (`pyppmd`); the
default `pip install tac` install path is permissive-only (MIT / Apache-2.0 /
BSD-3-Clause).

## Acknowledgments

Security reports are credited in `CHANGELOG.md` unless the reporter requests
otherwise.
