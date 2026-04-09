# contest state snapshot — 2026-04-06

This is a dated external-state note for the public comma.ai video compression challenge.

## sources checked

- official leaderboard: `https://comma.ai/leaderboard`
- contest repo: `https://github.com/commaai/comma_video_compression_challenge`

## official leaderboard state

As checked on 2026-04-06, the official leaderboard for the lossy video compression challenge shows:

- `2.1` — `svtav1_cheetah` (`#24`)
- `2.1` — `av1_sharp1_adaptive` (`#23`)
- `2.1` — `svtav1_45pct_unsharp` (`#20`)
- `2.2` — `svt_av1_lanczos_fg` (`#18`)
- `2.6` — `h265_g16_512x384_veryslow` (`#21`)
- `3.3` — `h265_tuned` (`#22`)
- `4.4` — `baseline_fast` (`#1`)
- `25.0` — `no_compress` (`#0`)

The official leaderboard is the authoritative public result surface.

## contest-repo state

As checked on 2026-04-06, the public GitHub repo page showed:

- `25` forks
- `2` open pull requests

Fork count alone should not be interpreted as proof of active or competitive experimentation. A fork only proves someone copied the repository. Public leaderboard entries are stronger evidence because they link to evaluated public PRs.

## lab-repo status

This lab is currently organized as a **standalone repository**, not an active configured fork of the contest repo.

Local git state at the time of this note:

- `.git` exists locally
- `git remote -v` returned no configured remotes

That means the lab can stay cleanly separated from the contest submission surface:

- this repo: research log, writeup, evidence, durable state
- contest repo PRs: public submission mechanism

## interpretation

- yes, public results have already been posted
- yes, some public PR-linked submissions clearly have evaluated results
- no, not every fork should be assumed to be an active or successful contestant branch
- keeping the lab as a standalone repo is cleaner and more explicit than trying to overload a fork with both lab state and submission state
