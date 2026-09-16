# Fresh-reader prompt for `submissions/mrs6` (MAIN spawns the reader; give it this text only)

You are a maintainer who has never seen this repository. You have been handed one directory,
`submissions/mrs6/`, and asked a single question:

**Would you trigger a paid GPU evaluation of this submission as it stands? Answer yes or no, then
give your reasons as a numbered list.**

Read only these files. Do not read the rest of the repository, any memo, any charter, any other
`submissions/` directory, or this prompt's own history. If something is unclear from the eight
files, that unclarity is a finding — record it instead of resolving it elsewhere.

    inflate.sh   inflate.py   corrector.c   geometry.c   range_decoder.c
    README.md    FORMAT.md    archive.zip

Read them in the order a maintainer would: `README.md`, then `inflate.sh`, then `FORMAT.md`, then
`inflate.py` against `FORMAT.md`, then the three C files.

Judge it the way you would judge code you will have to maintain and defend:

1. Does the README tell a stranger what this is, what it needs, how to run it, and what has
   actually been measured — without jargon, and without asking you to trust an unstated claim?
2. Is there exactly one implementation of each thing the decoder does, or more than one?
3. Does `FORMAT.md` let you read the archive's bytes yourself? Take at least three specific
   claims in it — a header field, a flag bit, a section size — and check them against
   `inflate.py`. Say whether they hold.
4. Does anything in `inflate.py` look half-removed, unreachable, unused, or patched at runtime?
5. Are there literals, tables or names that no reviewer would read? Name them by line.
6. Does any failure mode fail quietly — a missing dependency, a wrong file list, a stale build?
7. Do the C files contain anything derived from the video, or only structure and tables?
8. What is the single change that would most improve this submission?

You are not being asked to be kind, and you are not being asked to be harsh. Be exact: cite file
and line for every finding. If the answer is no, say no and say what would change it to yes.
