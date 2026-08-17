# Submission mechanics runbook — release + PR command sequence (2026-08-17, deadline: tonight)

**Owner: MAIN. Consumer: sr1 pass 4 (mechanics dry-run) + the operator's final confirm.**
Status: PREPARED, NOT EXECUTED. Nothing below runs until the operator's final one-line
confirm with exact candidate + score + URL. Publishing is outward-facing and irreversible.

## Candidate binding (freeze ~01:00Z)

- Default candidate: **rr4** — archive 181,161 B, sha
  `35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956`,
  S 0.15853325034789678 [contest-CUDA T4, n600]. Staged tree:
  `/Volumes/APDataStore/pact/ddm_pq2/submission_staging/`.
- Hot-swap: fx1 (180,601 B, sha `65c75d7f097df930…`, projected S 0.158160) ONLY if its
  receiver-closed chain + T4 row land before freeze. Swap runs SWAP_PROCEDURE.md steps 1–5
  (pq1 bundle) — every sha below re-derives from the new receipts; nothing transfers by hand.

## Preconditions (all verified before step 1 fires)

1. `gh auth status` → account `adpena`, active. (VERIFIED 2026-08-17.)
2. Fork `adpena/comma_video_compression_challenge` exists. (VERIFIED.)
3. sr1 gauntlet GREEN (counter target met per its charter) OR operator explicitly waives.
4. Staged tree sanitized: **strip AppleDouble/`.DS_Store` at copy time** — the ExFAT staging
   volume litters `._*` files (a live `._README.md` sits in the staged root right now).
   Every copy below uses the rsync exclude form; never bare `cp -r` from APDataStore.
5. PR body final: the `Download status: pending` line in PR_BODY_DRAFT.md replaced with the
   real release-asset URL (step 3 output) before `gh pr create`.
6. sr1 F1: BORROWED_SUBSTRATE_ACCOUNTING.md refreshed to the FROZEN candidate (section-level
   re-accounting per SWAP_PROCEDURE step 4) and staged so README.md:69's link resolves — the
   packet's strongest honesty document must not ship as a dangling reference.
7. sr1 F15 (next-candidate law): the receiver must guard its C-compiler dependency fail-closed
   (command -v cc probe + declared dep) FROM BIRTH — retrofitting changes the runtime-tree
   hash and forces a refire. fx1 instructed; verify on whichever candidate freezes.

## Command sequence (executed by MAIN only, after operator confirm)

Working root: a DETACHED clone under the scratchpad — never the shared pact worktree.

```bash
# 0. fresh detached clone of the fork, synced to upstream main
git clone https://github.com/adpena/comma_video_compression_challenge \
    "$SCRATCH/subm_clone" && cd "$SCRATCH/subm_clone"
git remote add upstream https://github.com/commaai/comma_video_compression_challenge
git fetch upstream && git checkout -B submission/rr4-free-corrector upstream/master

# 1. copy the staged packet into submissions/<name>/ (AppleDouble-safe)
mkdir -p submissions/rr4_free_corrector
rsync -a --exclude='._*' --exclude='.DS_Store' \
    /Volumes/APDataStore/pact/ddm_pq2/submission_staging/ \
    submissions/rr4_free_corrector/
# archive.zip is HOSTED, not committed — remove it from the tree copy:
rm submissions/rr4_free_corrector/archive.zip
# verify: no AppleDouble residue, no private paths
find submissions -name '._*' -o -name '.DS_Store' | wc -l   # must be 0
# normalize ExFAT 0700 modes (sr1 F3): docs/data 644, executables 755
find submissions/rr4_free_corrector -type f -exec chmod 644 {} +
chmod 755 submissions/rr4_free_corrector/inflate.sh

# 2. commit + push the branch to the fork
git add submissions/rr4_free_corrector
git commit -m "rr4 free-corrector re-encode submission"
git push -u origin submission/rr4-free-corrector

# 3. host archive.zip as a release asset on the FORK (public URL)
gh release create rr4-submission-20260817 \
    --repo adpena/comma_video_compression_challenge \
    --title "rr4 free-corrector re-encode archive" \
    --notes "archive.zip 181161 bytes sha256 35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956" \
    /Volumes/APDataStore/pact/ddm_pq2/submission_staging/archive.zip
# asset URL: https://github.com/adpena/comma_video_compression_challenge/releases/download/rr4-submission-20260817/archive.zip

# 4. finalize PR body: insert the asset URL + exact identity block, then open the PR
gh pr create \
    --repo commaai/comma_video_compression_challenge \
    --base master \
    --head adpena:submission/rr4-free-corrector \
    --title "rr4 free-corrector re-encode (0.15853 GPU-eval, 181,161 B)" \
    --body-file PR_BODY_FINAL.md

# 5. post-publish verification (immediately)
curl -sL <asset-url> | shasum -a 256      # must print 35ac2b9beb…
gh pr view --repo commaai/comma_video_compression_challenge <num> --json url,title
```

Flag audit (sr1 pass-4 duty): `gh release create <tag> [files...] --repo --title --notes`
and `gh pr create --repo --head owner:branch --title --body-file` verified against
`gh release create --help` / `gh pr create --help` on this host before fire (never-invent-flags).
Note the upstream default branch must be confirmed at dry-run time (`master` vs `main` —
step 0 pins whatever `gh repo view commaai/... --json defaultBranchRef` returns).

## What the operator's final confirm line contains

candidate id + archive sha (first 8) + S + the release tag name. One line, then MAIN fires
steps 0–5 in order, stopping on any mismatch.

## Refusal conditions (inherit SWAP_PROCEDURE + pq1 runbook)

- staged-tree sha mismatch vs RESULT_build.json / the recovered contest_auth_eval.json at
  copy time (NOT GENERATION_RECEIPT.json — sr1 measured that receipt STALE: it declares the
  superseded 182,759 B / 80d9c8c6… identity and would fire on every correct packet)
- any `._*`/private-path residue after step 1
- sr1 RED verdict without operator waiver
- report.txt score fields not byte-identical to the recovered contest_auth_eval.json
