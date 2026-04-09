# release checklist

## score lane

- [x] promoted honest Track B floor recorded (1.73 / 1.7269 exact, long1000 h64 QAT+EMA)
- [x] raw scorer report retained for promoted floor (`reports/raw/2026-04-09-long1000-h64-authoritative/`)
- [x] current_workflow vs rule_faithful kept separate
- [x] canonical default-config regression confirms the live promoted floor
- [x] written promotion review added
- [x] authoritative weights co-located at `submissions/robust_current/postfilter_int8.pt`
- [ ] run another scorer candidate only if it is a disciplined post-filter follow-on with smoke and scorer gates
- [ ] keep the bat00 parity rerun alive until it emits a materially stronger saved best than the current epoch `199` / `3.9258`
- [ ] proxy or scorer only a lane that actually writes a rankable artifact to disk

## writeup lane

- [x] dashboard built
- [x] lineage / mechanism sections added
- [x] promotion accounting table built
- [x] experiment journal added
- [x] latest promotion review summary added
- [x] mathematical investigation section added to `final_writeup_draft.md`
      (Jacobian failure, SVD rank-1 finding, CNN residual characterization)
- [ ] final polish and live-site consistency pass
- [ ] cross-reference `experiments/rd_bound_mine.py` MINE bound in the writeup once numbers stabilize

## PR submission lane (learned_postfilter_av1)

- [x] PR text at `experiments/pr_draft.md` reflects the promoted floor and the
      mathematical investigation
- [x] `workspace/upstream/.../learned_postfilter_av1/archive.zip` rebuilt with
      authoritative 524x394 `0.mkv` + promoted int8 weights
- [x] `postfilter_int8.pt` co-located alongside inflate.sh as a redundant copy
- [x] `inflate_postfilter.py` accepts the `saliency_weighted` variant tag and
      per-channel int8 scale vectors
- [x] `compress.sh` updated to the 524x394 downscale that matches the scored
      archive
- [ ] dry-run `inflate.sh` against the rebuilt archive on a clean checkout to
      confirm the self-contained path still works before opening the PR
