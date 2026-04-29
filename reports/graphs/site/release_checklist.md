# release checklist

## score lane (current — Era 2 / renderer)

- [x] Era 2 contest-CUDA floor recorded (Lane G v3 = `1.05` [contest-CUDA] / `1.04` [Modal-T4-CUDA])
- [x] raw scorer report retained for promoted floor (`experiments/results/lane_g_v3_landed/contest_auth_eval.json`)
- [x] Modal T4 reproduction independently verified within 0.01 noise (`experiments/results/modal_auth_eval_9b20bdfca246.json`)
- [x] canonical local E2E auth-eval smoke passes for the lane (`experiments/canonical_local_auth_eval_smoke.py --lane g_v3_corrected_kl_weight`)
- [x] strict-scorer rule respected (no PoseNet/SegNet weights at inflate time except gated waivers)
- [x] eval_roundtrip enforced everywhere (CLAUDE.md non-negotiable)
- [x] 78 STRICT preflight checks all pass on the submission archive
- [ ] Selfcomp-paradigm portfolio (MM / SA / SC++ / SO) lands at least one [contest-CUDA] score below 1.05
- [ ] 5-pass clean adversarial review on the submission archive (stricter than the standard 3-pass)
- [ ] proxy or scorer only a lane that actually writes a rankable artifact to disk

## score lane (historical — Era 1 / Track B post-filter)

- [x] promoted honest Track B floor recorded (1.73 / 1.7269 exact, long1000 h64 QAT+EMA)
- [x] raw scorer report retained for promoted floor (`reports/raw/2026-04-09-long1000-h64-authoritative/`)
- [x] current_workflow vs rule_faithful kept separate
- [x] canonical default-config regression confirms the live promoted floor (Era 1 only)
- [x] written promotion review added
- [x] authoritative weights co-located at `submissions/robust_current/postfilter_int8.pt`

## writeup lane

- [x] dashboard built
- [x] lineage / mechanism sections added
- [x] promotion accounting table built (Era 1 + Era 2)
- [x] experiment journal added
- [x] latest promotion review summary added
- [x] mathematical investigation section added to `final_writeup_draft.md`
      (Jacobian failure, SVD rank-1 finding, CNN residual characterization)
- [x] post-AV1 era documented (renderer paradigm, KL distill weight=0.002, pose TTO)
- [ ] Era 3 (Selfcomp paradigm) section drafted — pending [contest-CUDA] landings
- [ ] final polish and live-site consistency pass
- [ ] cross-reference `experiments/rd_bound_mine.py` MINE bound in the writeup once numbers stabilize

## strategic-secrecy gate (non-negotiable per CLAUDE.md)

- [ ] strategic-secrecy audit run on every public-facing markdown before May 3 push
- [ ] Cloudflare site URL NOT publicized until human explicitly says it is time
- [ ] Lane W (hard-pair self-compress) recipe NOT exposed on public surfaces
- [ ] Lane Ω (Hessian-aware quantization) recipe NOT exposed on public surfaces
- [ ] Lane DARTS-S (architecture search) recipe NOT exposed on public surfaces
- [ ] Selfcomp-paradigm portfolio sequencing NOT exposed at the level of "what we plan to ship"

## PR submission lane (live)

- [ ] target submission archive: pending Selfcomp-paradigm landings (or fall back to Lane G v3 1.05 if portfolio fails)
- [ ] dry-run `inflate.sh` against the rebuilt archive on a clean checkout
- [ ] confirm 30-min CPU budget on the actual contest hardware
- [ ] PR text updated to current operating point (replace 1.73 references with 1.05 if shipping renderer; with whatever lands [contest-CUDA] if shipping a Selfcomp-paradigm derivative)

## PR submission lane (historical — Era 1)

- [x] PR text at `experiments/pr_draft.md` reflects the promoted Era 1 floor and the
      mathematical investigation
- [x] `workspace/upstream/.../learned_postfilter_av1/archive.zip` rebuilt with
      authoritative 524x394 `0.mkv` + promoted int8 weights
- [x] `postfilter_int8.pt` co-located alongside inflate.sh as a redundant copy
- [x] `inflate_postfilter.py` accepts the `saliency_weighted` variant tag and
      per-channel int8 scale vectors
- [x] `compress.sh` updated to the 524x394 downscale that matches the scored archive
