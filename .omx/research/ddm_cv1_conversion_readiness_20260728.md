# ddm_cv1 — CONVERSION-READINESS (R6 exact-eval rehearsal + ship-path refresh)

**Date:** 2026-07-28 · **Arm:** ddm_cv1 (conversion-readiness) · **Budget:** $0 (no dispatch, no
submit, no upstream edits) · **Tags:** `[no-triality] [p0-ledger-ok]`

**Purpose:** when the fc1 composed archive lands at ≤~0.17 local-advisory, the campaign must CONVERT
within hours. This memo makes both conversion legs READY-TO-FIRE on paper and in $0-verifiable state.
Everything below was verified at $0 by reading source + running local structural checks; items I could
NOT verify at $0 are flagged **[UNVERIFIED-$0]** explicitly.

---

## Leg 1 — R6 EXACT-EVAL REHEARSAL

### (a) The canonical Modal exact-eval tool TODAY — and why

**Use `tools/dispatch_modal_paired_auth_eval.py` (`--execute` to fire).**

Rationale (verified by reading the sources):
- Its own docstring names it *"the canonical operator entry point so CPU/CUDA pairing is the default
  instead of an afterthought."* This is exactly the dual-axis requirement from CLAUDE.md "Submission
  auth eval — BOTH CPU AND CUDA."
- It builds and dispatches BOTH axes for one archive/runtime by delegating to the Modal wrapper
  `experiments/modal_auth_eval.py` (the entry point that owns the image + upload + `contest_auth_eval.py`
  invocation). The wrappers still own lane claims + artifact recovery; the paired tool orchestrates them.
- It is plan-only by default; `--execute` spawns both detached Modal jobs. It supports
  `--skip-axis-if-promotable-anchor-exists` (custody-validated anchor reuse) to avoid redundant re-fires.

**Do NOT** use `src/tac/deploy/witness_cloud_launcher.py` / `tools/launch_witness_cloud.py` for this —
that is the provider-neutral **TRAINING** launcher (Task #438, V9 CGauge CUDA training), not exact-eval.
The `#438` in the arm brief refers to the *provider-neutral launcher pattern*; the launcher instance
itself is a training surface. The exact-eval equivalent of that pattern is the paired auth-eval
dispatcher above, which routes through the provider-agnostic contracts in
`src/tac/deploy/provider_contracts.py` (`PROVIDER_CONTRACTS`, modal entry).

Supporting surfaces (all present, verified):
- `src/tac/deploy/modal/single_flight.py` — `assert_modal_single_flight(...)` pre-`.spawn()` guard (#513).
- `src/tac/deploy/modal/call_id_ledger.py` — `update_call_id_outcome(...)` writes terminal rows to BOTH
  ledgers in the same turn (dual-ledger terminality).
- `src/tac/deploy/modal/auth_eval.py` — upload payload + runtime manifest projection.
- `tools/harvest_modal_calls.py` — `.spawn()` result-cache harvester (HARVEST OR LOSE, ~24h TTL).
- `tools/plan_dual_device_auth_eval.py` — emits paired CPU/CUDA commands for the same archive/runtime.

### (b) Image / deps still build conceptually — uv dry-resolve PASS

Modal image (`experiments/modal_auth_eval.py:79-125`): `debian_slim(python_version="3.11")` +
apt(`ffmpeg git libglib2.0-0 libgl1 unzip xz-utils curl ca-certificates`) +
pip(`torch==2.5.1 torchvision safetensors einops segmentation-models-pytorch av click
nvidia-dali-cuda120==1.52.0 tqdm timm scipy numpy<2.0 Pillow pydantic>=2.0 brotli>=1.0
constriction>=0.4,<0.5 pyppmd>=1.3,<2.0 cryptography>=41.0`, extra_index `pypi.nvidia.com`) +
a pinned static-ffmpeg build + uv install.

**$0 dry-resolve (local, no remote build):**
`uv pip compile --python-platform linux --python-version 3.11` over the pip list (excluding the
NVIDIA-index-only `nvidia-dali-cuda120`) resolved cleanly — 53 pins, including `torch==2.5.1`,
`torchvision==0.20.1`, `constriction==0.4.2`, `pyppmd==1.3.1`. **No conflicts.**

Consistency vs current `pyproject.toml`: the image pins are a **superset** of and compatible with
pyproject (`torch>=2.0,<3.0`, `constriction>=0.4,<0.5`, `brotli>=1.0`, `pyppmd>=1.3,<2.0` [in the
`pr86_replay` extra], `cryptography>=41.0`, `timm>=0.9`, `segmentation-models-pytorch>=0.3`,
`safetensors`, `einops`, `scipy`, `numpy`, `pydantic`). The image is **self-contained** (explicit pins,
does not depend on pyproject resolution), so a pyproject drift cannot silently break it.
**[UNVERIFIED-$0]:** the actual remote Modal image *build* (apt + the static-ffmpeg `svtav1`/`in_primaries`
FATAL guards + NVIDIA DALI wheel pull) was NOT run — that requires a dispatch. The uv resolve is the
strongest $0 signal; the ffmpeg/DALI layers are the only untested build risk.

### (c) Single-flight + dual-ledger bookkeeping — 10-line runbook

```
1. tools/claim_lane_dispatch.py summary          # confirm NO active conflicting Modal claim (rc0)
2. .venv/bin/modal app list                      # confirm zero running Modal tasks (single-flight)
3. grep-tail .omx/state/modal_call_id_ledger.jsonl  # confirm no live (non-terminal) call_id rows
4. tools/claim_lane_dispatch.py claim --lane-id <lane> --platform modal \
       --instance-job-id <run_id> --agent ddm_cv1 --status eval --notes "R6 paired auth-eval"
5. DISPATCH (paired tool) — assert_modal_single_flight fires pre-.spawn() automatically
6. record BOTH spawned call_ids into modal_call_id_ledger.jsonl (wrapper writes spawn metadata)
7. HARVEST within 24h: tools/harvest_modal_calls.py  (result-cache TTL — HARVEST OR LOSE)
8. update_call_id_outcome(...) writes TERMINAL rows to BOTH ledgers same-turn (dual-ledger)
9. tools/claim_lane_dispatch.py claim --force --status completed_... (terminal claim, no phantom active)
10. tools/reconcile_dispatch_claims_with_modal_ledger.py  # verify claims ↔ ledger agree
```

### R6_CONVERSION_RUNBOOK (copy-pasteable; placeholders in `<...>`; GATES IN ORDER)

```bash
# ============ R6 PAIRED EXACT-EVAL (Modal contest-CPU + contest-CUDA) ============
# Preconditions: fc1 composed archive byte-closed at ≤~0.17 local-advisory; archive.zip + a COMPLETE
#   submission_dir (inflate.sh + inflate.py + README + runtime src tree) on disk.
cd /Users/adpena/Projects/pact
ARCHIVE=<path/to/fc1/submission_dir/archive.zip>
SUBDIR=<path/to/fc1/submission_dir>          # MUST contain inflate.sh + inflate.py + runtime tree
SHA=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
SIZE=$(stat -f%z "$ARCHIVE")                  # macOS stat; Linux: stat -c%s
LANE=lane_fc1_r6_paired_$(date -u +%Y%m%dT%H%M%SZ)

# GATE 0 — SINGLE-FLIGHT preclear (fail-closed on any live Modal work):
.venv/bin/python tools/claim_lane_dispatch.py summary
.venv/bin/modal app list
tail -3 .omx/state/modal_call_id_ledger.jsonl        # confirm last rows terminal

# GATE 1 — LANE CLAIM (atomic, refuses same-lane active within TTL):
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id "$LANE" --platform modal \
    --instance-job-id "$LANE" --agent ddm_cv1 --status eval \
    --notes "R6 fc1 paired auth-eval sha=$SHA"

# GATE 2 — PLAN (plan-only, prints both-axis command + custody expectations; NO spawn):
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive "$ARCHIVE" --submission-dir "$SUBDIR" \
    --expected-archive-sha256 "$SHA" --gpu T4 \
    --label fc1_r6 --lane-id-base "$LANE" \
    --json-out experiments/results/$LANE/paired_plan.json
#   REVIEW the plan JSON. Confirm CPU + CUDA axes, remote submission dir, expected shas.

# GATE 3 — EXECUTE (spawns BOTH detached Modal jobs; single-flight assert fires pre-.spawn):
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive "$ARCHIVE" --submission-dir "$SUBDIR" \
    --expected-archive-sha256 "$SHA" --gpu T4 \
    --label fc1_r6 --lane-id-base "$LANE" \
    --json-out experiments/results/$LANE/paired_run.json \
    --execute

# GATE 4 — HARVEST within 24h (result-cache TTL; do NOT skip):
.venv/bin/python tools/harvest_modal_calls.py
#   Artifacts per axis land under experiments/results/<lane>_modal/harvested_artifacts/:
#     contest_auth_eval.json  (canonical score record — THE authority)
#     report.txt              (Average PoseNet/SegNet Distortion + size + Final score)
#     contest_auth_eval.stdout.log / .stderr.log

# GATE 5 — TERMINAL CLAIM (no phantom active) + reconcile:
.venv/bin/python tools/claim_lane_dispatch.py claim --force --lane-id "$LANE" --platform modal \
    --instance-job-id "$LANE" --agent ddm_cv1 --status completed_ok --notes "harvested"
.venv/bin/python tools/reconcile_dispatch_claims_with_modal_ledger.py
```

### (d) Upload path + expected artifacts

- **Upload:** `experiments/modal_auth_eval.py` transports the `--submission-dir` (validated: no symlinks,
  no hidden/secret-looking files) and unzips under the remote `/tmp/modal_auth_eval/submission_dir`; it
  invokes `experiments/contest_auth_eval.py` there.
- **Expected artifacts (per axis):** `contest_auth_eval.json` (the canonical score record — recompute S
  from components; the rounded `final_score` field is advisory), `report.txt`, plus stdout/stderr logs.
  Harvested to `experiments/results/<lane>_modal/harvested_artifacts/`.
- **Score axis tags:** CUDA job → `[contest-CUDA]`; CPU job → `[contest-CPU]` (Modal CPU container is
  Linux x86_64, contest-CI-faithful per "Submission auth eval — BOTH CPU AND CUDA"). The CPU axis is the
  public-leaderboard ranking axis.

### (e) Cost estimate per n600 CPU + CUDA eval (from prior receipts)

Ledger receipts (`.omx/state/modal_call_id_ledger.jsonl`, elapsed_seconds; cost_usd not recorded so
costs are DERIVED from Modal published rates):

| Axis | Ledger elapsed anchor | Rate (Modal published) | Derived $ / eval |
|---|---|---|---|
| CUDA T4 | `lane_pr101_..._fec8_static T4` **84.7 s** | ~$0.000164/s ($0.59/hr) | ~$0.01–0.15 incl. cold-start |
| CPU (4-core) | `..._cpu` **256–315 s** (small archives) | ~$0.0000524/s (4×core) | ~$0.02 for small; **see caveat** |

**[UNVERIFIED-$0] cost caveat:** the ledger CPU elapsed rows (256–315 s) are for SMALL archives (~KB
codec packets). The fc1 / r6cal candidate is a **291 MB** archive; its inflate step is far heavier, and
CLAUDE.md notes a *full* n600 CPU eval on contest-CI-class hardware is **60–120 min**. Honest CPU band
for a heavy archive: **$0.19–$0.38** (4-core × 60–120 min). CUDA T4 with cold start + heavy inflate:
**$0.05–$0.50**. **Paired total realistic ≤ ~$1; hard envelope ≤$20 is comfortably safe.** Wall-clock:
CUDA ~2–5 min + cold start; CPU ~60–120 min. I could NOT measure the 291 MB inflate wall-clock at $0.

**Leg 1 verdict: READY** (dispatcher canonical + present, flags confirmed, image deps resolve, gates
ordered, artifacts + cost bounded). Residual risk: remote image build (ffmpeg/DALI layers) untested at $0.

---

## Leg 2 — SHIP PATH REFRESH (stale since 2026-05-19)

### (a) Compliance dry-run — CHECKLIST HEALTH (r6cal artifacts, contest-final)

**Command run ($0, structural only — does NOT run the scorer):**
```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir /Volumes/VertigoDataTier/pact/r6cal_byteclose_eval_20260727T230213Z/full \
  --archive .../archive.zip --auth-eval-json .../report.txt --contest-final --json-out <out>
```
The checker is **healthy**: it ran, parsed the archive, and emitted 39 structured pass/fail checks
(`passed=False` overall, as expected — r6cal S=194.43 is a CALIBRATION artifact, not a passing packet;
and its `full/` dir is NOT a real submission_dir — it has only `archive.zip` (symlink) + `report.txt`,
no `inflate.sh`/`inflate.py`/manifest).

**Result: 17 PASS / 22 FAIL of 39 checks.**

| Category | PASS | FAIL |
|---|---|---|
| Archive integrity (zip members, headers, payload readable, single container, no-dup) | 8 | 0 |
| Required files present | 2 (archive.zip, report.txt) | 1 (inflate.sh) |
| Auth-eval custody | 1 (exists) | 2 (json_object, contest_cpu_exists) |
| Submission runtime (inflate tree) | 0 | 1 |
| Archive manifest | 0 | 1 |
| Expected sha/size supplied (invocation args) | 0 | 2 |
| Report mentions sha/size | 0 | 2 |
| Post-deadline policy statement | 2 (not-template, not-negated) | 4 (present, names-mode, frontier-context, substantive) |
| Public source reproducibility (repo link, pinned rev, reproduce cmd) | 1 (no-placeholder) | 4 |
| Public evidence axis labels (CUDA/CPU) | 0 | 2 |
| Public template placeholders resolved | 1 | 0 |
| Contest-final (lane_id, job_id, selected-axis auth score) | 0 | 3 |
| Hosted-archive manifest (optional) | 1 | 0 |

**Interpretation (the important part):** EVERY failure is an **artifact-completeness or
invocation-argument gap**, not a script defect. They fall into three fixable buckets:
1. **Missing runtime tree** — r6cal `full/` lacks `inflate.sh`/`inflate.py`/`src`. Fixed by pointing the
   checker at a COMPLETE submission_dir (the fc1 packet will carry the full runtime; a complete
   reference exists at
   `experiments/results/pr101_frame_exploit_selector_fec8_static_second_order_k16_clean_20260526_codex/submission_dir`).
2. **Missing exact-eval custody JSON** — need the harvested `contest_auth_eval.json` (CPU + CUDA) from
   Leg 1, not just a `report.txt`. Pass via `--auth-eval-json` + `--contest-cpu-auth-eval-json`.
3. **Missing invocation args + release metadata** — `--expected-archive-sha256`, `--expected-archive-size-bytes`,
   `--expected-lane-id`, `--expected-job-id`, `--dispatch-claims-md`, hosted-archive manifest,
   public-source repo link + pinned revision + reproduce command, post-deadline policy statement,
   contest-CUDA/CPU axis labels in public text. All are supplied/authored at ship time.

The score-ceiling checks (`contest_final_selected_axis_auth_score_available`) fail here only because no
auth-eval JSON was supplied; they will evaluate the real score once Leg 1 lands. **The checklist is
structurally healthy TODAY** — no script rot, no missing check, no crash.

### SHIP_PATH_RUNBOOK (contest-CLOSED → PR against contest repo for official-table row)

**Contest status:** CLOSED (operator binding 2026-07-06). Shipping = open a PR against
`commaai/comma_video_compression_challenge` for the official-table row (merged-or-not per the 07-27 goal
re-anchor: bar = min(0.15, official-best ~0.172)). IP is open-source; no race/hold constraint. Operator
attribution ONLY — **NO Claude attribution on the public PR** (per "Public Disclosure Hygiene" +
`no-claude-attribution-public-pr`).

```bash
# ---- STEP 1: complete packet (fc1 submission_dir + harvested exact-eval JSON from Leg 1) ----
SUBDIR=<fc1 submission_dir with inflate.sh + inflate.py + README + runtime src>
ARCHIVE="$SUBDIR/archive.zip"
CUDA_JSON=<harvested contest_auth_eval.json [contest-CUDA]>
CPU_JSON=<harvested contest_auth_eval.json [contest-CPU]>   # public-leaderboard ranking axis
SHA=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}'); SIZE=$(stat -f%z "$ARCHIVE")

# ---- STEP 2: CONTEST-FINAL COMPLIANCE GATE (strict; must pass before any upload) ----
.venv/bin/python scripts/pre_submission_compliance_check.py \
    --submission-dir "$SUBDIR" --archive "$ARCHIVE" \
    --auth-eval-json "$CUDA_JSON" --contest-cpu-auth-eval-json "$CPU_JSON" \
    --expected-archive-sha256 "$SHA" --expected-archive-size-bytes "$SIZE" \
    --submission-score-axis contest_cpu --max-submission-score 0.172 \
    --expected-lane-id "$LANE" --expected-job-id "$LANE" \
    --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
    --require-auth-eval --require-t4-equivalent --require-submission-runtime-match \
    --require-competitive-or-innovative-statement \
    --contest-final --strict
#   MUST exit 0. Any FAIL is a blocker — resolve, do NOT --force.

# ---- STEP 3: HOST archive.zip (get a public download URL) ----
#   Options (operator picks): Cloudflare R2/Pages · Lightning public artifact · GitHub Release asset
#   on adpena/comma_video_compression_challenge fork. Record URL + sha256 + bytes in a hosted-archive
#   manifest and re-run STEP 2 with --hosted-archive-manifest-json.

# ---- STEP 4: PR body from docs/submission_template.md (Apogee template) ----
#   Replace ${APOGEE_ARCHIVE_ZIP_URL} with the hosted URL; paste the FINAL exact report.txt score block;
#   answer "requires GPU for inflation?"; include competitive-or-innovative statement + public-source
#   attribution. OPERATOR ATTRIBUTION ONLY — NO Claude/Anthropic attribution anywhere in the public PR.

# ---- STEP 5: open the PR (operator action) ----
gh pr create --repo commaai/comma_video_compression_challenge \
    --title "<submission name>" --body-file <pr_body.md>
```

**Ship blockers (named, TODO at conversion time):** (1) a passing fc1 exact row (Leg 1); (2) a hosted
`archive.zip` URL (we hold local bytes only); (3) both `contest_auth_eval.json` axes; (4) STEP-2 strict
compliance PASS; (5) competitive-or-innovative statement authored; (6) public-source attribution block.
Internal `tools/create_fork_pr_for_submission.py` is for self-eval GHA-CPU **fork** PRs to
`adpena/...`, NOT for the contest PR.

**Leg 2 verdict: READY (checklist healthy) / BLOCKED-on-artifacts (score + hosted URL + auth JSON).**

---

## MIDDLE-BAND PRE-REGISTRATION (pre-committed, no re-litigation)

If the fc1 composed row lands in **(0.17, 0.35]**, it is the **CALIBRATION row**. The pre-registered,
pre-committed response is:

> **ONE re-waterfill pass on MEASURED per-stream prices (two-pass encoding)** — take the measured
> per-stream byte costs from the first composed encode, feed them back as the KKT-waterfill prices, and
> re-encode once (second pass). This is the standard "prices are upper-bounds from a suboptimal
> proposal-search channel" correction (memory `distortion_byte_economics_are_upper_bounds_...`): the
> first pass measures the true marginal prices; the second pass allocates against them.

**No strategic re-litigation is opened by a single (0.17, 0.35] row.** Only a **second** composed row
that is **still > 0.35** after the re-waterfill pass opens a design fork. A row that lands ≤0.17 goes
straight to Leg 1 (R6 exact-eval); a row in (0.17, 0.35] gets exactly one re-waterfill pass then
re-evaluates against this same ladder.

---

## Honesty ledger (what I could NOT verify at $0)

- **Remote Modal image build** — NOT run (would require dispatch). uv Linux/py3.11 dep resolve PASSES;
  the ffmpeg-static (`svtav1`/`in_primaries` FATAL guards) + `nvidia-dali-cuda120` wheel layers are the
  only untested build risk.
- **291 MB archive inflate wall-clock + true CPU eval cost** — the ledger CPU receipts (256–315 s) are
  for small archives; the heavy-archive band ($0.19–0.38 CPU, 60–120 min) is DERIVED from CLAUDE.md +
  Modal rates, not measured. Still comfortably inside the ≤$20 envelope.
- **fc1 archive itself does not exist yet** — this rehearsal targets the *pending* composed archive;
  runbook placeholders (`<...>`) are filled at conversion time.
- All score/promotion authority remains the harvested `contest_auth_eval.json` recomputed-from-components
  (never the rounded `final_score`, never a proxy, never MPS).
