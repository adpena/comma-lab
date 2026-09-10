# PR #140 move-43 swap commands — NOT RUN

Status: BLOCKED. These are operator commands, not an authorization or a runnable release approval. No command below was executed by ddm_swp2. One-line confirmation is required by `p0_swap_procedure_no_push_without_confirm_20260817`; confirmation does not waive the eight strict failures or the stale README/MANIFEST. MAIN must resolve them, regenerate/review the final packet, and retain a fresh strict PASS first. The archived T4 receipt is not a release-policy clearance. This packet cannot honestly supply a release-ready one-command confirm while strict compliance is refused. The blocks below are exact conditional operator commands, all NOT RUN.

The immutable staging copy is the review reference. A release-ready successor must refresh README/MANIFEST, resolve runtime import and CPU/promotion policy, and keep its own fresh manifests. Move43 staging already clears archive-manifest, dispatch and portable runtime-tree checks. The eight exact failures are in BLOCKERS.json; nothing is waived. Do not modify sealed source custody. A changed runtime needs the prescribed review and authority proof; ddm_swp2 does not authorize any scorer or dispatch.

## Hosting phase — NOT RUN; only after explicit operator authorization

After MAIN clears every non-hosting blocker and the operator approves the concrete reviewed release, create a NEW release tag (never overwrite AFR1). The intended archive is pinned by STAGING_MANIFEST.json. These commands are contingent on the final successor retaining that archive identity.

```bash
set -euo pipefail
cd /Users/adpena/Projects/pact
# STOP unless operator approval and non-hosting clearance have been recorded by MAIN.
gh release create semantic_joint_ctxmix-move43 \
  submissions/_staging_move43_pr140_swap/archive.zip \
  --repo adpena/comma_video_compression_challenge \
  --target semantic_joint_ctxmix \
  --title 'semantic_joint_ctxmix move 43' \
  --notes 'Retained T4 evaluation; verify archive identity in the submission report.'
mkdir -p /Volumes/VertigoDataTier/pact/ddm_swp2_publish/retained
curl --fail --location --output /Volumes/VertigoDataTier/pact/ddm_swp2_publish/retained/archive.zip \
  'https://github.com/adpena/comma_video_compression_challenge/releases/download/semantic_joint_ctxmix-move43/archive.zip'
.venv/bin/python - <<'VERIFY'
from pathlib import Path
import hashlib,json
r=json.loads(Path('.omx/research/ddm_swp2_20260910/SCORE_RECOMPUTATION.json').read_text())
p=Path('/Volumes/VertigoDataTier/pact/ddm_swp2_publish/retained/archive.zip')
assert p.stat().st_size==r['archive_bytes']
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['archive_sha256']
VERIFY
```

MAIN must emit a real hosted-archive manifest from that retained download, run the strict checker with `--hosted-archive-manifest-json`, remove the draft release notice from the approved PR body, and repeat public hygiene against the final body. Keep the download and its provenance. No pre-existing AFR1 URL may be paired with the move-43 hash.

## Local copy and serializer, then PR update — NOT RUN

Prerequisites: the operator approves the final concrete tree/body; all strict checks pass on that exact tree; MAIN completes two real review-tracker passes for any changed Python; all copied text is public-safe. Use a dedicated PR checkout, never switch the shared dirty main worktree. The paths `APPROVED_PACKET` and `APPROVED_BODY` must name the reviewed successor; they deliberately have no default while the current staging tree is refused.

```bash
set -euo pipefail
: "${APPROVED_PACKET:?MAIN must supply the reviewed release-ready successor path}"
: "${APPROVED_BODY:?MAIN must supply the approved final PR body path}"
: "${APPROVED_COMPLIANCE:?MAIN must supply the fresh strict PASS JSON}"
export APPROVED_PACKET APPROVED_BODY APPROVED_COMPLIANCE
cd /Users/adpena/Projects/pact
.venv/bin/python - <<'GATE'
import json,os
from pathlib import Path
p=json.loads(Path(os.environ['APPROVED_COMPLIANCE']).read_text())
assert p['passed'] is True and all(x['passed'] for x in p['checks']), 'STRICT REFUSAL: no swap'
GATE
git clone --single-branch --branch semantic_joint_ctxmix \
  https://github.com/adpena/comma_video_compression_challenge.git \
  /Volumes/VertigoDataTier/pact/ddm_swp2_publish/pr_checkout
.venv/bin/python - <<'SWAP'
import hashlib,json,os,shutil,subprocess
from pathlib import Path
root=Path('/Users/adpena/Projects/pact')
repo=Path('/Volumes/VertigoDataTier/pact/ddm_swp2_publish/pr_checkout')
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
assert subprocess.check_output(['git','-C',str(repo),'branch','--show-current']).decode().strip()=='semantic_joint_ctxmix'
stage=Path(os.environ['APPROVED_PACKET']).resolve()
row=json.loads((root/'.omx/research/ddm_swp2_20260910/SCORE_RECOMPUTATION.json').read_text())
assert hashlib.sha256((stage/'archive.zip').read_bytes()).hexdigest()==row['archive_sha256']
assert (stage/'archive.zip').stat().st_size==row['archive_bytes']
live=repo/'submissions/semantic_joint_ctxmix'
backup=repo.parent/'retained/live_before_swap'
assert live.is_dir() and not backup.exists()
# Preserve the previous tree and every file hash before the replacement.
manifest=[{'path':str(p.relative_to(live)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(live.rglob('*')) if p.is_file()]
backup.parent.mkdir(parents=True,exist_ok=True)
(backup.parent/'LIVE_BEFORE_SWAP.json').write_text(json.dumps({'source':str(live),'destination':str(backup),'files':manifest,'reason':'lossless pre-swap custody'},indent=2)+'\n')
shutil.move(str(live),str(backup))
shutil.copytree(stage,live)
# The public repo tracks source; the archive remains a separately hosted asset.
tracked=subprocess.check_output(['git','-C',str(repo),'ls-files','submissions/semantic_joint_ctxmix']).decode().splitlines()
files=sorted(set(tracked)|{str(p.relative_to(repo)) for p in live.rglob('*') if p.is_file() and p.name!='archive.zip' and '__pycache__' not in p.parts})
cmd=[str(root/'.venv/bin/python'),str(root/'tools/subagent_commit_serializer.py'),'--repo-root',str(repo),'--label','ddm_swp2_publish','--no-co-author','--message',f"semantic_joint_ctxmix: stage move 43 S {row['score']} [no-triality] [p0-ledger-ok]",'--files',*files]
for name in files:
 p=repo/name
 if p.is_file():cmd+=['--expected-content-sha256',name+'='+hashlib.sha256(p.read_bytes()).hexdigest()]
env=dict(os.environ);env.pop('REVIEW_GATE_OVERRIDE',None)
subprocess.run(cmd,env=env,check=True)
SWAP
git -C /Volumes/VertigoDataTier/pact/ddm_swp2_publish/pr_checkout push origin HEAD:semantic_joint_ctxmix
gh pr edit 140 --repo commaai/comma_video_compression_challenge \
  --title 'semantic_joint_ctxmix (0.137)' --body-file "$APPROVED_BODY"
```

MAIN must verify the published PR/head and download afterward and record the publication receipt. No command is permitted to treat serializer fallback rc=17 as a successful branch landing. If the pointer changes first, re-stage the newly authorized submittable row; do not silently update the fixed move-43 packet.

## Current packet verification — NOT RUN by the swap procedure

The identical checker invocation was run during preparation and returned rc 1 (85/93 PASS). It must not be treated as release approval.

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py --submission-dir submissions/_staging_move43_pr140_swap --archive submissions/_staging_move43_pr140_swap/archive.zip --auth-eval-json .omx/research/ddm_swp2_20260910/contest_auth_eval.json --contest-final --strict --submission-score-axis contest_cuda --expected-archive-sha256 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e --expected-archive-size-bytes 180466 --expected-runtime-tree-sha256 a726739a52452f8824c4eb9f98322f927b8b5d2d4c6ac1717911f61b841a9803 --expected-lane-id ddm_sj1_t4_token_predistortion_pass6_20260910 --expected-job-id ddm_sj1_pass6_t4_20260910 --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md --competitive-or-innovative-statement-file .omx/research/ddm_swp2_20260910/COMPETITIVE_STATEMENT.md --public-scan-path .omx/research/ddm_swp2_20260910/PR_BODY.md --json-out .omx/research/ddm_swp2_20260910/COMPLIANCE.json --contest-cpu-auth-eval-json .omx/research/ddm_rp1_round2_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json --archive-manifest-json .omx/research/ddm_swp2_20260910/archive_manifest.json
```
