# ddm_sw2 — COMPREHENSIVE secrets sweep VERDICT: repo history is credential-clean; the 16 hits are third-party/expired; blocking commit-time guard landed

`date_utc: 2026-08-20` · `owner: MAIN` · `score_claim: false` · cost $0
Supersedes-in-part: the #1168 P0 framing (rotation urgency DOWNGRADED on measured evidence).

## THE ANSWER, FIRST

Operator asked for *"a comprehensive sweep to ensure no secrets are exposed or keys or
credentials or anything."* Verdict across four independent legs, full 15,340-commit public
history + working tree:

**No credential of OURS has ever been committed.** The only credential-shaped material in
history is the known 16-hit cluster in the deleted `.playwright-mcp/` console logs
(commit `a3237dd6b7`, 2026-04-10), and adjudication shows it is **third-party and expired**:

1. **The 3 JWTs** carry `aud=gpu-t4-s-kkb-usw1b1-183ci7vthn60x` — a Colab GPU-runtime VM
   identifier — with expiries ~55 min apart on 2026-04-10. Colab short-TTL session tokens,
   dead for 4+ months, bound to an ephemeral VM that no longer exists. Zero live risk.
2. **The 4 AIza keys** appear in console logs whose traffic is `colab.clients6.google.com` /
   `alkalimakersuite-pa.clients6.google.com` / gstatic — Google's OWN first-party public
   browser API keys, embedded in Google's client JS on every Colab page and visible to every
   visitor's console. Not keys from the operator's GCP account.
   **Residual operator action (60 s):** check the GCP console credentials page for any key
   with prefix `AIzaSyA2Bv` / `AIzaSyAmDc` / `AIzaSyB10s` / `AIzaSyCN_s`. Expected: none →
   nothing to rotate at all. (#1168 downgraded accordingly.)

## WHY PLAYWRIGHT WAS IN THE REPO (operator question)

An April 10 session used the Playwright-MCP browser extension to monitor a Google Colab
training run (`colab-epoch-check.png`, `colab-progress.png`). The extension dumps session
artifacts (console logs, page snapshots, screenshots) into `.playwright-mcp/`; a bare
`git add` in that 236-file commit swept the directory in. Later deleted; the directory is
now gitignored; history retained the blobs.

## SELECTIVE CLEANING — impossible by construction (operator question)

A git commit sha commits to its full tree AND its parent sha, so removing a file from an
April 2026 commit changes every descendant sha — measured blast radius **15,442 of 15,678
commits (98.5%)**, destroying the campaign's receipt/pin web. There is no selective
operation. Options were: full rewrite + force-push + GitHub Support cache purge (operator
DECLINED: "Don't do it I guess" — correct, and consistent with rotation-not-rewrite), or
neutralize the values — which the adjudication above shows are already inert
(expired/third-party). History clean is CLOSED-DECLINED; no further action.

## THE FOUR SWEEP LEGS (all executed, receipts on APDataStore)

| leg | scope | result |
|---|---|---|
| sensitive-named paths | all history, `git log --all --name-only` | CLEAN — no .env/pem/id_rsa/api-key/kaggle.json/fleet.local.toml ever tracked |
| ignore coverage | live tree | fleet.local.toml + .env ignored; vast_api_key outside repo |
| default+campaign rules, full history | 15,340 commits / 2.14 GB, gitleaks w/ `configs/gitleaks_pact.toml` (extends default + modal/hf/anthropic/tailscale/wandb shapes) | **0 alarm-class rows** (no Modal/HF/Anthropic/Tailscale/wandb/GitHub/AWS/OpenAI/Slack/private-key material, ever); 16 = the known playwright cluster; 38,869 `generic-api-key` rows adjudicated FALSE POSITIVES — internal ledger fields (`contract_key` 15,640 · `key` 20,675 · `evidence_key` 1,747 · `*_sha256`), content hashes not credentials |
| working tree | current HEAD (sw1 P4 + this pass) | CLEAN |

Receipts: `/Volumes/APDataStore/pact/ddm_sw1_20260820/receipts/gitleaks_custom_history.json`
(+ `.log`), redacted mode — no secret values reproduced anywhere, including this memo.

## THE NEVER-AGAIN GUARD (two-landing cure for "none of that should have been committed")

Landed `db153b5073`:
- **`tools/preflight_hook.py` step 1i `run_staged_secrets_scan`** — BLOCKING
  `gitleaks protect --staged` over exactly the bytes about to be committed, placed before
  `run_preflight()` (early-return-proof, same rationale as 1b–1h). Findings → rc=1 REFUSE
  with redacted output; binary-absent/timeout/tool-error → LOUD warn with the unscanned
  denominator (never skip-as-green, #875/#1050); waiver `TAC_SECRETS_WAIVE=1` prints as a
  deliberate exception.
- **`configs/gitleaks_pact.toml`** — canonical ruleset: default rules + the campaign's own
  token shapes (modal `(ak|as|tok)-…`, `hf_…`, `sk-ant-…`, `tskey-…`, wandb-context).
- **Controls EXECUTED both directions** (#1086 lesson — a control must be able to fail, and
  the first attempt DID honestly fail: probe path was gitignored, never staged; harness
  fixed, re-run): synthetic `hf_`-shaped token staged → FIRED rc=1; clean index → rc=0.
  Review pass 1 found + fixed a real defect (`.split()` denominator miscount on
  space-bearing filenames → `.splitlines()`).

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600] — unmoved this unit (sixteenth
move stands), archive df7fd266…, call fc-01M0G7QCQPACVJV29D7AAQSXAA.**
