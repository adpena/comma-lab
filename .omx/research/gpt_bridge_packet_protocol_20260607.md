# GPT bridge-packet protocol (operator-relayed collaboration)

UTC: 2026-06-07 · Source: GPT partner message via operator; adopted.

## Channel

Claude writes compact artifacts to `~/Downloads/pact_bridge/` and zips to
`~/Downloads/pact_bridge_packet_<utc>.zip`; the operator drags the zip into
the GPT conversation. GPT reviews and returns typed amendments (VERDICT /
CRITICAL RISK / PATCH TARGET / TEST / NEXT RUN / DO NOT). Claude→GPT packets
use CONTEXT / CLAIM / EVIDENCE / ASK / EXIT CRITERIA / ARTIFACTS.

## Packet layout (standardized)

```
pact_bridge/
  00_MANIFEST.json            # name/bytes/sha256 per file + created_utc
  01_STATUS.txt
  02_AUDIT_DECISION_MATRIX.jsonl
  03_V6_RAW_ACTIONEFFECT_ROWS.jsonl   # one JSON object per arm/action row
  04_V6_DECISION_MATRIX.txt
  05_RECEIPTS/
  06_PATCH_SUMMARIES/
  07_TEST_OUTPUTS/
```

Raw rows before polished memos; 03 is the highest-value file.

## Rules

- No untyped score; authority fields live ONLY on canonical repo artifacts —
  the bridge packet cites artifact ids/paths/sha256s and creates NO new score
  authority.
- Tainted (audit-flagged, unremediated) evidence is excluded from packets.
- Disclosure hygiene (CLAUDE.md): packets contain research memos + derived
  JSONL/test outputs only — never `.omx/state` ledgers, provider transcripts,
  credentials, or private infra paths. Dragging into ChatGPT is
  operator-initiated external disclosure of exactly those contents.
- File limits at the GPT side: 512 MB/file, ~2M tokens per text file — keep
  files focused; split rather than truncate.
- Same action_id continuity and dual-epsilon rules apply inside packet rows
  as in the repo.

## Standing state at adoption

Audit `wf_81fb975b-c30` pending → first packet ships with the audit decision
matrix + (if clean or post-remediation) raw v6 four-arm ActionEffect rows per
spec `252cf261c`. No further spec changes until verdict or rows exist.
