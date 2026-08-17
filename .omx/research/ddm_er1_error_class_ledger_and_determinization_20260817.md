# ddm_er1 — session error-class ledger: fix + structural self-protection per row (2026-08-17)

Operator bindings, verbatim: "Must fix and self protect against all errors" + "Every mistakes due
to doing things by hand should be fixed and deterministic and no longer ad hoc or manual."
Per the CLAUDE.md two-landing law, every row below carries BOTH a fix and a structural protection.
The unifying cure landed this session: **tools/fire_modal_auth_eval.py** (commit 65e15db4e9) — the
ONE deterministic path from candidate runtime to T4 row. Hand-assembled Modal dispatches are now a
forbidden pattern; the tool subsumes error rows E1–E5 below.

| # | error (all 2026-08-17) | root cause | fix (landed) | structural protection | residual |
|---|---|---|---|---|---|
| E1 | rr2 fired a hand-assembled `staged_*` tree missing the receiver's decoder guard → S 27.83, fired bytes ≠ proved bytes | by-hand runtime staging | rr4 fired `candidate_runtime` unmodified; row in flight | fire tool takes the runtime dir ITSELF; shas computed from actual bytes; `--require-archive-sha` refuses pin mismatch (rc=4, dry-run-proven) | proved-tree↔fired-tree RECEIPT binding (parse-back receipt carries proved tree sha; tool cross-checks) — owned MAIN, next apparatus window |
| E2 | AppleDouble/.DS_Store litter (ExFAT) tripped the remote hidden-file validator — one paid round wasted | macOS metadata on ExFAT + no local preflight | stripped by hand once | fire tool stage SANITIZE deletes litter deterministically every fire | none |
| E3 | `retained/` payload custody inside the tree tripped the secret-name validator remotely | custody dir colocated with runtime | relocated to sibling (payload law: moved, never deleted) | fire tool stage VALIDATE runs the SAME validators locally in one $0 pass and prints the exact relocation command | none |
| E4 | `--expected-runtime-tree-sha256` omitted by hand → dispatcher FATAL | hand-typed flag set | refired with `auto` | fire tool pins `auto` unconditionally in the fixed template | none |
| E5 | rr2's phantom-active Modal claim blocked single-flight; terminal-close args discovered by argparse error | claim closure was manual; harvest never closed claims | terminal row appended by hand once | fire tool stage CLAIMS auto-closes the exact provable-phantom condition (active claim + 0 live ledger call_ids, the reconcile tool's own PROBLEM predicate), automation recorded in notes | claim closure AT HARVEST (wire into modal_harvest_poller/endpoint closer) — owned MAIN |
| E6 | harvest poller armed with INVENTED flags (`--results-dir/--interval`) → died instantly, paid call unwatched — never-invent-flags violated by MAIN | hand-armed watcher | re-armed detached with real argparse (`--call-id/--output-dir`), pid 24100, done-receipt | fire tool stage ARM-POLLER reads call_id from the spawn record and arms the poller itself with the correct flags; no hand-arming remains on the canonical path | none |
| E7 | BG-bash until-loop waiter SIGURG-144 killed (~3 min class, KNOWN law m77, violated anyway) | hand-armed BG bash despite the law | replaced with detached launch_detached_process waiter | canonical path never uses BG bash: fire tool arms detached; hg1-style run waiters go through launch_detached_process + FIFO monitor | none |
| E8 | rr2 mechanism MISATTRIBUTED ("CPU-prob vs CUDA-prob") — the falsifying evidence sat unread in the two MODAL_REMOTE_RESULT.json receipts | mechanism written before reading the receipts | rr4 falsified it at source ($0); replacement law in rr4 memo §8 (correctly-rounded-arithmetic portability) | DESYNC falsifier now demands per-frame `decoder_bit_position` + coding-row digest telemetry so the divergent frame is NAMED, never inferred; read-receipts-before-mechanism encoded in memory | telemetry emit lands with the next receiver-touching change — owned MAIN |
| E9 | hg1 arm_a silent death (watchers:[], 17h unnoticed) — make-silence-loud class | launch without receipt watcher | resumed watched; arm_b launched with done-receipt from birth | FIFO monitor + done-receipts on every detached launch; liveness watcher (#1064) covers child-death | none |
| E10 | both landed rr2 instruments SyntaxError at HEAD — committed artifacts came from an uncommitted working copy | commit path runs no compile check on .py | arm fixed both + proved repair inert (byte-identical stream) | — | py_compile/E999 guard in the commit hook — owned MAIN, needs care (commit-path blast radius, #852 lineage) |
| E11 | MAIN relayed fork reports with GUESSED paths, corrected one message later | paths typed from memory | correction sent | never-guess-paths: verify with ls/Glob before citing — folded into the memory file below | none |

Sister incidents consumed: session-limit fleet wipe (recovery deterministic: transcripts survive,
SendMessage resumes; queue recorded in hot state at spawn) · tc1 fork peer-unreachability (final
messages ARE the durable delivery channel per np1; relay by exact path).

Memory: `hand_assembled_dispatch_is_the_error_factory_20260817` + MEMORY.md line. The three
residuals above are the open protection debt; each has an owner (MAIN) and a named window.

STORES CONSULTED: ddm_rr4_cuda_prob_reencode_20260817.md (§3.2, §6, §8) ·
ddm_rr2_t4_refusal_device_scoped_decode_identity_20260817.md · m77 waiter laws · m05/m23 ·
never-invent-flags (CLAUDE.md) · the-control-plane-fails-silently memory ·
ddm_ac1/ddm_dt1 determinization program (#1047 — this ledger extends it to the dispatch surface).
