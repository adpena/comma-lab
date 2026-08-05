# NEXT_IF_RESUMED ddm_jd4

1. Do not launch until the n600 both-bases endpoint probe completes and MAIN adjudicates the endpoint basis. This arm did not own the scorer slot.
2. Fire the ticket only through the governed launcher, consuming `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd4_ticket_cont_ep1406.json`. Do not hand-run the printed argv.
3. Before launch, refuse if `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1406/checkpoints/` already contains checkpoint NPZs; regenerate to a new child out-dir instead.
4. At resume, require telemetry rows `jd1_force_resume_epoch_reanchor` and `jd1_stage_ema_reanchor` with `forced_from_resume=true`, old carried U/provenance from the smoke continuation, and new derived U=18000 / decay `0.9997777777777778`.
5. If those rows are absent, stop and treat the ticket as not enforcing RR1-R4-F1; do not consume the endpoint as a full-window EMA-cured continuation.
6. Keep `recursive_encode_pass_loop.next_resume_from_template` under the jd4 child out-dir for any next pass. Never consume the old JD1 ancestor template from the prior fired ticket.
7. Keep `levers[*].overrides` value-custody strict: every declared override must match final argv after rewrites.
8. Any follow-on from the jd4 endpoint exits FIRED by MAIN, FOLDED with a typed reason, or QUEUED-WITH-A-FIRE-ORDER. Do not leave it as "noted."

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
