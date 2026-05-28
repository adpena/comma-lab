"""Register the canonical anti-pattern for operator-shared JS-rendered SPA URLs
inaccessible to WebFetch.

Source anchor: subagent operator_override_review_paper_plus_conversation_20260528
2026-05-28 STAND_DOWN + COMPLEMENTARY landing per Sister Variant 1+2 hybrid.

3 empirical recurrences today (this turn 4 WebFetch attempts + sister
operator_override_review_paper_rudin_daubechies_20260528 4 attempts) + 1 prior
Slot M 2026-05-19 Wayback-empty anchor = META-pattern confirmed.
"""

from __future__ import annotations

from tac.canonical_anti_patterns import AntiPattern, register_anti_pattern
from tac.provenance.builders import build_provenance_for_predicted


def main() -> None:
    """Register the canonical anti-pattern via canonical helper."""
    provenance = build_provenance_for_predicted(
        model_id="canonical_anti_patterns.builtins.operator_shared_js_rendered_spa_url_inaccessible_to_webfetch_v1",
        inputs_sha256="0" * 64,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc="2026-05-28T20:40:00Z",
    )

    anti_pattern = AntiPattern(
        anti_pattern_id="operator_shared_js_rendered_spa_url_inaccessible_to_webfetch_v1",
        description=(
            "Operator-shared URLs to JS-rendered SPA conversation/document hosts "
            "(chatgpt.com/share/*, chat.openai.com/share/*, claude.ai/chat/*, "
            "personal-blog SPAs like aaronleslie.dev/blog/*) are NOT extractable "
            "via WebFetch / curl / shell tooling. SSR shell carries TITLE + "
            "metadata only; conversation message text loads via authenticated API "
            "on browser hydration. Subagents that try to extract via 4+ fetch "
            "attempts waste tokens + fail to deliver. CANONICAL response: "
            "DEFER-pending-operator-paste of the conversation text inline."
        ),
        forbidden_pattern_predicate=(
            "operator.provides_url(url) AND "
            "(url.host MATCHES (chatgpt.com|chat.openai.com|claude.ai|aaronleslie.dev|*spa-rendered-domain*) "
            "OR url.path STARTS_WITH /share/) AND "
            "NOT operator.paste_content_inline AND "
            "NOT canonical_helper(headless_browser_render).exists"
        ),
        falsification_band={
            "webfetch_success_rate_lo": 0.0,
            "webfetch_success_rate_hi": 0.05,
        },
        recurrence_conditions=(
            "WebFetch returns ChatGPT UI navigation only (no conversation content)",
            "curl with Chrome User-Agent returns SSR HTML with conversation TITLE + 29 message references but no actual message text",
            "curl <share-url>.json returns identical SSR shell (no JSON variant)",
            "chatgpt.com/backend-api/share/<uuid> returns HTTP 403 Forbidden",
            "Wayback Machine + Google cache return empty for personal-blog SPAs",
            "Subagent operator_override_review_paper_plus_conversation_20260528 hits all 4 above 2026-05-28T20:30Z",
            "Sister operator_override_review_paper_rudin_daubechies_20260528 hits same 4 2026-05-28T20:28Z",
            "Slot M 2026-05-19 feedback_pr_95_full_deep_research_landed_20260519T192300Z Wayback-empty anchor",
        ),
        canonical_source_anchor=(
            "subagent_id:operator_override_review_paper_plus_conversation_20260528 "
            "+ subagent_id:operator_override_review_paper_rudin_daubechies_20260528 "
            "+ feedback_pr_95_full_deep_research_landed_20260519T192300Z "
            "+ .omx/research/operator_override_review_paper_STAND_DOWN_per_sister_convergence_20260528.md"
        ),
        canonical_unwind_path=(
            "(1) IMMEDIATE: operator pastes the conversation text inline in next prompt "
            "(no apparatus change needed; sister-Variant-1 STAND_DOWN + COMPLEMENTARY "
            "anti-pattern lands). (2) SHORT-TERM: future subagents query "
            "tac.canonical_anti_patterns.load_anti_patterns_strict() filtering by this "
            "anti_pattern_id and STAND_DOWN-with-DEFER-pending-operator-paste BEFORE "
            "spending tokens on 4 redundant fetch attempts. (3) LONG-TERM: queue canonical "
            "helper tac.operator_workflow.fetch_or_defer_share_url(url) -> "
            "SharedConversationVerdict that returns INACCESSIBLE_JS_SPA_DEFER_PENDING_"
            "OPERATOR_PASTE verdict immediately for any URL matching the predicate, AND "
            "optionally invokes a headless-browser canonical helper if available locally."
        ),
        canonical_producers=(
            "tools/register_anti_pattern_operator_shared_js_spa_url_inaccessible.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "future:tac.operator_workflow.fetch_or_defer_share_url",
        ),
        paradigm_class="discipline_anti_pattern",
        severity="medium_substrate_regression",
        provenance=provenance,
        empirical_falsifications=(),
        last_recalibration_utc="2026-05-28T20:40:00Z",
        next_recalibration_trigger="when_3+_new_empirical_falsifications_in_domain",
    )

    result = register_anti_pattern(
        anti_pattern,
        agent="claude",
        subagent_id="operator_override_review_paper_plus_conversation_20260528",
        notes=(
            "STAND_DOWN + COMPLEMENTARY canonical apparatus mutation per Sister "
            "Variant 1+2 hybrid; operator override directive 2026-05-28 'Memos "
            "must be acted upon'; review-memo content owned by sister "
            "operator_override_review_paper_rudin_daubechies_20260528."
        ),
    )
    print(f"Registered: {result.anti_pattern_id}")
    print(f"Severity: {result.severity}")
    print(f"Paradigm class: {result.paradigm_class}")


if __name__ == "__main__":
    main()
