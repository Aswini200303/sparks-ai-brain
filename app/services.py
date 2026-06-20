"""Google Gemini AI service for Amazon Ads search term optimization."""

import asyncio
import json
import logging

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.config import settings
from app.models import (
    ActionType,
    HarvestRequest,
    HarvestResponse,
    OptimizationAction,
    PriorityLevel,
)

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)


def build_prompt(request: HarvestRequest) -> str:
    """Build Gemini prompt with full campaign context and per-term metrics."""

    search_term_sections = []
    for i, term in enumerate(request.search_terms, 1):
        m = term.metrics

        acos_display = (
            f"{m.acos:.2f}%"
            if m.acos is not None
            else "N/A (no sales yet)"
        )
        current_bid_display = (
            f"${m.current_bid:.2f}"
            if m.current_bid is not None
            else "N/A (auto-target — no explicit bid set, use observed CPC as anchor)"
        )
        cost_per_conv_display = (
            f"${m.cost_per_conversion:.2f}"
            if m.cost_per_conversion is not None
            else "N/A (no conversions yet)"
        )

        section = f"""
[Term {i}]
Search Term      : {term.search_term}
Target Text      : {term.target_text}
Target ID        : {term.target_id}
Ad Group ID      : {term.ad_group_id}
Match Type       : {term.match_type}
Target Type      : {term.target_type}
Target Status    : {term.target_status}

Performance (last 7 days):
  Impressions      : {m.impressions}
  Clicks           : {m.clicks}
  Spend            : ${m.spend:.2f}
  Sales            : ${m.sales:.2f}
  Orders           : {m.orders}
  ACoS             : {acos_display}
  CTR              : {m.ctr:.4f}%
  CPC              : ${m.cpc:.2f}
  Conversion Rate  : {m.conversion_rate:.2f}%
  Cost/Conversion  : {cost_per_conv_display}
  Revenue/Click    : ${m.revenue_per_click:.2f}
  Current Bid      : {current_bid_display}
"""
        search_term_sections.append(section)

    search_terms_block = "\n".join(search_term_sections)

    prompt = f"""You are an expert Amazon Ads optimization analyst with deep knowledge of
Sponsored Products bid management and keyword harvesting strategies.

Your task is to analyze each search term's performance data and return
exactly one optimization action per search term.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAMPAIGN INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Campaign      : {request.campaign_name} (ID: {request.campaign_id})
Business      : {request.business_context}
Target ACoS   : {request.user_constraints.target_acos}%
Max Bid       : ${request.user_constraints.max_bid:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Search Term     : The actual query the customer typed into Amazon search.
                  This is the text used for add_negative_keyword actions.

Target Text     : The keyword or auto-target expression you are bidding on.
                  For EXACT/PHRASE/BROAD match: this is the keyword.
                  For TARGETING_EXPRESSION / TARGETING_EXPRESSION_PREDEFINED:
                  this is an auto-target group (close-match, loose-match, etc.)
                  The search term may DIFFER from the target text — especially
                  in broad/auto campaigns.

Match Type      : How loosely the target matches customer queries.
                  EXACT = very tight match.
                  BROAD / TARGETING_EXPRESSION / TARGETING_EXPRESSION_PREDEFINED
                  = wider matching, more likely to surface irrelevant queries.
                  For BROAD/AUTO match types, adding a negative keyword blocks
                  only this specific query — it does NOT disable the whole target.

Target Type     : keyword / auto / product
                  This tells you what TYPE of Amazon targeting is in use.

Current Bid     : The explicit bid set on this target.
                  "N/A" means it is an auto-target with Amazon's default bid —
                  in this case use the observed CPC as the bid anchor
                  when recommending a starting bid.

ACoS            : (Spend / Sales) × 100. null means zero sales so far.
                  This is your PRIMARY decision metric.
                  Compare against Target ACoS of {request.user_constraints.target_acos}%.

Revenue/Click   : Sales ÷ Clicks. A high Revenue/Click relative to CPC
                  is a strong signal to increase bid and capture more volume.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEARCH TERM PERFORMANCE DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{search_terms_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACTION 1 — increase_bid
  When to use:
  • ACoS is significantly below target ACoS (strong performer)
  • Conversion rate is healthy (> 0 orders)
  • Revenue/Click is high relative to CPC
  • Sufficient data (clicks >= 10 preferred)
  Bid math: new bid = current_bid × 1.10 to 1.20 (10–20% increase)
  If current_bid is N/A: set a starting bid using observed CPC × 1.5,
  capped at max_bid.

ACTION 2 — decrease_bid
  When to use:
  • ACoS is above target ACoS but sales > 0 (overspending but converting)
  • NOT zero conversions (that is add_negative_keyword territory)
  Bid math: new bid = current_bid × 0.70 to 0.90 (10–30% decrease)

ACTION 3 — add_negative_keyword
  When to use:
  • Zero or near-zero sales despite meaningful spend AND clicks >= 10
  • Search term is semantically IRRELEVANT to the business context
    (wrong intent: informational, competitor brand, repair, accessory,
     job-seeking, or completely different product category)
  • The "Search Term" text (not target text) is what gets negated
  • For BROAD / AUTO match types: safe to negate the specific query
    without disabling the whole targeting expression
  Note: Even if ACoS looks acceptable, add_negative if the search term
  is semantically irrelevant to the business. A lucky conversion on an
  irrelevant term does not make the term relevant.

ACTION 4 — no_action
  When to use:
  • Clicks < 10: insufficient data regardless of metrics
  • ACoS is near target (within 5 percentage points either way): stable
  • Data is ambiguous and more monitoring is needed
  Note: Prefer no_action over a wrong action. Conservative is better
  than incorrect when money is involved.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIORITY ASSIGNMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIGH   : confidence >= 0.85 AND any of:
         • ACoS > 2× target_acos with sales > 0
         • Zero sales with spend > max_bid × 3 and clicks >= 15
         • ACoS < target_acos × 0.5 with sales > 0
         • Semantically irrelevant term with high spend

MEDIUM : confidence >= 0.65 and does not meet HIGH criteria

LOW    : confidence < 0.65
         OR clicks < 10 (insufficient data)
         OR no_action cases

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD RULES — NEVER VIOLATE THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER recommend a bid above ${request.user_constraints.max_bid:.2f} (max_bid)
2. For add_negative_keyword and no_action: omit recommended_bid entirely
3. Always copy the exact target_id from the input — never modify or invent it
4. Always copy the exact ad_group_id from the input
5. Return EXACTLY one action per search term, in the same order as input
6. When clicks < 10 and you are uncertain: default to no_action
7. Confidence cap: when clicks < 10, cap confidence at 0.75 regardless
   of how good the metrics look (small samples carry statistical noise)
8. Reasoning must reference the actual numbers from the data
   (e.g. "ACoS 13.3% is below the 25% target" — not generic statements)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return structured JSON with:
- One action per search term (same count and order as input)
- A summary paragraph highlighting HIGH priority actions first
- Specific numbers in every reasoning field
"""
    return prompt


def build_response_schema() -> dict:
    """
    JSON schema enforcing Gemini structured output.

    Rules:
    - No 'minimum', 'maximum', or 'nullable' — Gemini SDK rejects these
    - recommended_bid is NOT in required — Gemini omits it for
      add_negative_keyword and no_action
    - priority and target_id ARE required for every action
    """
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "search_term":     {"type": "string"},
                        "target_id":       {"type": "string"},
                        "ad_group_id":     {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": [
                                "increase_bid",
                                "decrease_bid",
                                "add_negative_keyword",
                                "no_action",
                            ],
                        },
                        "recommended_bid": {"type": "number"},
                        "reasoning":       {"type": "string"},
                        "confidence":      {"type": "number"},
                        "priority": {
                            "type": "string",
                            "enum": ["HIGH", "MEDIUM", "LOW"],
                        },
                    },
                    "required": [
                        "search_term",
                        "target_id",
                        "ad_group_id",
                        "action",
                        "reasoning",
                        "confidence",
                        "priority",
                        # recommended_bid intentionally excluded —
                        # Gemini omits it for add_negative_keyword / no_action
                    ],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["actions", "summary"],
    }


async def analyze_search_terms(request: HarvestRequest) -> HarvestResponse:
    """
    Core service function.

    Flow:
    1. Build valid_target_ids set for hallucination guard
    2. Build prompt
    3. Call Gemini with structured output config
    4. Parse and validate response
    5. Return HarvestResponse
    """
    logger.info(
        "Analysis started | campaign='%s' | terms=%d",
        request.campaign_name,
        len(request.search_terms),
    )

    # ── Hallucination guard ───────────────────────────────────────
    # Gemini must echo back a target_id that exists in the input.
    # If it invents one, we catch it here and return a clean 422
    # instead of passing bad data downstream to Node.js.
    valid_target_ids = {term.target_id for term in request.search_terms}

    try:
        model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)
        prompt = build_prompt(request)

        logger.debug("Prompt length: %d characters", len(prompt))

        generation_config = GenerationConfig(
            temperature=settings.GEMINI_TEMPERATURE,
            # temperature=0.1 → consistent, deterministic decisions
            # Never use high temperature for financial bid decisions

            response_mime_type="application/json",
            # Forces Gemini to return raw JSON, not markdown code blocks

            response_schema=build_response_schema(),
            # Enforces exact structure — Gemini cannot return free text

            max_output_tokens=settings.GEMINI_MAX_TOKENS,
        )

        # ── Gemini API call with timeout ──────────────────────────
        response = await asyncio.wait_for(
            model.generate_content_async(
                prompt,
                generation_config=generation_config,
            ),
            timeout=120,
            # 120s timeout — Gemini cold starts can exceed 60s
            # especially for large batches of search terms
        )

        # ── Parse response ────────────────────────────────────────
        parsed = json.loads(response.text)

        # ── Build OptimizationAction list ─────────────────────────
        actions = []
        for raw_action in parsed["actions"]:

            target_id = raw_action["target_id"]

            # Hallucination check — target_id must match input
            if target_id not in valid_target_ids:
                raise ValueError(
                    f"Gemini returned unknown target_id '{target_id}' "
                    f"for search term '{raw_action.get('search_term', 'unknown')}'. "
                    f"Valid IDs are: {sorted(valid_target_ids)}"
                )

            actions.append(
                OptimizationAction(
                    search_term=raw_action["search_term"],
                    target_id=target_id,
                    ad_group_id=raw_action["ad_group_id"],
                    action=ActionType(raw_action["action"]),
                    recommended_bid=raw_action.get("recommended_bid"),
                    reasoning=raw_action["reasoning"],
                    confidence=float(raw_action["confidence"]),
                    priority=PriorityLevel(raw_action["priority"]),
                )
            )

        logger.info(
            "Analysis complete | campaign='%s' | actions=%d",
            request.campaign_name,
            len(actions),
        )

        return HarvestResponse(
            campaign_id=request.campaign_id,
            campaign_name=request.campaign_name,
            total_terms_analyzed=len(request.search_terms),
            actions=actions,
            summary=parsed["summary"],
            model_used=settings.GEMINI_MODEL,
        )

    # ── Error handling ────────────────────────────────────────────
    except json.JSONDecodeError as exc:
        logger.error("Gemini returned invalid JSON: %s", exc)
        raise ValueError("AI model returned malformed response") from exc

    except asyncio.TimeoutError as exc:
        logger.error("Gemini call timed out after 120 seconds")
        raise RuntimeError(
            "AI analysis failed: request timed out after 120 seconds"
        ) from exc

    except ValueError:
        # Re-raise our own validation errors (target_id mismatch,
        # bad ActionType/PriorityLevel enum value from Gemini).
        # main.py maps ValueError → HTTP 422 ai_response_invalid.
        raise

    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        raise RuntimeError(f"AI analysis failed: {exc}") from exc