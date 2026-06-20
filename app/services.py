"""Google Gemini AI service for Amazon Ads search term optimization."""

import asyncio
import json
import logging

from google import genai
from google.genai import types

from app.config import settings
from app.models import (
    ActionType,
    HarvestRequest,
    HarvestResponse,
    OptimizationAction,
    PriorityLevel,
)

logger = logging.getLogger(__name__)

# New SDK uses a Client object that holds the API key
client = genai.Client(api_key=settings.GEMINI_API_KEY)


def build_prompt(request: HarvestRequest) -> str:
    """Build the analysis prompt with campaign context and decision guidelines."""
    search_term_sections = []
    for i, term in enumerate(request.search_terms, 1):
        m = term.metrics

        acos_display = (
            f"{m.acos:.2f}%" if m.acos is not None else "N/A (no sales yet)"
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

CAMPAIGN INFORMATION
Campaign      : {request.campaign_name} (ID: {request.campaign_id})
Business      : {request.business_context}
Target ACoS   : {request.user_constraints.target_acos}%
Max Bid       : ${request.user_constraints.max_bid:.2f}

FIELD DEFINITIONS
Search Term     : The actual query the customer typed. Used for add_negative_keyword.
Target Text     : The keyword or auto-target expression you are bidding on.
Match Type      : EXACT/PHRASE = tight match. BROAD/TARGETING_EXPRESSION/
                  TARGETING_EXPRESSION_PREDEFINED = wider matching.
Target Type     : keyword / auto / product
Current Bid     : "N/A" means auto-target with default bid — use observed CPC as anchor.
ACoS            : (Spend/Sales)*100. null means zero sales. Primary decision metric.
Revenue/Click   : Sales/Clicks. High value vs CPC = increase_bid signal.

SEARCH TERM PERFORMANCE DATA
{search_terms_block}

DECISION FRAMEWORK

ACTION 1 — increase_bid
  ACoS significantly below target, healthy conversion rate, sufficient clicks.
  Bid math: current_bid × 1.10 to 1.20. If current_bid is N/A, use CPC × 1.5, capped at max_bid.

ACTION 2 — decrease_bid
  ACoS above target but sales > 0. Bid math: current_bid × 0.70 to 0.90.

ACTION 3 — add_negative_keyword
  Zero/near-zero sales with meaningful spend AND clicks >= 10, OR semantically
  irrelevant to business context (wrong intent: informational, competitor,
  repair, accessory, job-seeking, different product category).

ACTION 4 — no_action
  Clicks < 10: insufficient data regardless of metrics. Prefer no_action over
  a wrong action when uncertain.

PRIORITY ASSIGNMENT
HIGH   : confidence >= 0.85 AND (ACoS > 2x target with sales > 0, OR zero sales
         with spend > max_bid*3 and clicks >= 15, OR ACoS < target*0.5 with sales > 0,
         OR semantically irrelevant term with high spend)
MEDIUM : confidence >= 0.65 and does not meet HIGH criteria
LOW    : confidence < 0.65, OR clicks < 10, OR no_action cases

HARD RULES
1. NEVER recommend a bid above ${request.user_constraints.max_bid:.2f}
2. For add_negative_keyword and no_action: omit recommended_bid entirely
3. Always copy the exact target_id and ad_group_id from the input
4. Return EXACTLY one action per search term, in the same order as input
5. When clicks < 10 and uncertain: default to no_action
6. Confidence cap: when clicks < 10, cap confidence at 0.75
7. Reasoning must reference the actual numbers from the data

OUTPUT FORMAT
Return structured JSON with one action per search term and a summary
highlighting HIGH priority actions first.
"""
    return prompt


def build_response_schema() -> dict:
    """JSON schema enforcing Gemini structured output."""
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
                    ],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["actions", "summary"],
    }


async def analyze_search_terms(request: HarvestRequest) -> HarvestResponse:
    """Core service function using the google-genai SDK."""
    logger.info(
        "Analysis started | campaign='%s' | terms=%d",
        request.campaign_name,
        len(request.search_terms),
    )

    valid_target_ids = {term.target_id for term in request.search_terms}

    try:
        prompt = build_prompt(request)
        logger.debug("Prompt length: %d characters", len(prompt))

        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.GEMINI_TEMPERATURE,
                    response_mime_type="application/json",
                    response_schema=build_response_schema(),
                    max_output_tokens=settings.GEMINI_MAX_TOKENS,
                ),
            ),
            timeout=120,
        )

        parsed = json.loads(response.text)

        actions = []
        for raw_action in parsed["actions"]:
            target_id = raw_action["target_id"]

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

    except json.JSONDecodeError as exc:
        logger.error("Gemini returned invalid JSON: %s", exc)
        raise ValueError("AI model returned malformed response") from exc

    except asyncio.TimeoutError as exc:
        logger.error("Gemini call timed out after 120 seconds")
        raise RuntimeError(
            "AI analysis failed: request timed out after 120 seconds"
        ) from exc

    except ValueError:
        raise

    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        raise RuntimeError(f"AI analysis failed: {exc}") from exc

