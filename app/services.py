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
            f"{m.acos:.2f}%" if m.acos is not None else "N/A"
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
"""
        search_term_sections.append(section)

    search_terms_block = "\n".join(search_term_sections)

    prompt = f"""You are an expert Amazon Ads semantic relevance analyst with deep knowledge of
Sponsored Products search-term intent and keyword harvesting strategies.

Ignore bid optimization and mathematical calculations. Node.js handles bids and ACoS.
Your job is only semantic relevance and intent analysis.

Analyze each search term and return exactly one semantic action per search term.

CAMPAIGN INFORMATION
Campaign      : {request.campaign_name} (ID: {request.campaign_id})
Business      : {request.business_context}

FIELD DEFINITIONS
Search Term     : The actual query the customer typed.
Target Text     : The keyword or auto-target expression you are bidding on.
Match Type      : EXACT/PHRASE = tight match. BROAD/TARGETING_EXPRESSION/
                  TARGETING_EXPRESSION_PREDEFINED = wider matching.
Target Type     : keyword / auto / product
Performance     : Use conversions, sales, and volume only as evidence of customer
                  intent strength. Do not calculate bids or optimize from ACoS.

SEARCH TERM PERFORMANCE DATA
{search_terms_block}

DECISION FRAMEWORK

ACTION 1 — add_negative_exact
  Use when the specific search term is irrelevant to the product, but the issue
  appears limited to that exact query. Example: "window cleaner" for a shower
  glass cleaner product.

ACTION 2 — add_negative_phrase
  Use when the root concept is bad and many variants are likely irrelevant.
  Common bad roots include repair, replacement, reviews, manual, parts, jobs,
  free, used, DIY, and competitor-only research terms.

ACTION 3 — create_exact_keyword
  Use when the search term is highly relevant and conversions or sales indicate
  strong purchase intent, especially when discovered from BROAD, AUTO,
  TARGETING_EXPRESSION, or TARGETING_EXPRESSION_PREDEFINED traffic.

ACTION 4 — no_action
  Use when more data is needed, the query is acceptable as-is, or you are not
  confident enough to block or graduate it.

PRIORITY ASSIGNMENT
HIGH   : high-confidence semantic block or high-confidence exact keyword graduation
MEDIUM : useful semantic signal, but with moderate ambiguity or limited data
LOW    : no_action cases, weak evidence, or low confidence

HARD RULES
1. Do NOT calculate bid increases, bid decreases, bid percentages, or bid prices
2. Do NOT recommend or return recommended_bid
3. Do NOT optimize based on ACoS math; Node.js handles ACoS and bid decisions
4. Always copy the exact target_id and ad_group_id from the input
5. Return EXACTLY one action per search term, in the same order as input
6. Irrelevant query -> add_negative_exact
7. Bad root concept -> add_negative_phrase
8. Highly profitable broad/auto query -> create_exact_keyword
9. Need more data or query is acceptable -> no_action
10. Reasoning must explain semantic relevance and intent, optionally citing
    conversions, sales, orders, clicks, or match type as supporting evidence

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
                                "add_negative_exact",
                                "add_negative_phrase",
                                "create_exact_keyword",
                                "no_action",
                            ],
                        },
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
