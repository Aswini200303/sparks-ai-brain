"""Pydantic models for Sparks AI Brain — Amazon Ads optimization."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    ADD_NEGATIVE_EXACT = "add_negative_exact"
    ADD_NEGATIVE_PHRASE = "add_negative_phrase"
    CREATE_EXACT_KEYWORD = "create_exact_keyword"
    NO_ACTION = "no_action"


class PriorityLevel(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class MetricPayload(BaseModel):
    # ── Volume ───────────────────────────────────────────────────
    impressions: int   = Field(..., ge=0)
    clicks:      int   = Field(..., ge=0)
    spend:       float = Field(..., ge=0, description="Total cost from Search Term report")
    sales:       float = Field(..., ge=0, description="Total attributed revenue")
    orders:      int   = Field(..., ge=0, description="Total purchases/conversions")

    # ── Bid anchor (from Targeting report) ───────────────────────
    current_bid: Optional[float] = Field(
        None,
        description="target_bid from Targeting report. "
                    "null for auto/predefined targets."
    )

    # ── Derived by Node.js before sending ────────────────────────
    acos: Optional[float] = Field(
        None,
        description="(spend/sales)*100. null when sales=0. "
                    "Never send 0.0 — use null to mean no sales yet."
    )
    ctr:                float          = Field(..., ge=0, description="CTR % parsed from string")
    cpc:                float          = Field(..., ge=0, description="spend/clicks. 0 if no clicks.")
    conversion_rate:    float          = Field(..., ge=0, description="(orders/clicks)*100")
    cost_per_conversion: Optional[float] = Field(
        None,
        description="Cost per purchase. null when orders=0."
    )
    revenue_per_click: float = Field(
        ..., ge=0,
        description="sales/clicks. 0 if no clicks. "
                    "May indicate strong customer intent."
    )


class SearchTermData(BaseModel):
    # ── From Search Term report ───────────────────────────────────
    search_term: str = Field(
        ...,
        description="Raw customer query. Only source for add_negative text."
    )
    target_id:   str = Field(
        ...,
        description="Join key. Node.js uses to call Amazon bid update API."
    )
    ad_group_id: str = Field(
        ...,
        description="Required for scoping negative keyword correctly."
    )

    # ── From Targeting report (joined on target_id) ───────────────
    target_text: str = Field(
        ...,
        description="Keyword or auto-target expression being bid on."
    )
    match_type: str = Field(
        ...,
        description="EXACT / BROAD / PHRASE / "
                    "TARGETING_EXPRESSION / TARGETING_EXPRESSION_PREDEFINED"
    )
    target_type: str = Field(
        ...,
        description="keyword / auto / product. "
                    "Node.js uses to call correct Amazon API endpoint."
    )
    target_status: str = Field(
        ...,
        description="enabled / paused / archived. "
                    "Should always be 'enabled' — "
                    "Node.js must filter out non-enabled before sending."
    )

    # ── Performance ───────────────────────────────────────────────
    metrics: MetricPayload


class UserConstraints(BaseModel):
    target_acos: float = Field(
        ..., gt=0, lt=100,
        description="Target ACoS %. Node.js uses this for mathematical optimization."
    )
    max_bid: float = Field(
        ..., gt=0,
        description="Hard bid ceiling. Node.js uses this for bid optimization."
    )


class HarvestRequest(BaseModel):
    campaign_id:      str
    campaign_name:    str
    business_context: str = Field(
        ...,
        description="What the campaign sells and target audience. "
                    "From DB. Gemini uses for semantic relevance checks."
    )
    user_constraints: UserConstraints
    search_terms:     List[SearchTermData] = Field(..., min_length=1)


class OptimizationAction(BaseModel):
    search_term:    str
    target_id:      str   = Field(..., description="Echoed from input for Node.js API call.")
    ad_group_id:    str
    action:         ActionType
    reasoning:  str
    confidence: float         = Field(..., ge=0, le=1)
    priority:   PriorityLevel


class HarvestResponse(BaseModel):
    campaign_id:          str
    campaign_name:        str
    total_terms_analyzed: int
    actions:              List[OptimizationAction]
    summary:              str
    model_used:           str = Field(default="gemini-2.5-flash")
    generated_at:         str = Field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    )
