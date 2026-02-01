from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, confloat, field_serializer, validator


# Validate request
class SessionContext(BaseModel):
    local_ts: Optional[str]
    country_name: Optional[str]
    region_name: Optional[str]
    city_name: Optional[str]
    user_agent: str

    @validator("user_agent")
    def user_agent_not_empty(cls, v):
        if not v:
            raise ValueError("user_agent cannot be an empty string")
        return v


class SearchContext(BaseModel):
    query: str
    arrival_date: Optional[date]
    departure_date: Optional[date]
    num_rooms: Optional[int]
    accessible: Optional[bool]
    num_adults: Optional[int]
    num_children: Optional[int]
    special_rate_code: Optional[str]
    special_rate_category: Optional[str]
    use_points: Optional[bool]

    @validator("query")
    def query_not_empty(cls, v):
        if not v:
            raise ValueError("query cannot be an empty string")
        return v

    @field_serializer("arrival_date")
    def serialize_arrival(self, dt: date) -> str:
        return dt.strftime("%Y-%m-%d")

    @field_serializer("departure_date")
    def serialize_departure(self, dt: date) -> str:
        return dt.strftime("%Y-%m-%d")


class APIRequest(BaseModel):
    search_id: str = Field(
        ..., description="UUID string generated for each search flow execution."
    )
    visitor_id: str = Field(..., description="AdobeId for Web, TealiumId for App.")
    member_id: Optional[str] = Field(
        ..., description="Member id if logged in. NULLABLE if not logged in."
    )
    channel: Literal["web", "app"] = Field(..., description="Channel ENUM: [web, app].")
    session_context: SessionContext
    search_context: SearchContext

    @validator("search_id", "visitor_id", "channel", pre=True)
    def not_empty(cls, v):
        if isinstance(v, str) and not v:
            raise ValueError("Field cannot be an empty string")
        return v


class ToxicityScores(BaseModel):
    toxic: confloat(ge=0, le=1)
    severe_toxic: confloat(ge=0, le=1)
    obscene: confloat(ge=0, le=1)
    threat: confloat(ge=0, le=1)
    insult: confloat(ge=0, le=1)
    identity_hate: confloat(ge=0, le=1)


class APIResponse(BaseModel):
    label: Literal["INTENT", "DESTINATION", "IRRELEVANT"]
    probability: confloat(ge=0, le=1)
    toxicity: ToxicityScores


class ErrorMessage(BaseModel):
    error_detail: str = Field(description="Error message detail")
