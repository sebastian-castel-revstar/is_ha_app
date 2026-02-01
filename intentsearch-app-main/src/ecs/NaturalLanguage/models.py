from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_serializer, validator


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


class Request(BaseModel):
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


class Destination(BaseModel):
    name: str
    lat: Optional[float]
    lon: Optional[float]


class Recommendations(BaseModel):
    recommendations: List[Destination]


class EntitySentiment(BaseModel):
    name: str
    sentiment: Optional[str]


class PriceFilter(BaseModel):
    threshold: Optional[float]
    currency: Optional[str]
    operator: Optional[Literal["eq", "gte", "lte"]]
    rate: Optional[Literal["pernight", "total"]]


class PriceThreshold(BaseModel):
    value: Optional[float]
    currency: Optional[str]
    operator: Optional[Literal["gte", "lte", "eq"]]


class AwardCategory(BaseModel):
    value: Optional[int]
    operator: Optional[Literal["gte", "lte", "eq"]]


class EnhancedPriceFilter(BaseModel):
    threshold: Optional[List[PriceThreshold]]
    rate: Optional[Literal["pernight", "total"]]


class Response(BaseModel):
    query_embedding: Optional[List[float]]
    embedding_source: Optional[str]
    destinations: List[Destination]
    brands: List[EntitySentiment]
    price_filter: PriceFilter
    enhanced_price_filter: EnhancedPriceFilter
    themes: List[EntitySentiment]
    experiences: List[EntitySentiment]
    attributes: List[EntitySentiment]
    points_of_interest: List[EntitySentiment]
    accommodation_type: List[EntitySentiment]
    traveler_type: List[EntitySentiment]
    weather_geography: List[EntitySentiment]
    award_category: List[AwardCategory]
    hotel_characteristics: List[EntitySentiment]
    places: List[EntitySentiment]


class ErrorMessage(BaseModel):
    error_detail: str = Field(description="Error message detail")
