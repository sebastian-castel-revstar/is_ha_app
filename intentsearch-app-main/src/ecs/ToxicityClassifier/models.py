from typing import Literal

from pydantic import BaseModel, Field, confloat, validator


class Request(BaseModel):
    text: str = Field(..., description="The search query string")
    region_name: Literal["NA"] = Field(
        ..., description="Currently only North America(NA) is supported"
    )

    @validator("text", pre=True)
    def not_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("Field 'text' cannot be an empty string")
        return v


class ToxicityScores(BaseModel):
    toxic: confloat(ge=0, le=1)
    severe_toxic: confloat(ge=0, le=1)
    obscene: confloat(ge=0, le=1)
    threat: confloat(ge=0, le=1)
    insult: confloat(ge=0, le=1)
    identity_hate: confloat(ge=0, le=1)


class Response(BaseModel):
    toxicity: ToxicityScores


class ErrorMessage(BaseModel):
    error_detail: str = Field(description="Error message detail")
