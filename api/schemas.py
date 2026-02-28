"""
Pydantic schemas for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional


class FlightInput(BaseModel):
    """Input schema for flight delay prediction request."""

    month: int = Field(..., ge=1, le=12, description="Month of flight (1=Jan, 12=Dec)")
    day_of_week: int = Field(..., ge=1, le=7, description="Day of week (1=Mon, 7=Sun)")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    dep_hour: int = Field(..., ge=0, le=23, description="Scheduled departure hour (0-23)")
    carrier: str = Field(..., min_length=2, max_length=2, description="2-letter airline code e.g. AA")
    origin: str = Field(..., min_length=3, max_length=3, description="3-letter origin airport code e.g. JFK")
    dest: str = Field(..., min_length=3, max_length=3, description="3-letter destination airport code e.g. LAX")
    distance: float = Field(..., gt=0, le=10000, description="Flight distance in miles")
    crs_elapsed_time: float = Field(..., gt=0, le=1200, description="Scheduled flight duration in minutes")

    @validator("carrier")
    def carrier_uppercase(cls, v):
        return v.upper()

    @validator("origin", "dest")
    def airport_uppercase(cls, v):
        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
                "month": 6,
                "day_of_week": 5,
                "day_of_month": 15,
                "dep_hour": 8,
                "carrier": "AA",
                "origin": "JFK",
                "dest": "LAX",
                "distance": 2475.0,
                "crs_elapsed_time": 330.0,
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction response."""

    delayed: int = Field(..., description="1 = Delayed (>15 min), 0 = On Time")
    probability: float = Field(..., description="Probability of delay (0.0 to 1.0)")
    status: str = Field(..., description="Human-readable status: 'DELAYED' or 'ON TIME'")
    confidence: str = Field(..., description="Confidence level: HIGH / MEDIUM / LOW")

    class Config:
        json_schema_extra = {
            "example": {
                "delayed": 1,
                "probability": 0.73,
                "status": "DELAYED",
                "confidence": "HIGH",
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"
