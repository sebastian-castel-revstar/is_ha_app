from typing import Dict, Optional

from aws_lambda_powertools import Logger
from pydantic import BaseModel, Field

from GCPGemini.GCPService.Constants import (
    GCP_MODEL_ID_GEMINI_FLASH_2_0,
)


class GcpConfig(BaseModel):
    service_account_info: Dict
    cached_gcp_cred: Optional[Dict]
    project_id: str
    location: str
    model_id: str = Field(
        default=GCP_MODEL_ID_GEMINI_FLASH_2_0, description="The model ID to use."
    )
    logger: Logger

    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ()
