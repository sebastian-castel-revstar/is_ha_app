from typing import Union

from pydantic import BaseModel, ValidationError

from GCPGemini.GCPService.Constants import GCP_SERVICE
from GCPGemini.GCPService.GcpConfig import GcpConfig
from GCPGemini.GCPService.GcpService import GcpSummaryService

"""
Factory class to create the appropriate summary service based on the service name.
"""


class FactoryService:
    @staticmethod
    def get_factory_service(
        service_name: str, config: BaseModel
    ) -> Union[
        GcpConfig,
        # CohereSummaryServiceConfig, TitanSummaryServiceConfig
    ]:
        try:
            if service_name == GCP_SERVICE:
                if not isinstance(config, GcpConfig):
                    raise TypeError("Invalid configuration type for GCP service")
                return GcpSummaryService(config=config)
            # elif service_name == COHERE_SERVICE:
            #     if not isinstance(config, CohereSummaryServiceConfig):
            #         raise TypeError("Invalid configuration type for Cohere service")
            #     return CohereSummaryService(config=config)
            # elif service_name == TITAN_SERVICE:
            #     if not isinstance(config, TitanSummaryServiceConfig):
            #         raise TypeError("Invalid configuration type for Titan service")
            #     return TitanSummaryService(config=config)
            else:
                raise ValueError("Unknown service name: %s", service_name)
        except ValidationError as e:
            raise ValueError("Invalid configuration parameters: %s", e) from None
