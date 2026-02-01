"""
This file contains the constants used by the summary services.
"""

# general constants
AWS_DEFAULT_REGION = "us-east-1"

# service names supported
TITAN_SERVICE = "titan"
GCP_SERVICE = "gcp"
COHERE_SERVICE = "cohere"

# titan constants
TITAN_MODEL_ID_TEXT_LITE_V2 = "amazon.titan-text-lite-v1"
TITAN_MODEL_ID_TEXT_EXPRESS_V1 = "amazon.titan-text-express-v1"
# TITAN_MODEL_ID_TEXT_PREMIER_V1 = "amazon.titan-text-premier-v1:0"

# cohere constants
COHERE_MODEL_ID_TEXT_v14 = "cohere.command-text-v14"
COHERE_MODEL_ID_LIGHT_TEXT_v14 = "cohere.command-light-text-v14"
# COHERE_MODEL_ID_TEXT_PLUS_v1 = "cohere.command-r-plus-v1:0"

# gcp constants
GCP_MODEL_ID_GEMINI_FLASH_2_0 = "gemini-2.0-flash-001"
