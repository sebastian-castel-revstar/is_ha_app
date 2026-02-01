import base64
import gzip
import json
import os
import uuid
from datetime import datetime
from typing import Literal

import boto3
import urllib3
from api_utils.arize_tracing import otel_trace, register
from aws_lambda_powertools import Logger, Tracer
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel, Field, ValidationError, confloat, validator

logger = Logger(log_uncaught_exceptions=True)
tracer = Tracer()
register(project_name="IntentSearch", auto_instrument_bedrock=False)
otel_tracer = trace.get_tracer(__name__)

# Initialize the SageMaker runtime client
sagemaker_runtime = boto3.client(
    "sagemaker-runtime", region_name=boto3.Session().region_name
)

# Define the SageMaker endpoint names
environment = os.environ["env"]
toxicity_endpoint = os.environ["toxicity_endpoint_name"]
toxicity_endpoint = f"{toxicity_endpoint}-{environment}"

kinesis_firehose_toxicity_request_log = os.environ[
    "kinesis_firehose_toxicity_request_log"
]
kinesis_firehose_toxicity_response_log = os.environ[
    "kinesis_firehose_toxicity_response_log"
]

# aws session token created by lambda
sessions_token = os.environ["AWS_SESSION_TOKEN"]
# secrets port
secrets_extension_http_port = os.environ["PARAMETERS_SECRETS_EXTENSION_HTTP_PORT"]
http = urllib3.PoolManager()


@tracer.capture_method
def get_secret_dict(secret_arn):
    url = f"http://localhost:{secrets_extension_http_port}/secretsmanager/get?secretId={secret_arn}"
    headers = {"X-Aws-Parameters-Secrets-Token": sessions_token}
    response = http.request("GET", url, headers=headers)

    return json.loads(json.loads(response.data)["SecretString"])


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


@tracer.capture_method
def invoke_endpoint(endpoint_name, payload_json):
    """
    Function to invoke a SageMaker endpoint.
    """
    logger.info("Invoking endpoint: %s with payload: %s", endpoint_name, payload_json)

    with otel_tracer.start_as_current_span(name="toxicity_classifier") as span:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload_json,
        )
        result_value = response["Body"].read().decode()
        result = json.loads(result_value)

        span.set_attributes(
            {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: "ML_MODEL",
                SpanAttributes.INPUT_VALUE: json.dumps(payload_json),
                SpanAttributes.OUTPUT_VALUE: result_value,
            }
        )
        span.set_status(trace.Status(trace.StatusCode.OK))

    logger.info("Received response from %s: %s", endpoint_name, result)

    return result


@tracer.capture_method
def log_request(toxicity_request_log):
    records = []
    payload = json.dumps(toxicity_request_log)

    if os.getenv("enable_firehose_compression") and (len(payload) > 200_000):
        payload = gzip.compress(payload.encode("utf-8"))
        payload = base64.b64encode(payload).decode("ascii")
        if len(payload) > 256_000:  # cloudwatch limit
            logger.warning(
                "firehose log too large for single record. Will spill into multiple records"
            )

    records.append({"Data": payload})

    # Push to firehose via ASYNC Logging Extension
    if len(toxicity_request_log.keys()) > 0:
        event = {
            "firehose_name": kinesis_firehose_toxicity_request_log,
            "payload": records,
        }
        print("KINESIS_LOG_EVENT:{}".format(json.dumps(event)), flush=True)
    else:
        logger.error("No Context Log")


@tracer.capture_method
def log_response(toxicity_response_log):
    records = []
    records.append({"Data": json.dumps(toxicity_response_log)})

    # Batch-write to firehose via ASYNC Logging Extension
    if len(records) > 0:
        event = {
            "firehose_name": kinesis_firehose_toxicity_response_log,
            "payload": records,
        }
        print("KINESIS_LOG_EVENT:{}".format(json.dumps(event)), flush=True)
    else:
        logger.info("No query classfier results returned")


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@otel_trace(span_kind="TOOL", span_name="toxicity_classifier_handler", auto_flush=True)
def lambda_handler(event, context):
    """
    Lambda function to invoke two SageMaker endpoints asynchronously.
    """
    search_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    try:
        # Validate the event using Pydantic
        Request(**event)

    except ValidationError as e:
        logger.exception("Request validation error: %s", e)
        raise ValueError("INTENT_SEARCH_API_ERROR_400") from None

    try:
        toxicity_request_log = {
            "toxicty_query_log_id": str(uuid.uuid4()),
            "region_name": event["region_name"],
            "query": event["text"],
            "search_ts": search_ts,
        }
        log_request(toxicity_request_log)

        Input = {"search_context": {"query": event["text"]}}
        payload_json = json.dumps(Input)
        toxicity_result = invoke_endpoint(toxicity_endpoint, payload_json)

        # Validate api response using pydantic
        Response(**toxicity_result)

        toxicity_response_log = {
            "toxicty_query_log_id": str(uuid.uuid4()),
            "toxic": toxicity_result["toxicity"].get("toxic"),
            "severe_toxic": toxicity_result["toxicity"].get("severe_toxic"),
            "obscene": toxicity_result["toxicity"].get("obscene"),
            "threat": toxicity_result["toxicity"].get("threat"),
            "insult": toxicity_result["toxicity"].get("insult"),
            "identity_hate": toxicity_result["toxicity"].get("identity_hate"),
            "search_ts": search_ts,
        }
        log_response(toxicity_response_log)

        logger.info("API response: %s", toxicity_result)
        return toxicity_result

    except (ValidationError, Exception) as err:
        logger.exception("lambda_handler has failed: Runtime Error")
        raise ValueError("TOXICITY_API_ERROR_500") from err
