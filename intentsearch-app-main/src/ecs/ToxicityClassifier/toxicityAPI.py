import json
import os
import threading
import uuid
from datetime import datetime
from functools import lru_cache

import boto3
import requests
from api_utils.arize_tracing import otel_trace, register
from aws_lambda_powertools import Logger
from ddtrace import tracer
from models import Response
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from pydantic import ValidationError

logger = Logger(log_uncaught_exceptions=True)
logger.append_keys(
    source="ecs", hostname="api", service=os.getenv("DD_SERVICE"), env=os.getenv("env")
)

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

secret_client = boto3.client("secretsmanager")


@tracer.wrap()
@lru_cache(maxsize=32)
def get_secret_dict(secret_arn):
    response = secret_client.get_secret_value(SecretId=secret_arn)
    return json.loads(response["SecretString"])


@tracer.wrap()
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


def post_request(url, data):
    if isinstance(data, dict):
        data = json.dumps(data)
    try:
        response = requests.post(
            url,
            data=data,
        )
        response.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
        logger.exception("post request to %s failed", url)


@tracer.wrap()
def log_request(toxicity_request_log):
    records = []
    payload = json.dumps(toxicity_request_log)

    records.append({"Data": payload})

    # Push to firehose via ASYNC Logging Extension
    if len(toxicity_request_log.keys()) > 0:
        threading.Thread(
            target=post_request,
            kwargs={
                "url": "http://localhost:8800/firehose",
                "data": {
                    "firehose_name": kinesis_firehose_toxicity_request_log,
                    "payload": records,
                },
            },
        ).start()
    else:
        logger.error("No Context Log")


@tracer.wrap()
def log_response(toxicity_response_log):
    records = []
    records.append({"Data": json.dumps(toxicity_response_log)})

    # Batch-write to firehose via ASYNC Logging Extension
    if len(records) > 0:
        threading.Thread(
            target=post_request,
            kwargs={
                "url": "http://localhost:8800/firehose",
                "data": {
                    "firehose_name": kinesis_firehose_toxicity_response_log,
                    "payload": records,
                },
            },
        ).start()
    else:
        logger.info("No query classfier results returned")


@tracer.wrap()
@otel_trace(span_kind="TOOL", span_name="toxicity_classifier_handler", auto_flush=True)
def lambda_handler(event):
    """
    Lambda function to invoke two SageMaker endpoints asynchronously.
    """
    search_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

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
