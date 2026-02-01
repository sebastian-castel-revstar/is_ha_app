# HOTFIX removed arize
# removed dependencies
# arize-otel==0.7.3
# openinference-instrumentation-bedrock==0.1.12
# opentelemetry-exporter-otlp==1.29.0

import base64
import concurrent.futures
import gzip
import json
import os
import random
import threading
import time
import uuid
from datetime import date, datetime
from typing import Literal, Optional

import boto3
import requests
import urllib3
from amazondax import AmazonDaxClient

# from api_utils.arize_tracing import otel_trace, register
from api_utils.human_review_config import QCConfig, ToxConfig
from api_utils.human_review_utils import get_lambda_tags, post_to_labeling_job
from api_utils.toxicity_checks import toxicity_blacklist, toxicity_whitelist
from api_utils.trigger_words import qc_contains_trigger_words
from aws_lambda_powertools import Logger
from boto3.dynamodb.conditions import Key
from cachetools import TTLCache, cached
from ddtrace import tracer

# from openinference.semconv.trace import SpanAttributes
# from opentelemetry import trace
# from opentelemetry.context import attach, detach
# from opentelemetry.context import get_current as get_current_context
from pydantic import BaseModel, Field, ValidationError, confloat, validator

logger = Logger(log_uncaught_exceptions=True)

# register(project_name="IntentSearch", auto_instrument_bedrock=False)
# otel_tracer = trace.get_tracer(__name__)

# Creating a TTL cache
# 4500 max sixe to maximize item caching for 300 seconds based on load capacity 15qps
property_metadata_cache = TTLCache(maxsize=4500, ttl=300)

# Initialize AWS clients
sagemaker = boto3.client(
    service_name="sagemaker", region_name=boto3.Session().region_name
)

sagemaker_runtime = boto3.client(
    "sagemaker-runtime", region_name=boto3.Session().region_name
)

sns = boto3.client(service_name="sns", region_name=boto3.Session().region_name)

lambda_client = boto3.client(
    service_name="lambda",
    region_name=boto3.Session().region_name,
)

# Initialize Dynamodb client
dynamodb_client = boto3.client(
    service_name="dynamodb", region_name=boto3.Session().region_name
)

# Define the SageMaker endpoint names
environment = os.environ["env"]
query_clf_endpoint = os.environ["query_classifier_endpoint_name"]
query_clf_endpoint = f"{query_clf_endpoint}-{environment}"
toxicity_endpoint = os.environ["toxicity_endpoint_name"]
toxicity_endpoint = f"{toxicity_endpoint}-{environment}"
api_key_secret_arn = os.environ["qc_api_key_secret_arn"].replace(
    "ENV", environment.upper()
)
kinesis_firehose_request_log = os.environ["kinesis_firehose_context_log"]
kinesis_firehose_response_query_log = os.environ["kinesis_firehose_classification_log"]
kinesis_firehose_response_toxicity_log = os.environ["kinesis_firehose_toxicity_log"]

# stage replication constants
SHADOWTESTING_URL = "https://vpce-0d16b9a55a46d203b-jz9cogr0.execute-api.us-east-1.vpce.amazonaws.com/staging/"
SHADOWTESTING_HOST = "m3v9ig309b.execute-api.us-east-1.amazonaws.com"

# dynamodb and dax for intent cache
dax_endpoint = os.environ["dax_cluster_endpoint"]
property_dynamo_table_name = os.environ["dynamo_property_table_name"]
dax = AmazonDaxClient.resource(endpoint_url=dax_endpoint)
dynamodb = boto3.resource("dynamodb", region_name=boto3.Session().region_name)
property_dynamo_table = dax.Table(property_dynamo_table_name)

# human review
qcapi_human_review_confidence_threshold = float(
    os.environ["qcapi_human_review_confidence_threshold"]
)
toxapi_human_review_replication_rate = float(
    os.environ["toxapi_human_review_replication_rate"]
)
qcapi_hr_config = QCConfig.get_config()
toxapi_hr_config = ToxConfig.get_config()
qcapi_sns_topic_arn = os.environ["qcapi_sns_topic_arn"]
toxapi_sns_topic_arn = os.environ["toxapi_sns_topic_arn"]


# aws session token created by lambda
sessions_token = os.environ["AWS_SESSION_TOKEN"]
# secrets port
secrets_extension_http_port = os.environ["PARAMETERS_SECRETS_EXTENSION_HTTP_PORT"]
http = urllib3.PoolManager()

# Dynamodb Table for logging Intent
dynamodb_intent_label_table = os.environ["dynamodb_intent_label_table"]

# QC threshold
qc_destination_threshold = float(os.environ["qc_destination_threshold"])


@tracer.wrap()
def get_secret_dict(secret_arn):
    url = f"http://localhost:{secrets_extension_http_port}/secretsmanager/get?secretId={secret_arn}"
    headers = {"X-Aws-Parameters-Secrets-Token": sessions_token}
    response = http.request("GET", url, headers=headers)

    return json.loads(json.loads(response.data)["SecretString"])


@tracer.wrap()
def log_request(context_log, local_ts, search_ts):  # pylint: disable=C0116
    # Get firehose name from config

    firehose_target = kinesis_firehose_request_log

    records = []

    context_log_norm = {
        "search_id": context_log["search_id"],
        "visitor_id": context_log["visitor_id"],
        "member_id": context_log["member_id"],
        "channel": context_log["channel"],
        "local_ts": local_ts,
        "country_name": context_log["session_context"].get("country_name"),
        "region_name": context_log["session_context"].get("region_name"),
        "city_name": context_log["session_context"].get("city_name"),
        "user_agent": context_log["session_context"].get("user_agent"),
        "query": context_log["search_context"].get("query"),
        "arrival_date": context_log["search_context"].get("arrival_date"),
        "departure_date": context_log["search_context"].get("departure_date"),
        "num_rooms": context_log["search_context"].get("num_rooms"),
        "accessible": context_log["search_context"].get("accessible"),
        "num_adults": context_log["search_context"].get("num_adults"),
        "num_children": context_log["search_context"].get("num_children"),
        "special_rate_code": context_log["search_context"].get("special_rate_code"),
        "special_rate_category": context_log["search_context"].get(
            "special_rate_category"
        ),
        "use_points": context_log["search_context"].get("use_points"),
        "search_ts": search_ts,
    }
    payload = json.dumps(context_log_norm)

    if os.getenv("enable_firehose_compression") and (len(payload) > 200_000):
        payload = gzip.compress(payload.encode("utf-8"))
        payload = base64.b64encode(payload).decode("ascii")
        if len(payload) > 256_000:  # cloudwatch limit
            logger.warning(
                "firehose log too large for single record. Will spill into multiple records"
            )

    records.append({"Data": payload})

    # Push to firehose via ASYNC Logging Extension
    if len(context_log_norm.keys()) > 0:
        event = {"firehose_name": firehose_target, "payload": records}
        print("KINESIS_LOG_EVENT:{}".format(json.dumps(event)), flush=True)
    else:
        logger.error("No Context Log")


@tracer.wrap()
def log_response(result, firehose_target):  # pylint: disable=C0116
    # Explode to individual records
    records = []
    records.append({"Data": json.dumps(result)})

    # Batch-write to firehose via ASYNC Logging Extension
    if len(records) > 0:
        event = {"firehose_name": firehose_target, "payload": records}
        print("KINESIS_LOG_EVENT:{}".format(json.dumps(event)), flush=True)
    else:
        logger.info("No query classfier results returned")


# Shadow testing
def replicate_to_staging(event, shadowtesting_api_key):  # pylint: disable=C0116
    headers = {
        "Host": SHADOWTESTING_HOST,
        "x-api-key": shadowtesting_api_key,
        "Content-Type": "application/json",
    }
    url = SHADOWTESTING_URL

    logger.info("Replicating event to staging: %s", event)

    requests.post(url, data=json.dumps(event), headers=headers)


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


class ToxicityScores(BaseModel):
    toxic: confloat(ge=0, le=1)
    severe_toxic: confloat(ge=0, le=1)
    obscene: confloat(ge=0, le=1)
    threat: confloat(ge=0, le=1)
    insult: confloat(ge=0, le=1)
    identity_hate: confloat(ge=0, le=1)


class Response(BaseModel):
    label: Literal["INTENT", "DESTINATION", "IRRELEVANT"]
    probability: confloat(ge=0, le=1)
    toxicity: ToxicityScores


@tracer.wrap()
# def invoke_endpoint(endpoint_name, payload_json, span_name, otel_context):
def invoke_endpoint(endpoint_name, payload_json, span_name):
    """
    Function to invoke a SageMaker endpoint.
    """
    logger.info("Invoking endpoint: %s with payload: %s", endpoint_name, payload_json)
    # token = attach(otel_context) if otel_context else None

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload_json,
    )

    result_value = response["Body"].read().decode()
    result = json.loads(result_value)

    # with otel_tracer.start_as_current_span(name=span_name) as span:
    #    response = sagemaker_runtime.invoke_endpoint(
    #        EndpointName=endpoint_name,
    #        ContentType="application/json",
    #        Body=payload_json,
    #    )
    #    result_value = response["Body"].read().decode()
    #    result = json.loads(result_value)
    #    span.set_attributes(
    #        {
    #            SpanAttributes.OPENINFERENCE_SPAN_KIND: "ML_MODEL",
    #            SpanAttributes.INPUT_VALUE: json.dumps(payload_json),
    #            SpanAttributes.OUTPUT_VALUE: result_value,
    #        }
    #    )
    #    span.set_status(trace.Status(trace.StatusCode.OK))

    logger.info("Received response from %s: %s", endpoint_name, result)
    # detach(token) if token is not None else None
    return result


@cached(property_metadata_cache)
@tracer.wrap()
def query_check_dynamodb(query):
    try:
        query = query.lower().strip()

        indices = [
            {"index": "property_nm_index", "key": "property_nm"},
            {"index": "city_nm_index", "key": "city_nm"},
        ]

        response = None

        for index in indices:
            response = property_dynamo_table.query(
                IndexName=index["index"],
                KeyConditionExpression=Key(index["key"]).eq(query),
                ProjectionExpression="spirit_cd",
            )

            if response.get("Items"):
                break

        if response and response.get("Items"):
            return {
                "label": "DESTINATION",
                "probability": 1,
                "toxicity": {
                    "toxic": 0,
                    "severe_toxic": 0,
                    "obscene": 0,
                    "threat": 0,
                    "insult": 0,
                    "identity_hate": 0,
                },
            }

    except Exception:
        logger.exception("Failed to scan DynamoDB table")
        return None


@logger.inject_lambda_context(log_event=True)
@tracer.wrap()
# @otel_trace(span_kind="TOOL", span_name="query_classifier_handler", auto_flush=True)
def lambda_handler(event, context):
    """
    Lambda function to invoke two SageMaker endpoints asynchronously.
    """

    search_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if event["session_context"]["local_ts"]:
        local_ts = datetime.utcfromtimestamp(
            float(event["session_context"]["local_ts"]) / 1000.0
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    else:
        local_ts = event["session_context"]["local_ts"]

    try:
        # Validate the event using Pydantic
        Request(**event)
        log_request(event, local_ts, search_ts)

    except ValidationError as e:
        logger.exception("Request validation error: %s", e)
        raise ValueError("INTENT_SEARCH_API_ERROR_400") from None

    # trace.get_current_span().set_attributes(
    #     {
    #         "hyatt.intent.search_id": event["search_id"],
    #         "hyatt.intent.channel": event["channel"],
    #         "user.id": event["visitor_id"],
    #     }
    # )
    logger.append_keys(search_id=event["search_id"])

    try:
        stg_repl_rate = float(os.environ["staging_replication_rate"])
        if environment == "prod":
            shadowtesting_api_key = get_secret_dict(api_key_secret_arn)["key"]

            rand = random.random()

            # Sampled replication
            if rand <= stg_repl_rate:
                repl_event = event.copy()
                logger.info("Sending replicated event to Staging")

                # Asynchronous call (dont join/wait for response)
                thread = threading.Thread(
                    target=replicate_to_staging,
                    args=(
                        repl_event,
                        shadowtesting_api_key,
                    ),
                )
                thread.start()

    except:
        # Log error but dont fail the API
        logger.exception("Staging replication has failed: Runtime Error")

    try:
        random_sampling = random.random()
        if random_sampling <= toxapi_human_review_replication_rate:
            post_to_labeling_job(
                sagemaker=sagemaker,
                sns=sns,
                query=event["search_context"]["query"],
                sns_topic_arn=toxapi_sns_topic_arn,
                tags=get_lambda_tags(
                    lambda_client=lambda_client,
                    lambda_arn=f"{context.invoked_function_arn.split(':function:')[0]}:function:{context.function_name}",
                ),
                labeling_job_configuration=toxapi_hr_config,
            )
    except:
        logger.exception("Human Review replication has failed: Runtime Error")

    try:
        query_string = event["search_context"]["query"]

        query_clf_result = None
        toxicity_result = None

        # check if the query is either a city or a property matching with existing values in dynamodb
        query_clf_result = query_check_dynamodb(query_string)

        payload_json = json.dumps(event)
        # otel_context = get_current_context()

        if query_clf_result is None:
            # if query contains trigger words - classify as INTENT
            if qc_contains_trigger_words(query_string):
                query_clf_result = {"label": "INTENT", "probability": 1}

            # checks if query matches toxicity white or black lists
            toxicity_scores = None
            if toxicity_blacklist(query_string):
                toxicity_scores = ToxicityScores(
                    toxic=1,
                    severe_toxic=0,
                    obscene=0,
                    threat=0,
                    insult=0,
                    identity_hate=0,
                )
            elif toxicity_whitelist(query_string):
                toxicity_scores = ToxicityScores(
                    toxic=0,
                    severe_toxic=0,
                    obscene=0,
                    threat=0,
                    insult=0,
                    identity_hate=0,
                )

            if toxicity_scores:
                toxicity_result = {"toxicity": toxicity_scores.dict()}

            if query_clf_result is None and toxicity_result is None:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit both tasks to the executor
                    future_1 = executor.submit(
                        invoke_endpoint,
                        query_clf_endpoint,
                        payload_json,
                        "query_classifier",
                        # otel_context,
                    )
                    future_2 = executor.submit(
                        invoke_endpoint,
                        toxicity_endpoint,
                        payload_json,
                        "toxicity_classifier",
                        # otel_context,
                    )

                    # Wait for both futures to complete
                    query_clf_result = future_1.result()
                    toxicity_result = future_2.result()
            elif query_clf_result is None:
                query_clf_result = invoke_endpoint(
                    query_clf_endpoint,
                    payload_json,
                    "query_classifier",
                    # otel_context,
                )
            elif toxicity_result is None:
                toxicity_result = invoke_endpoint(
                    toxicity_endpoint,
                    payload_json,
                    "toxicity_classifier",
                    # otel_context,
                )

            query_clf_result.update(toxicity_result)

        Response(**query_clf_result)

        # Set threshold - label as DESTINATION only if probability>=threshold
        if (
            query_clf_result["label"] == "DESTINATION"
            and query_clf_result["probability"] < qc_destination_threshold
        ):
            query_clf_result["label"] = "INTENT"
            query_clf_result["probability"] = 1 - query_clf_result["probability"]

        # Log intent
        intent_log = {
            "search_classification_log_id": str(uuid.uuid4()),
            "search_id": event["search_id"],
            "label": query_clf_result.get("label"),
            "probability": query_clf_result.get("probability"),
            "search_ts": search_ts,
        }
        log_response(intent_log, kinesis_firehose_response_query_log)

        # Computing TTL (current time + 300 seconds for 5 minutes)
        expiration_ts = int(time.time()) + 300

        # Creating a DynamoDB compatible dictionary for intent_log
        dynamo_intent_log = {
            "search_classification_log_id": {"S": str(uuid.uuid4())},
            "search_id": {"S": event["search_id"]},
            "label": {"S": query_clf_result.get("label")},
            "probability": {"N": str(query_clf_result.get("probability"))},
            "search_ts": {"S": search_ts},
            "expiration_ts": {"N": str(expiration_ts)},
        }
        # Logging intent on dynamodb
        dynamodb_client.put_item(
            TableName=dynamodb_intent_label_table, Item=dynamo_intent_log
        )

        if (
            float(query_clf_result.get("probability"))
            <= qcapi_human_review_confidence_threshold
        ):
            post_to_labeling_job(
                sagemaker=sagemaker,
                sns=sns,
                query=event["search_context"]["query"],
                sns_topic_arn=qcapi_sns_topic_arn,
                tags=get_lambda_tags(
                    lambda_client=lambda_client,
                    lambda_arn=f"{context.invoked_function_arn.split(':function:')[0]}:function:{context.function_name}",
                ),
                labeling_job_configuration=qcapi_hr_config,
            )

        # Log toxicity
        toxicity_log = {
            "query_toxicity_log_id": str(uuid.uuid4()),
            "search_id": event["search_id"],
            "toxic": query_clf_result["toxicity"].get("toxic"),
            "severe_toxic": query_clf_result["toxicity"].get("severe_toxic"),
            "obscene": query_clf_result["toxicity"].get("obscene"),
            "threat": query_clf_result["toxicity"].get("threat"),
            "insult": query_clf_result["toxicity"].get("insult"),
            "identity_hate": query_clf_result["toxicity"].get("identity_hate"),
            "search_ts": search_ts,
        }
        log_response(toxicity_log, kinesis_firehose_response_toxicity_log)

        logger.info("API response: %s", query_clf_result)
        return query_clf_result

    except (ValidationError, Exception) as err:
        logger.exception("lambda_handler has failed: Runtime Error")
        raise ValueError("INTENT_SEARCH_API_ERROR_500") from err
