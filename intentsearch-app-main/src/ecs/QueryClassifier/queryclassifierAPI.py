# HOTFIX removed arize
# removed dependencies
# arize-otel==0.7.3
# openinference-instrumentation-bedrock==0.1.12
# opentelemetry-exporter-otlp==1.29.0

import concurrent.futures
import json
import os
import random
import threading
import time
import uuid
from datetime import datetime

import boto3
import requests
from amazondax import AmazonDaxClient

# from api_utils.arize_tracing import otel_trace, register
from api_utils.human_review_config import QCConfig, ToxConfig
from api_utils.human_review_utils import get_ecs_tags, post_to_labeling_job
from api_utils.toxicity_checks import toxicity_blacklist, toxicity_whitelist
from api_utils.trigger_words import qc_contains_trigger_words
from apiutils.cognito_auth import CognitoAuthenticator  # hy-api-utilities
from aws_lambda_powertools import Logger
from boto3.dynamodb.conditions import Key
from cachetools import LRUCache, TTLCache, cached
from ddtrace import tracer
from models import APIResponse, ToxicityScores

# from openinference.semconv.trace import SpanAttributes
# from opentelemetry import trace
# from opentelemetry.context import attach, detach
# from opentelemetry.context import get_current as get_current_context

logger = Logger(log_uncaught_exceptions=True)
logger.append_keys(
    source="ecs", hostname="api", service=os.getenv("DD_SERVICE"), env=os.getenv("env")
)
try:
    # Acquisition of access_tokens for staging replication
    cognito_auth = CognitoAuthenticator(
        credentials_secret_name=os.getenv("COGNITO_STG_AUTH_SECRET_ARN"),
        cognito_domain=os.getenv("COGNITO_STG_DOMAIN"),
        scope=os.getenv("COGNITO_CLIENT_SCOPE"),
    )
except:  # noqa: E722
    logger.error("Could not initialize staging authenticator")

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

ecs_client = boto3.client(
    service_name="ecs",
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

# Dynamodb Table for logging Intent
dynamodb_intent_label_table = os.environ["dynamodb_intent_label_table"]

# QC threshold
qc_destination_threshold = float(os.environ["qc_destination_threshold"])

secret_client = boto3.client("secretsmanager")


@tracer.wrap()
@cached(LRUCache(maxsize=32))
def get_secret_dict(secret_arn):
    response = secret_client.get_secret_value(SecretId=secret_arn)
    return json.loads(response["SecretString"])


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

    records.append({"Data": payload})

    # Push to firehose via ASYNC Logging Extension
    if len(context_log_norm.keys()) > 0:
        threading.Thread(
            target=post_request,
            kwargs={
                "url": "http://localhost:8800/firehose",
                "data": {
                    "firehose_name": firehose_target,
                    "payload": records,
                },
            },
        ).start()
    else:
        logger.error("No Context Log")


@tracer.wrap()
def log_response(result, firehose_target):  # pylint: disable=C0116
    # Explode to individual records
    records = []
    records.append({"Data": json.dumps(result)})

    # Batch-write to firehose via ASYNC Logging Extension
    if len(records) > 0:
        threading.Thread(
            target=post_request,
            kwargs={
                "url": "http://localhost:8800/firehose",
                "data": {
                    "firehose_name": firehose_target,
                    "payload": records,
                },
            },
        ).start()
    else:
        logger.info("No query classfier results returned")


def replicate_to_staging(event, max_retries=3):  # pylint: disable=C0116
    attempt = 0
    backoff = 0.1
    while attempt < max_retries:
        response = requests.post(
            os.getenv("APPLICATION_DOMAIN_NAME").replace("analytics", "analyticsdev")
            + "/query-classifier",
            data=json.dumps(event),
            headers={"Authorization": f"Bearer {cognito_auth.access_token}"},
        )
        if response.status_code == 200:
            break
        elif response.status_code == 401:
            cognito_auth.renew_token()
        else:
            logger.exception(
                "Could not replicate to staging environment %s", response.content
            )

        attempt += 1
        time.sleep(backoff)
        backoff *= 5


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


@tracer.wrap()
# @otel_trace(span_kind="TOOL", span_name="query_classifier_handler", auto_flush=True)
def lambda_handler(event):
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

    log_request(event, local_ts, search_ts)

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
            rand = random.random()

            # Sampled replication
            if rand <= stg_repl_rate:
                repl_event = event.copy()
                logger.info("Sending replicated event to Staging")

                # Asynchronous call (dont join/wait for response)
                thread = threading.Thread(
                    target=replicate_to_staging,
                    args=(repl_event,),
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
                tags=get_ecs_tags(
                    ecs_client=ecs_client,
                    # TODO
                    # lambda_arn=f"{context.invoked_function_arn.split(':function:')[0]}:function:{context.function_name}",
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

        APIResponse(**query_clf_result)

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
                tags=get_ecs_tags(ecs_client=ecs_client),
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

    except Exception as err:
        logger.exception("lambda_handler has failed: Runtime Error")
        raise ValueError("INTENT_SEARCH_API_ERROR_500") from err
