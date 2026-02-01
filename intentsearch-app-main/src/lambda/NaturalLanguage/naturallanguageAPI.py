# HOTFIX removed arize
# removed dependencies
# arize-otel==0.7.3
# openinference-instrumentation-bedrock==0.1.12
# opentelemetry-exporter-otlp==1.29.0
# opentelemetry-propagator-aws-xray==1.0.2

import concurrent.futures
import hashlib
import json
import os
import random
import threading
import uuid
from datetime import date, datetime
from typing import List, Literal, Optional

import boto3
import ldclient
import requests
import urllib3
from amazondax import AmazonDaxClient

# from api_utils.arize_tracing import otel_trace, register
from api_utils.human_review_config import NLConfig
from api_utils.human_review_utils import get_lambda_tags, post_to_labeling_job
from api_utils.trigger_words import enrich_ner_result_with_trigger_entities
from aws_lambda_powertools import Logger
from cachetools import TTLCache, cached
from ddtrace import tracer
from GCPGemini.GCPService.Constants import (
    GCP_MODEL_ID_GEMINI_FLASH_2_0,
)
from GCPGemini.GCPService.FactoryService import FactoryService
from GCPGemini.GCPService.GcpConfig import GcpConfig
from ldclient import Context
from ldclient.config import Config

# from openinference.semconv.trace import SpanAttributes
# from opentelemetry import trace
# from opentelemetry.context import attach, detach
# from opentelemetry.context import get_current as get_current_context
from pydantic import BaseModel, Field, ValidationError, validator

logger = Logger(log_uncaught_exceptions=True)

# register(project_name="IntentSearch", auto_instrument_bedrock=True)
# otel_tracer = trace.get_tracer(__name__)

# Creating a TTL cache
prompt_cache = TTLCache(maxsize=5, ttl=300)

# Initialize the boto3 clients
sagemaker = boto3.client(
    service_name="sagemaker",
    region_name=boto3.Session().region_name,
)

sagemaker_runtime = boto3.client(
    "sagemaker-runtime", region_name=boto3.Session().region_name
)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=boto3.Session().region_name,
)

bedrock_agent = boto3.client(
    service_name="bedrock-agent",
    region_name=boto3.Session().region_name,
)

sns = boto3.client(
    service_name="sns",
    region_name=boto3.Session().region_name,
)

lambda_client = boto3.client(
    service_name="lambda",
    region_name=boto3.Session().region_name,
)

s3_client = boto3.client(
    service_name="s3",
    region_name=boto3.Session().region_name,
)

# Get environment variables
environment = os.environ["env"]

# replication rate
stg_repl_rate = float(os.environ["staging_replication_rate"])

# endpoints
ner_endpoint = os.environ["ner_endpoint_name"]
ner_endpoint = f"{ner_endpoint}-{environment}"
# shadowtesting api key
api_key_secret_arn = os.environ["ner_api_key_secret_arn"]

# embedding
ner_embedding_source = os.environ["ner_embedding_source"]
ner_embedding_model_id = os.environ["ner_embedding_model_id"]

# feature flag
launchdarkly_api_key_secret_arn = os.environ["launchdarkly_api_key_secret_arn"]
bedrock_ld_feature_flag_key = os.environ["bedrock_ld_feature_flag_key"]

# firehose
kinesis_firehose_response_price_log = os.environ["kinesis_firehose_price_log"]
kinesis_firehose_response_ner_entity_log = os.environ["kinesis_firehose_ner_entity_log"]
kinesis_firehose_response_llm_entity_log = os.environ["kinesis_firehose_llm_entity_log"]
kinesis_feature_flag_log = os.environ["kinesis_firehose_feature_flag_log"]

# dynamodb and dax for intent cache
dax = AmazonDaxClient.resource(endpoint_url=os.environ["dax_cache"])
dynamodb = boto3.resource("dynamodb", region_name=boto3.Session().region_name)

gemini_cache_dynamo_table = dynamodb.Table(os.environ["gemini_cache_dynamo_table"])
gemini_cache_dax_table = dax.Table(os.environ["gemini_cache_dynamo_table"])

# gcp
gcp_secret_arn = os.environ["gcp_secret_arn"]
gcp_location = os.environ["gcp_location"]
gemini_model_id = os.environ.get("gemini_model_id", GCP_MODEL_ID_GEMINI_FLASH_2_0)
if environment == "staging":
    gcp_project = f"{os.environ['gcp_project']}-dev"
else:
    gcp_project = f"{os.environ['gcp_project']}-prod"

cache_path = "/tmp/gcp_sa.json"
if os.path.exists(cache_path):
    with open(cache_path, "r") as f:
        cached_gcp_cred = json.load(f)
else:
    cached_gcp_cred = None

# gemini prompt arn
intent_gemini_prompt_arn = os.environ["intent_gemini_prompt_arn"]

# human review
nlapi_hr_config = NLConfig.get_config()
hr_gt_repl_rate = float(os.environ["human_review_replication_rate"])
nlapi_sns_topic_arn = os.environ["nlapi_sns_topic_arn"]

# get known experience tags [INTENT CACHE]
entity_s3_bucket = os.environ["entity_s3_bucket"]
s3_obj = s3_client.get_object(
    Bucket=os.environ["entity_s3_bucket"],
    Key=f"""{os.environ["entity_s3_path"]}/experiences_list.json""",
)
known_experience_tags = set(
    json.loads(s3_obj["Body"].read().decode("utf-8")).get("experiences", [])
)


# Using a decorator to cache prompt text
@cached(prompt_cache)
def get_bedrock_prompt(prompt_arn):
    # Get the prompt text from bedrock prompt
    bedrock_response = bedrock_agent.get_prompt(promptIdentifier=prompt_arn)
    return bedrock_response["variants"][0]["templateConfiguration"]["text"]["text"]


# prompt for gemini calls
gemini_prompt = get_bedrock_prompt(intent_gemini_prompt_arn)

# aws session token created by lambda
sessions_token = os.environ["AWS_SESSION_TOKEN"]
# secrets port
secrets_extension_http_port = os.environ["PARAMETERS_SECRETS_EXTENSION_HTTP_PORT"]
http = urllib3.PoolManager()


@tracer.wrap()
def get_secret_dict(secret_arn):
    url = f"http://localhost:{secrets_extension_http_port}/secretsmanager/get?secretId={secret_arn}"
    headers = {"X-Aws-Parameters-Secrets-Token": sessions_token}
    response = http.request("GET", url, headers=headers)

    return json.loads(json.loads(response.data)["SecretString"])


def generate_flag(
    ld_client, feature_flag_name, context_id, search_id, visitor_id, search_ts, default
):
    feature_flag_value = ld_client.variation(
        key=feature_flag_name, context=context_id, default=default
    )
    # log feature flag
    feature_flag_log = {
        "search_id": search_id,
        "visitor_id": visitor_id,
        "search_ts": search_ts,
        "feature_flag_name": feature_flag_name,
        "feature_flag_value": str(feature_flag_value),
    }
    log_result(feature_flag_log, kinesis_feature_flag_log)

    return feature_flag_value


@tracer.wrap()
def log_result(result, firehose_target):  # pylint: disable=C0116
    # Explode to individual records
    records = []
    records.append({"Data": json.dumps(result)})

    # Batch-write to firehose via ASYNC Logging Extension
    if len(records) > 0:
        event = {"firehose_name": firehose_target, "payload": records}
        print("KINESIS_LOG_EVENT:{}".format(json.dumps(event)), flush=True)
    else:
        logger.info("No logging results returned")


@tracer.wrap()
def load_currency_mapping():
    """Load currency mapping from a JSON file."""
    try:
        with open("currency_mapping.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.exception("Error loading currency mapping")
        raise ValueError("INTENT_SEARCH_API_ERROR_500") from None


# Load currency mapping from the JSON file
currency_mapping = load_currency_mapping()

# Reverse the mapping to look up ISO codes by symbols and names
reverse_currency_mapping = {
    item.lower(): iso_code
    for iso_code, items in currency_mapping.items()
    for item in items
}


@tracer.wrap()
def convert_to_iso_currency(currency_list: list) -> str | None:
    for currency in currency_list:
        if currency:  # sometimes the endpoint returns [null] for currency
            iso_code = reverse_currency_mapping.get(currency.strip().lower(), None)
            # return the first non-null ISO code
            if iso_code:
                return iso_code
    return None


@tracer.wrap()
def validate_and_convert_number(value: str, num_type: Literal["int", "float"]):
    """Validate and convert number string to float or to int. Log a warning if invalid."""
    try:
        if num_type == "int":
            return int(float(value))
        else:
            return float(value)
    except ValueError:
        logger.warning("Invalid price/award category value: %s. Skipping.", value)
        return None


# Shadow testing
def replicate_to_staging(event, shadowtesting_api_key):  # pylint: disable=C0116
    headers = {
        "Host": "0zobc4e9v9.execute-api.us-east-1.amazonaws.com",
        "x-api-key": shadowtesting_api_key,
        "Content-Type": "application/json",
    }
    url = "https://vpce-0d16b9a55a46d203b-jz9cogr0.execute-api.us-east-1.vpce.amazonaws.com/staging/"

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


@tracer.wrap()
# def invoke_endpoint(endpoint_name, payload_json, otel_context):
def invoke_endpoint(endpoint_name, payload_json):
    """
    Function to invoke a SageMaker endpoint.
    """
    # token = attach(otel_context) if otel_context else None

    logger.info("Invoking endpoint: %s with payload: %s", endpoint_name, payload_json)
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload_json,
    )
    result_value = response["Body"].read().decode()
    result = json.loads(result_value)

    # with otel_tracer.start_as_current_span(name="ner_endpoint") as span:
    #     response = sagemaker_runtime.invoke_endpoint(
    #         EndpointName=endpoint_name,
    #         ContentType="application/json",
    #         Body=payload_json,
    #     )
    #     result_value = response["Body"].read().decode()
    #     result = json.loads(result_value)

    #     span.set_attributes(
    #         {
    #             SpanAttributes.OPENINFERENCE_SPAN_KIND: "ML_MODEL",
    #             SpanAttributes.INPUT_VALUE: json.dumps(payload_json),
    #             SpanAttributes.OUTPUT_VALUE: result_value,
    #         }
    #     )
    #     span.set_status(trace.Status(trace.StatusCode.OK))

    logger.info("Received response from %s: %s", endpoint_name, result)
    # detach(token) if token is not None else None
    return result


@tracer.wrap()
# def bedrock_embed(prompt_data, otel_context=None):
def bedrock_embed(prompt_data):
    # token = attach(otel_context) if otel_context else None

    # try:
    # Invoke model
    response = bedrock_runtime.invoke_model(
        body=json.dumps(
            {"inputText": prompt_data, "dimensions": 512, "normalize": True}
        ),  # Define model parameters
        modelId=ner_embedding_model_id,
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response["body"].read()).get("embedding")
    # finally:
    #     if token is not None:
    #         detach(token)


def create_ld_context(uuid_encrypted):
    """
    define a context for launchdarkly experimentation
    """
    context = Context.builder(uuid_encrypted).kind("user").build()
    return context


@tracer.wrap()
def write_to_gemini_intent_cache(
    recommendations: list, experience: str, destination: str = None
):
    try:
        if not destination:
            destination = "no_destination"
        item = {
            "Experience": experience,
            "Destination": destination,
            "Recommendations": [
                {
                    "name": item["name"],
                    "lat": str(item["lat"]),
                    "lon": str(item["lon"]),
                }
                for item in recommendations
            ],
        }
        gemini_cache_dynamo_table.put_item(Item=item)
        logger.info("Successfully cached recommendations for new experience")
    except Exception:
        logger.exception(
            "Error writing to Dynamo table: %s", gemini_cache_dynamo_table.table_name
        )


@tracer.wrap()
def read_from_intent_cache(
    cache_dax_table,
    key_dict: dict,
):
    try:
        response = cache_dax_table.get_item(Key=key_dict)
        if "Item" in response:
            logger.info("Found experience in cache: %s", response["Item"])
            return response["Item"]
        else:
            logger.info("Not found in intent cache")
            return None
    except Exception:
        logger.exception(
            "Error reading from Dynamo table: %s", cache_dax_table.table_name
        )
        return None


@tracer.wrap()
def get_factory_service(service_account: dict, service_name: str = "gcp"):
    # Create the configuration with the service account and other parameters
    client_params = GcpConfig(
        model_id=gemini_model_id,
        service_account_info=service_account,
        cached_gcp_cred=cached_gcp_cred,
        project_id=gcp_project,
        location=gcp_location,
        logger=logger,
    )
    # Get the factory service based on the service name
    gcp_service = FactoryService.get_factory_service(service_name, client_params)
    logger.info("Using model ID: %s", gcp_service.model_id)

    return gcp_service


@tracer.wrap()
def format_prompt(experience: str, destination: str, prompt_text: str):
    try:
        # Check if the placeholders are present in the prompt
        if "{experience}" not in prompt_text or "{destination}" not in prompt_text:
            logger.error(
                "Placeholders {experience} or {destination} not found in the bedrock prompt",
            )
            raise ValueError("INTENT_SEARCH_API_ERROR_500") from None

        # Replace placeholders with the provided variables
        prompt_text = prompt_text.replace("{experience}", experience).replace(
            "{destination}", destination
        )

        return prompt_text

    except Exception as e:
        logger.exception(
            "An unexpected error occurred when reading the Gemini prompt: %s", e
        )
        raise ValueError("INTENT_SEARCH_API_ERROR_500") from None


@tracer.wrap()
def get_recommendations_from_gemini(experience: str, destination: str) -> str:
    gemini_experience_prompt = format_prompt(experience, destination, gemini_prompt)
    try:
        # read gcp secret
        gcp_svc_acct = get_secret_dict(gcp_secret_arn)
        # Auth to GCP
        gcp_service = get_factory_service(gcp_svc_acct)
        gemini_response = gcp_service.get_response(gemini_experience_prompt)
        logger.info("Gemini response: %s", gemini_response)
        recommendations = json.loads(gemini_response)
        Recommendations(recommendations=recommendations)
    except json.JSONDecodeError:
        logger.exception(
            "Recommendations from Gemini is not valid json: %s", gemini_response.text
        )
        return None
    except ValidationError as e:
        logger.exception("Recommendations from Gemini validation error: %s", e)
        return None
    except Exception as e:
        logger.exception("Unexpected error calling Gemini: %s", e)
        raise ValueError("INTENT_SEARCH_API_ERROR_500") from None
        return None
    return recommendations


@tracer.wrap()
def get_recommendations(experience: str, destination: str):
    if not destination:
        destination = "no_destination"
    recommendations_in_cache = read_from_intent_cache(
        gemini_cache_dax_table,
        key_dict={"Experience": experience, "Destination": destination},
    )
    if recommendations_in_cache:
        logger.info(
            "Found in Gemini intent cache: experience: %s, destination: %s",
            experience,
            destination,
        )
        return recommendations_in_cache.get("Recommendations")
    else:
        recommendations_gemini = get_recommendations_from_gemini(
            experience, destination
        )
        if recommendations_gemini:
            write_to_gemini_intent_cache(
                recommendations=recommendations_gemini,
                experience=experience,
                destination=destination,
            )
            logger.info("New experience added to Gemini cache")
        return recommendations_gemini


ld_client = None


@logger.inject_lambda_context(log_event=True)
@tracer.wrap()
# @otel_trace(span_kind="TOOL", span_name="natural_language_handler", auto_flush=True)
def lambda_handler(event, context):
    """
    Lambda function to invoke two SageMaker endpoints asynchronously.
    """
    # feature flag
    global ld_client  # making ld_client a global variable to avoid creating a new client for every request

    if ld_client is None:
        # reading secrets inside lambda handler as secret layer cannot make a http connection outside
        ld_api_key = get_secret_dict(launchdarkly_api_key_secret_arn)["SDK_KEY"]
        ldclient.set_config(Config(ld_api_key))
        ld_client = ldclient.get()

    search_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    try:
        # Validate the event using Pydantic

        Request(**event)

        uuid_encrypted = hashlib.sha256(event["visitor_id"].encode()).hexdigest()

        context_id = create_ld_context(uuid_encrypted)

        is_bedrock_enabled = generate_flag(
            ld_client,
            feature_flag_name=bedrock_ld_feature_flag_key,
            context_id=context_id,
            search_id=event["search_id"],
            visitor_id=event["visitor_id"],
            search_ts=search_ts,
            default=False,
        )

    except ValidationError as e:
        logger.exception("Request validation error: %s", e)
        raise ValueError("INTENT_SEARCH_API_ERROR_400") from None
    logger.append_keys(search_id=event["search_id"])

    try:
        if environment == "prod":
            # shadowtesting
            shadowtesting_api_key = get_secret_dict(api_key_secret_arn)["key"]

            rand = random.random()
            # Sampled replication
            if rand <= stg_repl_rate:
                repl_event = event.copy()
                logger.info("Sending replicated event to Staging")
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
        if random_sampling <= hr_gt_repl_rate:
            post_to_labeling_job(
                sagemaker=sagemaker,
                sns=sns,
                query=event["search_context"]["query"],
                sns_topic_arn=nlapi_sns_topic_arn,
                tags=get_lambda_tags(
                    lambda_client=lambda_client,
                    lambda_arn=f"{context.invoked_function_arn.split(':function:')[0]}:function:{context.function_name}",
                ),
                labeling_job_configuration=nlapi_hr_config,
            )
    except:
        logger.exception("Human Review replication has failed: Runtime Error")

    try:
        payload_json = json.dumps(event)

        # span = trace.get_current_span()
        # span.set_attributes(
        #     {
        #         "hyatt.intent.search_id": event["search_id"],
        #         "hyatt.intent.channel": event["channel"],
        #         "user.id": event["visitor_id"],
        #     }
        # )

        # otel_context = get_current_context()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_1 = executor.submit(
                invoke_endpoint,
                ner_endpoint,
                payload_json,
                # , otel_context
            )
            future_2 = (
                executor.submit(
                    bedrock_embed,
                    event["search_context"]["query"],
                    # , otel_context
                )
                if is_bedrock_enabled
                else None
            )

            ner_result = future_1.result()
            embedding_result = future_2.result() if future_2 else None
            logger.info("Embedding result: %s", embedding_result)

        logger.info("NER result: %s", ner_result)
        logger.info("Embedding result: %s", embedding_result)

        # ########## 0422 HOT FIX add special triger words to entity lists ############
        ner_result = enrich_ner_result_with_trigger_entities(
            query=event["search_context"]["query"], ner_result=ner_result
        )

        # Form the response
        destinations = [
            {"name": destination, "lat": None, "lon": None}
            for destination in ner_result["destinations"]
        ]

        attributes = [
            {"name": entity.upper().replace(" ", "_"), "sentiment": None}
            for entity in ner_result["attributes"]
        ]
        brands = [{"name": brand, "sentiment": None} for brand in ner_result["brands"]]
        themes = [{"name": theme, "sentiment": None} for theme in ner_result["themes"]]
        experiences = [
            {"name": experience, "sentiment": None}
            for experience in ner_result["experiences"]
        ]
        points_of_interest = [
            {"name": poi, "sentiment": None} for poi in ner_result["points_of_interest"]
        ]
        accommodation_types = [
            {"name": at, "sentiment": None} for at in ner_result["accommodations"]
        ]
        traveler_types = [
            {"name": tt, "sentiment": None} for tt in ner_result["traveler_types"]
        ]
        weather_geography = [
            {"name": gt, "sentiment": None} for gt in ner_result["geo_types"]
        ]
        hotel_characteristics = [
            {"name": hc, "sentiment": None}
            for hc in ner_result["hotel_characteristics"]
        ]
        places = ner_result["places"]["values"]

        # Get recommendations from Known Intent Cache, Gemini Intent Cache, or Gemini
        llm_destinations = None
        if experiences:
            # only support single experience/destination
            experience = experiences[0][
                "name"
            ]  # experiences are standardized in NER model
            destination = (
                destinations[0]["name"].lower().strip() if destinations else None
            )
            logger.info("Searching intent cache for %s in %s", experience, destination)

            # Intent Cache
            # If experience exist in known intent cache - return model output destinations/POI
            if experience.lower() in known_experience_tags:
                logger.info(
                    "Found in known intent cache: %s",
                    experience,
                )
                # POI injection
                if not ner_result["destinations"] and ner_result["points_of_interest"]:
                    destinations = [
                        {"name": destination, "lat": None, "lon": None}
                        for destination in ner_result["points_of_interest"]
                    ]
            # Unknown experience - Use POI if available
            elif ner_result["points_of_interest"]:
                destinations = [
                    {"name": destination, "lat": None, "lon": None}
                    for destination in ner_result["points_of_interest"]
                ]
            # LLM if no POI
            else:
                llm_destinations = get_recommendations(experience, destination)

        # Transform price_filter and enhanced_price_filter
        values = ner_result.get("price_filter", {}).get("values", [])
        rate = ner_result.get("price_filter", {}).get("rate")

        # Validate and convert values
        valid_values = []
        for item in values:
            converted_threshold = validate_and_convert_number(
                item.get("threshold"), "float"
            )
            if converted_threshold is not None:
                valid_values.append(
                    {
                        "value": converted_threshold,
                        "currency": item.get("currency"),
                        "operator": item.get("operator"),
                    }
                )

        if not valid_values:
            transformed_price = {
                "price_filter": {
                    "currency": None,
                    "operator": None,
                    "threshold": None,
                    "rate": None,
                },
                "enhanced_price_filter": {"threshold": None, "rate": None},
            }
        else:
            # Determine price_filter (use "lte" if available, otherwise use first valid item)
            price_filter_value = next(
                (item for item in valid_values if item["operator"] == "lte"),
                valid_values[0],
            )
            transformed_price = {
                "price_filter": {
                    "currency": price_filter_value["currency"],
                    "operator": price_filter_value["operator"],
                    "threshold": price_filter_value["value"],
                    "rate": rate,
                },
                "enhanced_price_filter": {"threshold": valid_values, "rate": rate},
            }

        # Convert award_category threshold values to integers
        award_category = []
        for item in ner_result["award_category"]:
            converted_threshold = validate_and_convert_number(
                item.get("threshold"), "int"
            )
            if converted_threshold is not None:
                award_category.append(
                    {"value": converted_threshold, "operator": item.get("operator")}
                )

        api_response = {
            "query_embedding": embedding_result,
            "embedding_source": f"{ner_embedding_source}:{ner_embedding_model_id}"
            if embedding_result
            else None,
            "destinations": llm_destinations if llm_destinations else destinations,
            "brands": brands,
            "price_filter": transformed_price["price_filter"],
            "enhanced_price_filter": transformed_price["enhanced_price_filter"],
            "award_category": award_category,
            "themes": themes,
            "experiences": experiences,
            "attributes": attributes,
            "points_of_interest": points_of_interest,
            "accommodation_type": accommodation_types,
            "traveler_type": traveler_types,
            "weather_geography": weather_geography,
            "hotel_characteristics": hotel_characteristics,
            "places": places,
        }

        # Log price
        price_log = {
            "search_price_log_id": str(uuid.uuid4()),
            "search_id": event["search_id"],
            "price": api_response["price_filter"].get("threshold"),
            "operator": api_response["price_filter"].get("operator"),
            "currency": api_response["price_filter"].get("currency"),
            "rate": api_response["price_filter"].get("rate"),
            "search_ts": search_ts,
        }
        log_result(price_log, kinesis_firehose_response_price_log)

        # Log NER entities
        entity_type_list = [
            "themes",
            "attributes",
            "experiences",
            "brands",
            "points_of_interest",
            "accommodation_type",
            "traveler_type",
            "weather_geography",
            "hotel_characteristics",
            "places",
        ]

        for i in entity_type_list:
            for entity in api_response[i]:
                entity_dict = {
                    "search_entity_log_id": str(uuid.uuid4()),
                    "search_id": event["search_id"],
                    "entity_type": i,
                    "sentiment": entity.get("sentiment"),
                    "entity_value": entity["name"] if entity.get("name") else json.dumps(entity),
                    "search_ts": search_ts,
                }

                log_result(entity_dict, kinesis_firehose_response_ner_entity_log)

        # Log NER destinations
        for destination in destinations:
            entity_dict = {
                "search_entity_log_id": str(uuid.uuid4()),
                "search_id": event["search_id"],
                "entity_type": "destinations",
                "sentiment": None,
                "entity_value": destination["name"],
                "search_ts": search_ts,
            }

            log_result(entity_dict, kinesis_firehose_response_ner_entity_log)

        # Log LLM destinations
        if llm_destinations:
            for destination in llm_destinations:
                llm_entity_log = {
                    "search_destination_log_id": str(uuid.uuid4()),
                    "search_id": event["search_id"],
                    "destination_name": destination["name"],
                    "lat": float(destination["lat"]),
                    "lon": float(destination["lon"]),
                    "search_ts": search_ts,
                }
                log_result(llm_entity_log, kinesis_firehose_response_llm_entity_log)

        return api_response

    except (ValidationError, Exception) as err:
        logger.exception("lambda_handler has failed: Runtime Error")
        raise ValueError("INTENT_SEARCH_API_ERROR_500") from err
