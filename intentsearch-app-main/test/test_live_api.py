import csv
import json
import time
from typing import Optional, Tuple, Union

import boto3

with open("test/sample_event.json", "r") as f:
    json_template = f.read()


def get_endpoint_response(endpoint_name: str, request_json: dict) -> Tuple[dict, float]:
    sagemaker_client = boto3.client("runtime.sagemaker")

    start_time = time.time()
    response = sagemaker_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(request_json),
    )
    end_time = time.time()

    result = json.loads(response["Body"].read().decode())
    latency = start_time - end_time
    latency_ms = latency * 1000

    return result, latency_ms


def latency_check(nlp_latency_ms: int, qc_latency_ms: int, tx_latency_ms: int) -> None:
    assert (
        nlp_latency_ms < 200
    ), "NLP endpoint failed latency check. Response Time greater than 200ms"
    assert (
        qc_latency_ms < 50
    ), "Query Classifier endpoint failed latency check. Response Time greater than 50ms"
    assert (
        tx_latency_ms < 50
    ), "Toxicity endpoint failed latency check. Response Time greater than 200ms"


def check_keys(response_json: dict, required_keys: list) -> None:
    missing_keys = [key for key in required_keys if key not in response_json]
    assert not missing_keys, f"Missing keys in response JSON: {missing_keys}"


def check_response_keys(
    nlp_response_json: dict, qc_response_json: dict, tx_response_json: dict
) -> None:
    check_keys(qc_response_json, ["label", "probability"])
    check_keys(tx_response_json, ["toxicity"])
    check_keys(
        tx_response_json["toxicity"],
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
    )
    check_keys(nlp_response_json, ["destinations", "attributes", "price_filter"])


def check_null_values(obj: Union[dict, list, str, None], path="") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            check_null_values(value, f"{path}.{key}" if path else key)

    elif obj is None or (isinstance(obj, str) and obj.strip() == ""):
        assert 0 == 1, f"Null or empty values found in {obj}"


def apply_null_check(qc_response_json: dict, tx_response_json: dict) -> None:
    check_null_values(qc_response_json)
    check_null_values(tx_response_json)


def validate_response_values(qc_response_json: dict, tx_response_json: dict) -> None:
    if qc_response_json["label"] not in ["INTENT", "DESTINATION", "IRRELEVANT"]:
        assert 0 == 1, "Invalid 'label' value"

    if not isinstance(qc_response_json["probability"], (int, float)) or not (
        0 <= qc_response_json["probability"] <= 1
    ):
        assert (
            0 == 1
        ), "'probabilty' value is not int or float or 'probabilty' value is not less than 1"

    for key, value in tx_response_json["toxicity"].items():
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            assert (
                0 == 1
            ), f"{key} value is not int or float or {key} value is not less than 1: {value}"


def validate_query_labels(qc_response_json: dict, label: str) -> None:
    assert qc_response_json["label"] == label, "Query Classifier label not as expected"


def validate_attributes(nlp_response_json: dict) -> None:
    if "attributes" in nlp_response_json and len(nlp_response_json["attributes"]) > 0:
        for val in nlp_response_json["attributes"]:
            if val.upper().replace(" ", "_") not in [
                "BUSINESS_SERVICES",
                "CASINO",
                "CHROMECAST",
                "CONCIERGE",
                "DIGITAL_CHECK_IN",
                "DIGITAL_KEY",
                "EV_CHARGING",
                "FITNESS_CENTER",
                "FREE_BREAKFAST",
                "FREE_INTERNET",
                "FREE_PARKING",
                "GOLF",
                "KIDS_PROGRAM",
                "LAUNDRY",
                "MEETING_FACILITIES",
                "PET_FRIENDLY",
                "POOL",
                "RESORT_PROPERTY",
                "RESTAURANT_ONSITE",
                "ROOM_SERVICE",
                "SKI",
                "SPA",
            ]:
                assert (
                    0 == 1
                ), "Attribute value in Natural Language response not as expected"


def validate_toxicity_scores(tx_response_json: dict) -> None:
    assert tx_response_json["toxicity"]["toxic"] < 0.5, "'toxic' score not as expected"


def validate_toxic_queries(tx_response_json: dict, query) -> None:
    assert (
        tx_response_json["toxicity"]["toxic"] > 0.5
    ), f"'toxic' score not as expected:{query}"


def validate_identity_hate(tx_response_json: dict) -> None:
    assert (
        tx_response_json["toxicity"]["identity_hate"] > 0.5
    ), "Identity hate statements incorrectly scored by toxicity api"


def validate_obscene_score(tx_response_json) -> None:
    assert (
        tx_response_json["toxicity"]["obscene"] > 0.5
    ), "Obscenity incorrectly scored in toxic query by toxicity api"


def validate_nlp_response(nlp_response_json: dict) -> None:
    if "chicago" not in nlp_response_json["destinations"]:
        assert 0 == 1, "Destination not identified correctly"
    if "POOL" not in nlp_response_json["attributes"]:
        assert 0 == 1, "Attribute not identified correctly"
    if (
        "USD" not in nlp_response_json["price_filter"]["values"][0]["currency"]
        or int(float(nlp_response_json["price_filter"]["values"][0]["threshold"]))
        != 500
    ):
        assert 0 == 1, "Price filter not identified correctly"


def run_tests(query: str, label: str):
    json_event = json_template.replace(
        "{query}",
        query,
    )
    data = json.loads(json_event)

    nlp_response_json, nlp_latency_ms = get_endpoint_response(
        "h-ds-genai-search-ner-v47-staging", data
    )
    qc_response_json, qc_latency_ms = get_endpoint_response(
        "h-ds-genai-search-query-clf-v26-staging", data
    )
    tx_response_json, tx_latency_ms = get_endpoint_response(
        "h-ds-genai-search-toxicity-v5-staging", data
    )

    latency_check(nlp_latency_ms, qc_latency_ms, tx_latency_ms)
    check_response_keys(nlp_response_json, qc_response_json, tx_response_json)
    apply_null_check(qc_response_json, tx_response_json)
    validate_response_values(qc_response_json, tx_response_json)
    validate_query_labels(qc_response_json, label)
    validate_attributes(nlp_response_json)
    validate_toxicity_scores(tx_response_json)

    if "dollars" in query:
        validate_nlp_response(nlp_response_json)


def run_tests_for_toxic_queries(query: str, flag: Optional[str] = None):
    json_event = json_template.replace(
        "{query}",
        query,
    )
    data = json.loads(json_event)

    tx_response_json, tx_latency_ms = get_endpoint_response(
        "h-ds-genai-search-toxicity-v5-staging", data
    )
    validate_toxic_queries(tx_response_json, query)

    if flag == "identity_hate":
        validate_identity_hate(tx_response_json)

    if flag == "obscenity":
        validate_obscene_score(tx_response_json)


class TestClass:
    def test_query(self):
        with open("test/test_queries.csv", newline="") as f:
            queries = [tuple(row) for row in csv.reader(f)][1:]

        for query in queries:
            run_tests(query[0], query[1])

    def test_toxic_queries(self):
        with open("test/toxic_queries.csv", newline="") as f:
            toxic_queries = [row[0] for row in csv.reader(f)][1:]

        for query in toxic_queries:
            run_tests_for_toxic_queries(query)

    run_tests_for_toxic_queries(
        "Washington, DC - Home of the most annoying Muslims and Mexicans!",
        "identity_hate",
    )
    run_tests_for_toxic_queries(
        "I need a room in some hotel in that shithole called Denver, CO...", "obscenity"
    )
