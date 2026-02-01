import re

from api_utils.toxicity_blacklist_values import (
    EXACT_MATCH_VALUES as blacklist_exact_match_values,
)
from api_utils.toxicity_blacklist_values import (
    REGEX_EXPRESSIONS as blacklist_regex_expressions,
)
from api_utils.toxicity_whitelist_values import (
    EXACT_MATCH_VALUES as whitelist_exact_match_values,
)
from api_utils.toxicity_whitelist_values import (
    REGEX_EXPRESSIONS as whitelist_regex_expressions,
)


def normalize_expression(query: str) -> str:
    query = query.lower().replace(",", " ")
    query = re.sub(r"[_\-]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    return query


def toxicity_whitelist(query: str) -> bool:
    print("Checking toxicity whitelist")
    return check_if_applies(
        query, whitelist_exact_match_values, whitelist_regex_expressions
    )


def toxicity_blacklist(query: str) -> bool:
    print("Checking toxicity blacklist")
    return check_if_applies(
        query, blacklist_exact_match_values, blacklist_regex_expressions
    )


def check_if_applies(query: str, values: list, regex_expressions: list) -> bool:
    exact_match = check_exact_match(query, values)
    regex_match = check_regex_match(query, regex_expressions)
    return exact_match or regex_match


def check_exact_match(query: str, values: list) -> bool:
    for phrase in values:
        pattern = rf"(^|\s){re.escape(normalize_expression(phrase))}(\s|$)"
        if re.search(pattern, normalize_expression(query)):
            print(f"Query: {query} matched against {pattern}")
            return True
    return False


def check_regex_match(query: str, regex_expressions: list) -> bool:
    for pattern in regex_expressions:
        if re.search(pattern, query, re.IGNORECASE):
            print(f"Query: {query} matched against {pattern}")
            return True
    return False
