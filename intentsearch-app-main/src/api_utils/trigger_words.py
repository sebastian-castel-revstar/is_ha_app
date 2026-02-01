import json
import os
import re

import inflect

# plural to singular
p = inflect.engine()

CURRENCY_SYMBOLS = {"$", "€", "£", "¥"}
CURRENCY_REGEX = "[" + re.escape("".join(CURRENCY_SYMBOLS)) + "]"

with open(
    os.path.join(os.path.dirname(__file__), "trigger_entities.json"),
    "r",
    encoding="utf-8",
) as f:
    TRIGGER_ENTITIES = json.load(f)

QC_TRIGGER_WORDS = {
    "category",
    "catogories" "price",
    "in",
    *CURRENCY_SYMBOLS,
    *[
        synonym.lower()
        for entity_type in TRIGGER_ENTITIES.values()
        for synonyms in entity_type.values()
        for synonym in synonyms
        if " " not in synonym
    ],
}

TRIGGER_PHRASES = {
    synonym.lower()
    for entity_type in TRIGGER_ENTITIES.values()
    for synonyms in entity_type.values()
    for synonym in synonyms
    if " " in synonym
}


def normalize_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"[_\-]", " ", query)
    query = re.sub(r"[^\w\s{re.escape(''.join(CURRENCY_SYMBOLS))}]", "", query)
    query = re.sub(r"\s+", " ", query).strip()

    return query


def tokenize_query(query: str) -> set[str]:
    tokens = re.findall(rf"\b\w+\b|{CURRENCY_REGEX}", query)
    singularized = {p.singular_noun(t) or t for t in tokens}
    return singularized


def qc_contains_trigger_words(query: str) -> bool:
    normalized_query = normalize_query(query)
    if not normalized_query:
        return False

    # regex patterns
    # Match "cat" only when followed by number
    if re.search(r"\bcat\s?\d+\b", normalized_query):
        return True

    # Phrase match - phrase with leading/trailing whitespaces or phrase==query
    for phrase in TRIGGER_PHRASES:
        pattern = rf"(^|\s){re.escape(phrase)}(\s|$)"
        if re.search(pattern, normalized_query):
            return True

    tokens = set(re.findall(rf"\b\w+\b|{CURRENCY_REGEX}", query))
    singularized = {p.singular_noun(t) or t for t in tokens}

    return bool(tokens & QC_TRIGGER_WORDS) or bool(singularized & QC_TRIGGER_WORDS)


def enrich_ner_result_with_trigger_entities(query: str, ner_result: dict):
    normalized_query = normalize_query(query)
    if not normalized_query:
        return ner_result

    tokens = set(re.findall(rf"\b\w+\b|{CURRENCY_REGEX}", query))
    singularized = {p.singular_noun(t) or t for t in tokens}

    for entity_type, tag_map in TRIGGER_ENTITIES.items():
        if ner_result[entity_type] == []:
            for tag, synonyms in tag_map.items():
                if any(s in tokens for s in synonyms) or any(
                    s in singularized for s in synonyms
                ):
                    ner_result[entity_type].append(tag)
                    break

                for synonym in synonyms:
                    if " " in synonym:
                        pattern = rf"(^|\s){re.escape(synonym)}(\s|$)"
                        if re.search(pattern, normalized_query):
                            ner_result[entity_type].append(tag)
                            break

    # Deduplicate
    for k, v in ner_result.items():
        if isinstance(v, list) and all(isinstance(item, str) for item in v):
            ner_result[k] = list(set(v))

    # Testing "ski in europe" -> no results
    # Remove double tagging in amenity: SKI, GOLF, CASINO
    overlap_tags = {"SKI", "GOLF", "CASINO"}
    if "attributes" in ner_result and "experiences" in ner_result:
        attrs = set(ner_result["attributes"])
        exps = set(ner_result["experiences"])
        to_remove = overlap_tags & attrs & exps
        ner_result["attributes"] = list(attrs - to_remove)

    return ner_result
