"""Local evaluation scorer (no extra model call) and the per-case result shape."""

from __future__ import annotations

import math
import re
from typing import Any

from .schemas import EvalCase


def normalize_text(value: str) -> str:
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.%+-]+", " ", value.lower())).strip()


NUMERIC_FACT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?%?$")


NUMERIC_VALUE_RE = re.compile(r"(?<![a-z0-9])[-+]?\d[\d,]*(?:\.\d+)?%?(?![a-z0-9])", re.IGNORECASE)


def numeric_fact_value(token: str) -> float | None:
    if not NUMERIC_FACT_RE.fullmatch(token):
        return None
    try:
        return float(token.rstrip("%").replace(",", ""))
    except ValueError:
        return None


def reply_numbers(text: str) -> list[tuple[float, float]]:
    # Each number in the reply with how far it may sit from the expected value. A reply that rounds
    # to fewer decimals (45.3 for 45.27) still matches; whole numbers stay strict (45 never matches 45.27).
    numbers = []
    for match in NUMERIC_VALUE_RE.finditer(text):
        raw = match.group(0).rstrip("%").replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tolerance = max(0.01, 0.5 * 10 ** -decimals + 1e-9) if decimals else 0.01
        numbers.append((float(raw), tolerance))
    return numbers


def number_matches(expected: float, candidates: list[tuple[float, float]]) -> bool:
    return any(abs(expected - value) <= tolerance for value, tolerance in candidates)


def fact_is_present(token: str, actual: str, actual_numbers: list[tuple[float, float]]) -> bool:
    expected_number = numeric_fact_value(token)
    if expected_number is not None:
        return number_matches(expected_number, actual_numbers)
    return token in actual


def paired_facts_are_present(actual: str, expected: str) -> tuple[bool, str | None]:
    """Keep IDs associated with their expected values instead of matching facts anywhere."""
    actual_id_matches = list(re.finditer(r"mr-\d{2}", actual))
    for raw_clause in re.split(r"[;\n]", expected):
        tokens = normalize_text(raw_clause).split()
        identifiers = [token for token in tokens if re.fullmatch(r"mr-\d{2}", token)]
        expected_numbers = [numeric_fact_value(token) for token in tokens if numeric_fact_value(token) is not None]
        if not identifiers or not expected_numbers:
            continue
        for identifier in identifiers:
            found_pair = False
            for index, match in enumerate(actual_id_matches):
                if match.group(0) != identifier:
                    continue
                segment_end = actual_id_matches[index + 1].start() if index + 1 < len(actual_id_matches) else min(len(actual), match.end() + 300)
                segment = actual[match.start():segment_end]
                segment_numbers = reply_numbers(segment)
                if all(number_matches(value, segment_numbers) for value in expected_numbers):
                    found_pair = True
                    break
            if not found_pair:
                return False, f"expected fact pairing not found: {identifier} with {', '.join(str(value) for value in expected_numbers)}"
    return True, None


def score_reply(reply: str, expected: str) -> tuple[bool, str]:
    # Eval scorer (no extra LLM call): the reply must contain the expected numbers and robot IDs,
    # plus most of the expected words.
    actual = normalize_text(reply)
    target = normalize_text(expected)
    if not actual:
        return False, "empty reply"
    if target in actual:
        return True, "expected answer found in reply"
    paired, pairing_reason = paired_facts_are_present(actual, expected)
    if not paired:
        return False, pairing_reason or "expected fact pairing not found"
    target_tokens = [token for token in target.split() if token not in {"the", "a", "an", "is", "was", "of", "on", "and", "to", "in", "for", "with"}]
    numeric_or_ids = [token for token in target_tokens if any(character.isdigit() for character in token) or token.startswith("mr-")]
    actual_numbers = reply_numbers(reply)
    missing_facts = [token for token in numeric_or_ids if not fact_is_present(token, actual, actual_numbers)]
    if missing_facts:
        return False, f"missing key fact(s): {', '.join(missing_facts)}"
    matched = sum(fact_is_present(token, actual, actual_numbers) for token in target_tokens)
    threshold = max(1, math.ceil(len(target_tokens) * 0.65))
    if matched >= threshold:
        return True, f"matched {matched}/{len(target_tokens)} expected facts"
    return False, f"matched {matched}/{len(target_tokens)} expected facts"


def evaluation_result(index: int, case: EvalCase, run: dict[str, Any], include_trace: bool = False) -> dict[str, Any]:
    passed, reason = score_reply(run["reply"], case.expected)
    result = {
        "index": index,
        "passed": passed,
        "reason": reason,
        "expected": case.expected,
        "reply": run["reply"],
        "visualization": run["visualization"],
        "evidence": run["evidence"],
        "trace_id": run["trace_id"],
    }
    if include_trace:
        result["trace"] = run["trace"]
    return result
