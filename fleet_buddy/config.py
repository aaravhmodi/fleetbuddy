"""Settings: the CSV shape we accept, which model to use, pricing, and safety limits."""

import os


# --- Settings: the CSV shape we accept, which model to use, pricing, and safety limits ---
EXPECTED_COLUMNS = [
    "ts",
    "robot_id",
    "field",
    "state",
    "battery_pct",
    "nitrogen_applied_l",
    "distance_m",
]


VALID_STATES = {"applying", "driving", "charging", "idle", "fault"}


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


INPUT_PRICE_PER_TOKEN = 0.15 / 1_000_000


OUTPUT_PRICE_PER_TOKEN = 0.60 / 1_000_000


MAX_MODEL_CALLS = 8


MAX_TOOL_ROWS = 50


MAX_QUERY_WINDOW_DAYS = 90


MAX_FILTER_STRING_LENGTH = 64


PROMPT_VERSION = "fleet-buddy-2026-10-07.4"


OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
