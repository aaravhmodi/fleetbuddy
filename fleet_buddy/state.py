"""In-memory storage shared by the whole app. Everything here is lost when the server restarts."""

from __future__ import annotations

import threading
from typing import Any

import pandas as pd


class Dataset:
    def __init__(self, dataset_id: str, frame: pd.DataFrame, profile: dict[str, Any]):
        self.id = dataset_id
        self.frame = frame
        self.profile = profile


# Everything lives in memory (lost on restart): uploaded datasets, chat traces, and cached
# location lookups. The lock stops two requests from changing these at the same time.
DATASETS: dict[str, Dataset] = {}


TRACES: list[dict[str, Any]] = []


WEATHER_CACHE: dict[str, dict[str, Any]] = {}


STATE_LOCK = threading.RLock()
