"""External weather and soil-moisture lookups from Open-Meteo."""

from __future__ import annotations

import json
from typing import Any

import httpx

from . import state
from .config import MAX_FILTER_STRING_LENGTH, OPEN_METEO_ARCHIVE_URL, OPEN_METEO_GEOCODING_URL
from .data import parse_date, validate_tool_window
from .state import Dataset


def run_get_weather(dataset: Dataset, args: dict[str, Any]) -> dict[str, Any]:
    # Tool 3: historical weather or soil moisture from Open-Meteo.
    # Steps: check inputs -> turn a city name into coordinates -> fetch archive data -> shape into daily rows.
    location = str(args.get("location") or "").strip()
    data_type = str(args.get("data_type") or "weather").strip().lower()
    if data_type not in {"weather", "soil_moisture"}:
        raise ValueError("data_type must be weather or soil_moisture.")
    latitude = args.get("latitude")
    longitude = args.get("longitude")
    if len(location) > MAX_FILTER_STRING_LENGTH:
        raise ValueError(f"location exceeds maximum length of {MAX_FILTER_STRING_LENGTH}.")
    if location:
        # Treat the user's location text as authoritative. Models sometimes add inferred
        # coordinates as well; geocoding the supplied text avoids trusting that guess.
        latitude = longitude = None
    elif latitude is None or longitude is None:
        raise ValueError("A weather location is required. Provide location or both latitude and longitude.")
    if latitude is not None or longitude is not None:
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError) as exc:
            raise ValueError("latitude and longitude must be numeric.") from exc
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("latitude must be between -90 and 90 and longitude between -180 and 180.")

    validate_tool_window(args)
    date_value = args.get("date")
    if date_value:
        start_date = end_date = parse_date(str(date_value)).strftime("%Y-%m-%d")
    else:
        start = parse_date(args.get("start_date"))
        end = parse_date(args.get("end_date"))
        start_date = (start or parse_date(dataset.profile["time_range"]["start"])).strftime("%Y-%m-%d")
        end_date = (end or start or parse_date(dataset.profile["time_range"]["end"])).strftime("%Y-%m-%d")
    if end_date < start_date:
        raise ValueError("end_date must be after start_date.")

    if location:
        cache_key = json.dumps({"location": location.lower()}, sort_keys=True)
        with state.STATE_LOCK:
            geocoded = state.WEATHER_CACHE.get(cache_key)
        if geocoded is None:
            response = httpx.get(OPEN_METEO_GEOCODING_URL, params={"name": location, "count": 1, "language": "en", "format": "json"}, timeout=10.0)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []
            if not results:
                raise ValueError(f"Open-Meteo could not find a location matching '{location}'.")
            match = results[0]
            geocoded = {
                "name": match.get("name") or location,
                "admin1": match.get("admin1"),
                "country": match.get("country"),
                "latitude": float(match["latitude"]),
                "longitude": float(match["longitude"]),
            }
            with state.STATE_LOCK:
                state.WEATHER_CACHE[cache_key] = geocoded
        latitude = geocoded["latitude"]
        longitude = geocoded["longitude"]
        resolved_location = ", ".join(part for part in [geocoded.get("name"), geocoded.get("admin1"), geocoded.get("country")] if part)
    else:
        resolved_location = f"{latitude:.4f}, {longitude:.4f}"

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    if data_type == "soil_moisture":
        params["hourly"] = "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm"
    else:
        params["daily"] = "weather_code,temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,wind_speed_10m_max"
    response = httpx.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload.get("reason") or "Open-Meteo returned a weather error.")
    rows = []
    if data_type == "soil_moisture":
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        variables = {
            "soil_moisture_0_to_7cm": "soil_moisture_0_to_7cm_m3_m3",
            "soil_moisture_7_to_28cm": "soil_moisture_7_to_28cm_m3_m3",
            "soil_moisture_28_to_100cm": "soil_moisture_28_to_100cm_m3_m3",
            "soil_moisture_100_to_255cm": "soil_moisture_100_to_255cm_m3_m3",
        }
        grouped: dict[str, dict[str, list[float]]] = {}
        for index, timestamp in enumerate(times):
            date = str(timestamp)[:10]
            day = grouped.setdefault(date, {output: [] for output in variables.values()})
            for source, output in variables.items():
                values = hourly.get(source) or []
                value = values[index] if index < len(values) else None
                if isinstance(value, (int, float)):
                    day[output].append(float(value))
        for date, values_by_depth in grouped.items():
            rows.append({
                "date": date,
                **{
                    key: round(sum(values) / len(values), 4) if values else None
                    for key, values in values_by_depth.items()
                },
            })
    else:
        daily = payload.get("daily") or {}
        times = daily.get("time") or []
        for index, date in enumerate(times):
            rows.append({
                "date": date,
                "weather_code": (daily.get("weather_code") or [None] * len(times))[index],
                "temperature_mean_c": (daily.get("temperature_2m_mean") or [None] * len(times))[index],
                "temperature_max_c": (daily.get("temperature_2m_max") or [None] * len(times))[index],
                "temperature_min_c": (daily.get("temperature_2m_min") or [None] * len(times))[index],
                "precipitation_mm": (daily.get("precipitation_sum") or [None] * len(times))[index],
                "rain_mm": (daily.get("rain_sum") or [None] * len(times))[index],
                "wind_max_kmh": (daily.get("wind_speed_10m_max") or [None] * len(times))[index],
            })
    return {
        "source": "Open-Meteo historical weather API",
        "data_type": data_type,
        "location": resolved_location,
        "latitude": latitude,
        "longitude": longitude,
        "effective_time_range": {"start": f"{start_date}T00:00:00", "end": f"{end_date}T23:59:59"},
        "rows": rows,
        "units": ({"soil_moisture": "m³/m³"} if data_type == "soil_moisture" else {"temperature": "°C", "precipitation": "mm", "wind": "km/h"}),
        "note": ("Historical soil moisture is external reanalysis data, not an on-farm sensor measurement or a value stored in the uploaded CSV." if data_type == "soil_moisture" else "Historical weather is external reanalysis data, not a measurement stored in the uploaded CSV."),
    }
