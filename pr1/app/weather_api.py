"""
weather_api — модуль інтеграції з Open-Meteo API.

Веб-рівень НЕ залежить від requests і не знає деталей HTTP-запитів.
Він викликає лише get_weather(city) і отримує:
  • словник з погодними даними (успіх), або
  • WeatherApiError з зрозумілим повідомленням (помилка).
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

# ── Константи ────────────────────────────────────────────────────────
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 10  # секунд


# ── Виняток модуля ───────────────────────────────────────────────────
class WeatherApiError(Exception):
    """
    Єдиний виняток, що перетинає межу модуля.

    Атрибути:
        message — текст для користувача.
        status_hint — рекомендований HTTP-статус для веб-рівня:
            • 404 — місто не знайдено / порожній запит
            • 502 — зовнішній API повернув помилку або несподіваний формат
            • 504 — зовнішній API не відповів вчасно (timeout)
    """

    def __init__(self, message: str, *, status_hint: int = 502):
        super().__init__(message)
        self.message = message
        self.status_hint = status_hint


# ── Допоміжний тип результату ────────────────────────────────────────
@dataclass
class WeatherResult:
    """Готовий результат для веб-рівня: назва міста + погодні дані."""

    city: str
    country: str
    latitude: float
    longitude: float
    temperature_c: float
    wind_speed_kmh: float
    temperature_unit: str
    wind_speed_unit: str


# ── Внутрішні функції ────────────────────────────────────────────────

def _geocode(city_name: str) -> dict:
    """
    Геокодування: назва міста → координати.

    Повертає перший результат з Geocoding API або кидає WeatherApiError.
    """
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": city_name, "count": 1, "language": "uk"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise WeatherApiError(
            "Сервіс геокодування не відповів вчасно. Спробуйте пізніше.",
            status_hint=504,
        )
    except requests.exceptions.ConnectionError:
        raise WeatherApiError(
            "Не вдалося з'єднатися з сервісом геокодування.",
            status_hint=502,
        )
    except requests.exceptions.RequestException as exc:
        raise WeatherApiError(
            f"Помилка при з'єднанні з сервісом геокодування: {exc}",
            status_hint=502,
        )

    # ── Обробка HTTP-статусу ──
    if 400 <= resp.status_code < 500:
        raise WeatherApiError(
            f"Некоректний запит до геокодування (HTTP {resp.status_code}).",
            status_hint=502,
        )
    if resp.status_code >= 500:
        raise WeatherApiError(
            f"Сервіс геокодування тимчасово недоступний (HTTP {resp.status_code}).",
            status_hint=502,
        )

    # ── Розбір JSON ──
    try:
        data = resp.json()
    except (ValueError, TypeError):
        raise WeatherApiError(
            "Сервіс геокодування повернув невалідну відповідь.",
            status_hint=502,
        )

    # Ключовий момент: для ненайденого міста API НЕ повертає ключ "results"
    # взагалі, а не повертає порожній список.
    results = data.get("results")
    if not results:
        raise WeatherApiError(
            f"Місто «{city_name}» не знайдено.",
            status_hint=404,
        )

    first = results[0]

    # Перевіряємо наявність обов'язкових полів
    for field in ("latitude", "longitude", "name"):
        if field not in first:
            raise WeatherApiError(
                "Відповідь геокодування має несподіваний формат (відсутнє поле "
                f"'{field}').",
                status_hint=502,
            )

    return first


def _forecast(latitude: float, longitude: float) -> dict:
    """
    Запит поточної погоди за координатами.

    Повертає словник з секцією "current" або кидає WeatherApiError.
    """
    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m",
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise WeatherApiError(
            "Сервіс прогнозу не відповів вчасно. Спробуйте пізніше.",
            status_hint=504,
        )
    except requests.exceptions.ConnectionError:
        raise WeatherApiError(
            "Не вдалося з'єднатися з сервісом прогнозу.",
            status_hint=502,
        )
    except requests.exceptions.RequestException as exc:
        raise WeatherApiError(
            f"Помилка при з'єднанні з сервісом прогнозу: {exc}",
            status_hint=502,
        )

    # ── HTTP-статус ──
    if 400 <= resp.status_code < 500:
        raise WeatherApiError(
            f"Некоректний запит до прогнозу (HTTP {resp.status_code}).",
            status_hint=502,
        )
    if resp.status_code >= 500:
        raise WeatherApiError(
            f"Сервіс прогнозу тимчасово недоступний (HTTP {resp.status_code}).",
            status_hint=502,
        )

    # ── JSON ──
    try:
        data = resp.json()
    except (ValueError, TypeError):
        raise WeatherApiError(
            "Сервіс прогнозу повернув невалідну відповідь.",
            status_hint=502,
        )

    if "current" not in data:
        raise WeatherApiError(
            "Відповідь прогнозу має несподіваний формат (відсутня секція 'current').",
            status_hint=502,
        )

    current = data["current"]
    for field in ("temperature_2m", "wind_speed_10m"):
        if field not in current:
            raise WeatherApiError(
                f"Відповідь прогнозу має несподіваний формат (відсутнє поле "
                f"'{field}').",
                status_hint=502,
            )

    # Одиниці виміру (якщо є)
    units = data.get("current_units", {})
    data["_units"] = {
        "temperature": units.get("temperature_2m", "°C"),
        "wind_speed": units.get("wind_speed_10m", "km/h"),
    }

    return data


# ── Публічний API модуля ─────────────────────────────────────────────

def get_weather(city_name: str) -> WeatherResult:
    """
    Головна функція модуля.

    Приймає назву міста (рядок), повертає WeatherResult.
    У разі будь-якої проблеми кидає WeatherApiError.
    """
    # Валідація вхідних даних
    if not city_name or not city_name.strip():
        raise WeatherApiError(
            "Назва міста не може бути порожньою.",
            status_hint=404,
        )

    city_name = city_name.strip()

    # 1. Геокодування: назва → координати
    geo = _geocode(city_name)

    # 2. Прогноз за координатами
    forecast_data = _forecast(geo["latitude"], geo["longitude"])
    current = forecast_data["current"]
    units = forecast_data["_units"]

    return WeatherResult(
        city=geo["name"],
        country=geo.get("country", ""),
        latitude=geo["latitude"],
        longitude=geo["longitude"],
        temperature_c=current["temperature_2m"],
        wind_speed_kmh=current["wind_speed_10m"],
        temperature_unit=units["temperature"],
        wind_speed_unit=units["wind_speed"],
    )
