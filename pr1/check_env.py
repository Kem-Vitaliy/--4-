"""
check_env.py — перевірка середовища та доступу до Open-Meteo API.
Запустіть перед роботою: python check_env.py
"""

import sys


def check_imports():
    """Перевіряє, чи встановлені потрібні бібліотеки."""
    errors = []
    for name in ("fastapi", "uvicorn", "requests"):
        try:
            __import__(name)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} — не знайдено")
            errors.append(name)
    return errors


def check_geocoding():
    """Перевіряє доступ до Geocoding API."""
    import requests

    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(url, params={"name": "Kyiv", "count": 1}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise RuntimeError("Geocoding повернув порожній результат для 'Kyiv'")
    city = results[0]
    print(f"  Місто: {city['name']}, lat={city['latitude']}, lon={city['longitude']}")
    return city["latitude"], city["longitude"]


def check_forecast(lat: float, lon: float):
    """Перевіряє доступ до Forecast API."""
    import requests

    url = "https://api.open-meteo.com/v1/forecast"
    resp = requests.get(
        url,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current", {})
    print(
        f"  Температура: {current.get('temperature_2m')} °C, "
        f"Вітер: {current.get('wind_speed_10m')} км/г"
    )


def main():
    print(f"Python {sys.version}\n")

    print("1. Перевірка бібліотек:")
    errors = check_imports()
    if errors:
        print(f"\n[FAIL] Встановіть відсутні пакети: pip install {' '.join(errors)}")
        sys.exit(1)

    print("\n2. Перевірка Geocoding API (Kyiv):")
    try:
        lat, lon = check_geocoding()
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        sys.exit(1)

    print("\n3. Перевірка Forecast API:")
    try:
        check_forecast(lat, lon)
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        sys.exit(1)

    print("\n[OK] Середовище готове до роботи!")


if __name__ == "__main__":
    main()
