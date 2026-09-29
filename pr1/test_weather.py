"""Quick test script for all weather API scenarios."""
import json
import requests

BASE = "http://127.0.0.1:8000"

tests = [
    ("Kyiv", "valid city"),
    ("London", "valid city"),
    ("Tokyo", "valid city"),
    ("asdfghjkl123", "non-existent city"),
    ("", "empty string"),
    ("   ", "whitespace only"),
]

for city, desc in tests:
    r = requests.get(f"{BASE}/weather", params={"city": city})
    body = r.json()
    status = r.status_code
    if status == 200:
        print(f"[{status}] {desc:20s} | city={city!r:20s} -> "
              f"{body['temperature_c']}{body['temperature_unit']}, "
              f"wind {body['wind_speed_kmh']} {body['wind_speed_unit']}")
    else:
        print(f"[{status}] {desc:20s} | city={city!r:20s} -> {body['detail']}")
