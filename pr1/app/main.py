"""
main.py — веб-рівень застосунку (FastAPI).

Не містить деталей HTTP-клієнта: все делеговано модулю weather_api.
Відповідальності:
  • HTML-сторінка з формою для вводу міста (GET /)
  • JSON-ендпоінт для отримання погоди (GET /weather?city=...)
  • Обробка помилок: WeatherApiError → відповідний HTTP-статус + JSON
"""

from __future__ import annotations

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .weather_api import WeatherApiError, get_weather

app = FastAPI(title="Weather App", version="1.0.0")


# ── HTML-сторінка ────────────────────────────────────────────────────

HTML_PAGE = """\
<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Погода</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 2rem;
    }
    .card {
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,.2);
      padding: 2.5rem;
      max-width: 480px;
      width: 100%;
    }
    h1 {
      text-align: center;
      color: #333;
      margin-bottom: 1.5rem;
      font-size: 1.75rem;
    }
    .form-group {
      display: flex;
      gap: .75rem;
      margin-bottom: 1.5rem;
    }
    input[type=text] {
      flex: 1;
      padding: .75rem 1rem;
      border: 2px solid #ddd;
      border-radius: 8px;
      font-size: 1rem;
      outline: none;
      transition: border-color .2s;
    }
    input[type=text]:focus { border-color: #667eea; }
    button {
      padding: .75rem 1.5rem;
      background: #667eea;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
      transition: background .2s;
    }
    button:hover { background: #5a6fd6; }
    #result {
      min-height: 80px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    .weather-info {
      text-align: center;
      padding: 1.5rem;
      background: #f7f8fc;
      border-radius: 12px;
      width: 100%;
    }
    .weather-info .city-name {
      font-size: 1.3rem;
      font-weight: 600;
      color: #333;
      margin-bottom: .75rem;
    }
    .weather-info .metrics {
      display: flex;
      justify-content: center;
      gap: 2rem;
    }
    .metric { text-align: center; }
    .metric .value {
      font-size: 2rem;
      font-weight: 700;
      color: #667eea;
    }
    .metric .label {
      font-size: .85rem;
      color: #888;
      margin-top: .25rem;
    }
    .error-msg {
      color: #e74c3c;
      text-align: center;
      font-weight: 500;
    }
    .loading {
      color: #888;
      font-style: italic;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>🌤️ Погода</h1>
    <div class="form-group">
      <input type="text" id="city" placeholder="Введіть назву міста…"
             autofocus autocomplete="off">
      <button onclick="fetchWeather()">Пошук</button>
    </div>
    <div id="result"></div>
  </div>
  <script>
    const input = document.getElementById('city');
    const resultDiv = document.getElementById('result');

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') fetchWeather();
    });

    async function fetchWeather() {
      const city = input.value.trim();
      if (!city) {
        resultDiv.innerHTML = '<p class="error-msg">Будь ласка, введіть назву міста.</p>';
        return;
      }
      resultDiv.innerHTML = '<p class="loading">Завантаження…</p>';
      try {
        const resp = await fetch(`/weather?city=${encodeURIComponent(city)}`);
        const data = await resp.json();
        if (!resp.ok) {
          resultDiv.innerHTML = `<p class="error-msg">${data.detail || 'Невідома помилка'}</p>`;
          return;
        }
        resultDiv.innerHTML = `
          <div class="weather-info">
            <div class="city-name">${data.city}, ${data.country}</div>
            <div class="metrics">
              <div class="metric">
                <div class="value">${data.temperature_c}${data.temperature_unit}</div>
                <div class="label">Температура</div>
              </div>
              <div class="metric">
                <div class="value">${data.wind_speed_kmh}</div>
                <div class="label">Вітер, ${data.wind_speed_unit}</div>
              </div>
            </div>
          </div>`;
      } catch (err) {
        resultDiv.innerHTML = '<p class="error-msg">Не вдалося завантажити дані. Перевірте з\\'єднання.</p>';
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    """Повертає HTML-сторінку з формою для вводу міста."""
    return HTML_PAGE


# ── JSON API ─────────────────────────────────────────────────────────

@app.get("/weather")
async def weather(city: str = Query(default="", description="Назва міста")):
    """
    Ендпоінт для отримання поточної погоди.

    Успіх → 200 + JSON з полями city, country, temperature_c, wind_speed_kmh, …
    Помилка → відповідний HTTP-статус (404 / 502 / 504) + JSON {"detail": "…"}
    """
    try:
        result = get_weather(city)
    except WeatherApiError as exc:
        return JSONResponse(
            status_code=exc.status_hint,
            content={"detail": exc.message},
        )

    return {
        "city": result.city,
        "country": result.country,
        "latitude": result.latitude,
        "longitude": result.longitude,
        "temperature_c": result.temperature_c,
        "wind_speed_kmh": result.wind_speed_kmh,
        "temperature_unit": result.temperature_unit,
        "wind_speed_unit": result.wind_speed_unit,
    }
