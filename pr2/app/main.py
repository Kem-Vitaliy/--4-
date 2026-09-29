"""
main.py — веб-рівень (FastAPI).

Відповідальність:
  - Прийняти завантажене зображення.
  - Валідувати файл (тип, непорожність).
  - Делегувати inference модулю detector.
  - Повернути JSON-результат або HTTP-помилку.
  - Віддати HTML-сторінку з інтерфейсом.

Веб-рівень НЕ імпортує ultralytics / YOLO напряму.
Він знає тільки про detector.load_model() і detector.detect().
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app import detector

# ── Допустимі MIME-типи ──────────────────────────────────────
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ── Lifespan: завантаження моделі один раз при старті ────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Завантажити модель при старті застосунку, а не на кожен запит."""
    detector.load_model()
    yield


app = FastAPI(
    title="Object Detector",
    description="YOLOv8n-based object detection web app",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Головна сторінка ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """Повертає HTML-сторінку з UI для завантаження зображення."""
    return _INDEX_HTML


# ── Ендпоінт детекції ────────────────────────────────────────
@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Query(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Мінімальний поріг впевненості (0.0 – 1.0)",
    ),
):
    """Прийняти зображення, виконати детекцію, повернути JSON.

    Query-параметр `confidence` дозволяє користувачеві змінювати
    поріг впевненості без перезапуску сервера.

    Обробка помилок:
      - 400: файл не вибрано / порожній
      - 415: файл не є зображенням (за MIME та PIL)
      - 413: файл завеликий
      - 500: несподівана помилка inference
    """
    # 1. Перевірка наявності файлу
    if not file or not file.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "Файл не вибрано."},
        )

    # 2. Перевірка MIME-типу (швидка, до читання тіла)
    if file.content_type and file.content_type not in ALLOWED_MIME:
        return JSONResponse(
            status_code=415,
            content={
                "error": f"Непідтримуваний тип файлу: {file.content_type}. "
                f"Очікується зображення (JPEG, PNG, WebP, BMP, GIF)."
            },
        )

    # 3. Прочитати вміст
    contents = await file.read()

    # 4. Перевірка порожності
    if len(contents) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Файл порожній."},
        )

    # 5. Перевірка розміру
    if len(contents) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413,
            content={"error": f"Файл завеликий (максимум {MAX_FILE_SIZE // (1024*1024)} MB)."},
        )

    # 6. Спробувати відкрити як зображення (глибша валідація)
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()  # примусово декодувати — ловить битий вміст
        image = image.convert("RGB")  # YOLO очікує RGB
    except (UnidentifiedImageError, Exception):
        return JSONResponse(
            status_code=415,
            content={"error": "Файл не вдалося розпізнати як зображення."},
        )

    # 7. Inference (делегуємо detector)
    try:
        result = detector.detect(image, confidence=confidence)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Помилка inference: {str(e)}"},
        )

    return result


# ── HTML UI ──────────────────────────────────────────────────
_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Object Detector — YOLOv8</title>
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --text-muted: #94a3b8; --accent: #3b82f6;
    --accent-hover: #2563eb; --success: #22c55e; --danger: #ef4444;
  }
  *, *::before, *::after { box-sizing: border-box; }
  body {
    margin: 0; font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; justify-content: center; padding: 2rem 1rem;
    min-height: 100vh;
  }
  .container { max-width: 900px; width: 100%; }
  h1 { text-align: center; font-size: 1.8rem; margin-bottom: 0.3rem; }
  .subtitle { text-align: center; color: var(--text-muted); margin-bottom: 2rem; }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
  }

  .form-row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: flex-end; }
  .form-group { flex: 1; min-width: 200px; }
  .form-group label { display: block; margin-bottom: 0.4rem; font-size: 0.9rem; color: var(--text-muted); }

  input[type="file"], input[type="range"] { width: 100%; }
  input[type="file"] {
    padding: 0.6rem; border: 1px dashed var(--border); border-radius: 8px;
    background: var(--bg); color: var(--text); cursor: pointer;
  }

  .slider-wrapper { display: flex; align-items: center; gap: 0.8rem; }
  input[type="range"] { flex: 1; accent-color: var(--accent); }
  .slider-value {
    min-width: 3rem; text-align: center; font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  button {
    padding: 0.7rem 2rem; background: var(--accent); color: #fff;
    border: none; border-radius: 8px; font-size: 1rem; cursor: pointer;
    transition: background 0.2s;
  }
  button:hover { background: var(--accent-hover); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }

  .preview-area {
    position: relative; margin-top: 1rem; text-align: center;
    display: none;
  }
  .preview-area img {
    max-width: 100%; max-height: 500px; border-radius: 8px;
  }
  .preview-area canvas {
    position: absolute; top: 0; left: 50%;
    transform: translateX(-50%); border-radius: 8px;
    pointer-events: none;
  }

  .stats {
    display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;
  }
  .stat-card {
    flex: 1; min-width: 120px; background: var(--bg);
    border-radius: 8px; padding: 0.8rem; text-align: center;
  }
  .stat-card .value { font-size: 1.4rem; font-weight: 700; }
  .stat-card .label { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem; }

  table {
    width: 100%; border-collapse: collapse; font-size: 0.9rem;
  }
  th, td {
    padding: 0.6rem 0.8rem; text-align: left;
    border-bottom: 1px solid var(--border);
  }
  th { color: var(--text-muted); font-weight: 500; font-size: 0.8rem; text-transform: uppercase; }

  .badge {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
    font-size: 0.8rem; font-weight: 600;
  }
  .badge-high { background: #16a34a33; color: #22c55e; }
  .badge-mid  { background: #ca8a0433; color: #eab308; }
  .badge-low  { background: #dc262633; color: #ef4444; }

  .error-box {
    background: #dc262620; border: 1px solid var(--danger); border-radius: 8px;
    padding: 1rem; color: var(--danger); display: none;
  }

  .spinner {
    display: none; margin: 1rem auto; width: 36px; height: 36px;
    border: 3px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #results-section { display: none; }
</style>
</head>
<body>
<div class="container">
  <h1>🔍 Object Detector</h1>
  <p class="subtitle">YOLOv8n &mdash; локальний inference</p>

  <div class="card">
    <div class="form-row">
      <div class="form-group" style="flex:2">
        <label for="file-input">Зображення</label>
        <input type="file" id="file-input" accept="image/*">
      </div>
      <div class="form-group">
        <label for="conf-slider">Поріг впевненості</label>
        <div class="slider-wrapper">
          <input type="range" id="conf-slider" min="0.05" max="0.95" step="0.05" value="0.25">
          <span class="slider-value" id="conf-value">0.25</span>
        </div>
      </div>
      <div>
        <button id="detect-btn" disabled>Detect</button>
      </div>
    </div>

    <div class="preview-area" id="preview-area">
      <img id="preview-img">
      <canvas id="overlay-canvas"></canvas>
    </div>
  </div>

  <div class="spinner" id="spinner"></div>
  <div class="error-box" id="error-box"></div>

  <div id="results-section">
    <div class="card">
      <div class="stats" id="stats"></div>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Клас</th><th>Впевненість</th>
            <th>x1</th><th>y1</th><th>x2</th><th>y2</th>
          </tr>
        </thead>
        <tbody id="results-body"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const fileInput   = document.getElementById('file-input');
const confSlider  = document.getElementById('conf-slider');
const confValue   = document.getElementById('conf-value');
const detectBtn   = document.getElementById('detect-btn');
const previewArea = document.getElementById('preview-area');
const previewImg  = document.getElementById('preview-img');
const canvas      = document.getElementById('overlay-canvas');
const spinner     = document.getElementById('spinner');
const errorBox    = document.getElementById('error-box');
const resultsSection = document.getElementById('results-section');
const statsDiv    = document.getElementById('stats');
const tbody       = document.getElementById('results-body');

confSlider.addEventListener('input', () => {
  confValue.textContent = parseFloat(confSlider.value).toFixed(2);
});

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if (!f) return;
  detectBtn.disabled = false;
  previewArea.style.display = 'block';
  const url = URL.createObjectURL(f);
  previewImg.onload = () => {
    canvas.width = previewImg.naturalWidth;
    canvas.height = previewImg.naturalHeight;
    canvas.style.maxWidth = previewImg.clientWidth + 'px';
    canvas.style.maxHeight = previewImg.clientHeight + 'px';
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };
  previewImg.src = url;
});

detectBtn.addEventListener('click', async () => {
  const f = fileInput.files[0];
  if (!f) return;

  detectBtn.disabled = true;
  spinner.style.display = 'block';
  errorBox.style.display = 'none';
  resultsSection.style.display = 'none';

  const formData = new FormData();
  formData.append('file', f);
  const conf = confSlider.value;

  try {
    const resp = await fetch(`/detect?confidence=${conf}`, {
      method: 'POST',
      body: formData,
    });
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }

    renderResults(data);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.style.display = 'block';
  } finally {
    detectBtn.disabled = false;
    spinner.style.display = 'none';
  }
});

function renderResults(data) {
  // Stats
  statsDiv.innerHTML = `
    <div class="stat-card"><div class="value">${data.count}</div><div class="label">Об'єктів</div></div>
    <div class="stat-card"><div class="value">${data.inference_time_ms}</div><div class="label">мс (inference)</div></div>
    <div class="stat-card"><div class="value">${data.confidence_threshold}</div><div class="label">Поріг</div></div>
    <div class="stat-card"><div class="value">${data.image_size.width}×${data.image_size.height}</div><div class="label">Розмір (px)</div></div>
  `;

  // Table
  tbody.innerHTML = '';
  data.detections.forEach((d, i) => {
    const badgeClass = d.confidence >= 0.7 ? 'badge-high' : d.confidence >= 0.4 ? 'badge-mid' : 'badge-low';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${d.class_name}</td>
      <td><span class="badge ${badgeClass}">${(d.confidence * 100).toFixed(1)}%</span></td>
      <td>${d.bbox.x1.toFixed(0)}</td><td>${d.bbox.y1.toFixed(0)}</td>
      <td>${d.bbox.x2.toFixed(0)}</td><td>${d.bbox.y2.toFixed(0)}</td>
    `;
    tbody.appendChild(tr);
  });

  // Draw bounding boxes on canvas
  drawBoxes(data.detections, data.image_size);

  resultsSection.style.display = 'block';
}

// Color palette for classes
const COLORS = [
  '#3b82f6','#22c55e','#ef4444','#eab308','#a855f7',
  '#ec4899','#14b8a6','#f97316','#6366f1','#06b6d4',
];
const classColorMap = {};
let colorIdx = 0;
function getColor(cls) {
  if (!(cls in classColorMap)) {
    classColorMap[cls] = COLORS[colorIdx % COLORS.length];
    colorIdx++;
  }
  return classColorMap[cls];
}

function drawBoxes(detections, imgSize) {
  const ctx = canvas.getContext('2d');
  canvas.width = imgSize.width;
  canvas.height = imgSize.height;
  canvas.style.maxWidth = previewImg.clientWidth + 'px';
  canvas.style.maxHeight = previewImg.clientHeight + 'px';
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  detections.forEach(d => {
    const color = getColor(d.class_name);
    const {x1, y1, x2, y2} = d.bbox;
    const w = x2 - x1, h = y2 - y1;

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, Math.round(imgSize.width / 300));
    ctx.strokeRect(x1, y1, w, h);

    const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
    const fontSize = Math.max(12, Math.round(imgSize.width / 50));
    ctx.font = `bold ${fontSize}px sans-serif`;
    const tm = ctx.measureText(label);
    const pad = 4;

    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - fontSize - pad * 2, tm.width + pad * 2, fontSize + pad * 2);

    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + pad, y1 - pad);
  });
}
</script>
</body>
</html>
"""
