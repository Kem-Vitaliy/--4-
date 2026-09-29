"""
detector.py — модуль inference (робота з моделлю).

Відповідальність:
  - Завантажити модель один раз (при імпорті / старті застосунку).
  - Виконати inference на зображенні з заданим порогом confidence.
  - Повернути структурований результат (список об'єктів + час inference).

Веб-рівень НЕ знає деталей моделі (ні YOLO, ні ultralytics).
Він викликає detect() і отримує чистий dict.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

# ── Константи ────────────────────────────────────────────────
DEFAULT_CONFIDENCE: float = 0.25  # поріг за замовчуванням
MODEL_WEIGHTS: str = "yolov8n.pt"  # легка модель, працює на CPU

# ── Singleton-завантаження моделі ────────────────────────────
_model: YOLO | None = None


def load_model() -> YOLO:
    """Завантажити модель один раз і закешувати.

    Чому саме при першому виклику, а не на рівні модуля?
      — Це дозволяє контролювати момент завантаження (наприклад,
        у lifespan FastAPI) і спрощує тестування.
    """
    global _model
    if _model is None:
        _model = YOLO(MODEL_WEIGHTS)
    return _model


def detect(
    image: Image.Image,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Виконати детекцію об'єктів на зображенні.

    Args:
        image: PIL-зображення (RGB).
        confidence: мінімальний поріг впевненості (0.0 – 1.0).

    Returns:
        dict з ключами:
          - detections: list[dict] — знайдені об'єкти
              кожен: {class_name, class_id, confidence, bbox:{x1,y1,x2,y2}}
          - count: int — кількість об'єктів
          - inference_time_ms: float — час inference у мілісекундах
          - confidence_threshold: float — використаний поріг
          - image_size: {width, height}
    """
    model = load_model()

    # ── Вимірювання часу inference ────────────────────────────
    # time.perf_counter() — найточніший таймер (наносекунди).
    # Вимірюємо ТІЛЬКИ model(...) — без парсингу результатів.
    start = time.perf_counter()
    results = model(image, conf=confidence, verbose=False)
    elapsed = time.perf_counter() - start
    inference_ms = round(elapsed * 1000, 2)

    # ── Парсинг результатів ───────────────────────────────────
    result = results[0]  # один кадр — один елемент
    boxes = result.boxes
    names = result.names  # {id: class_name}

    detections: list[dict[str, Any]] = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = round(float(box.conf[0].item()), 4)
        x1, y1, x2, y2 = [round(float(c), 2) for c in box.xyxy[0].tolist()]
        detections.append(
            {
                "class_name": names[cls_id],
                "class_id": cls_id,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )

    w, h = image.size
    return {
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": inference_ms,
        "confidence_threshold": confidence,
        "image_size": {"width": w, "height": h},
    }
