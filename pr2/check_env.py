"""
check_env.py — перевірка середовища та попереднє завантаження ваг моделі.

Запустіть цей скрипт один раз перед роботою, щоб:
  1. Переконатися, що всі пакети встановлені.
  2. Завантажити ваги YOLOv8n (≈6 MB) з мережі, якщо їх ще немає.
"""

import importlib
import sys


def check_package(name: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"  [OK] {name}")
        return True
    except ImportError:
        print(f"  [FAIL] {name} — не знайдено")
        return False


def main():
    print("=== Перевірка пакетів ===")
    packages = ["ultralytics", "fastapi", "uvicorn", "PIL", "jinja2"]
    results = [check_package(p) for p in packages]
    if not all(results):
        print("\nДеякі пакети відсутні. Встановіть їх:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print("\n=== Завантаження ваг YOLOv8n ===")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print(f"  Модель завантажена: {model.model_name if hasattr(model, 'model_name') else 'yolov8n.pt'}")

    print("\n=== Тестовий inference ===")
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(dummy, verbose=False)
    print(f"  Тестовий прогін успішний. Знайдено об'єктів: {len(results[0].boxes)}")

    print("\n✅ Середовище готове!")


if __name__ == "__main__":
    main()
