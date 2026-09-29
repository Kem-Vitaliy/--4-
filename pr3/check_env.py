"""
check_env.py — перевірка середовища перед запуском.

Перевіряє:
  1. Версію Python (>= 3.10)
  2. Наявність усіх залежностей із requirements.txt
  3. Наявність файлу .env та ключа LLM_API_KEY
  4. Наявність context.md
  5. Тестовий запит до API (опціонально)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Примусово UTF-8 для Windows-консолі (щоб emoji друкувалися)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def check_python_version():
    v = sys.version_info
    if v < (3, 10):
        print(f"❌ Python {v.major}.{v.minor} — потрібна версія >= 3.10")
        return False
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_dependencies():
    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "openai": "openai",
        "dotenv": "dotenv",
    }
    all_ok = True
    for name, import_name in packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} — не встановлено. Виконайте: pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_env_file():
    env_path = ROOT / ".env"
    if not env_path.exists():
        print("❌ Файл .env не знайдено. Скопіюйте .env.example → .env і заповніть LLM_API_KEY.")
        return False

    content = env_path.read_text(encoding="utf-8")
    if "LLM_API_KEY" not in content or "your-api-key-here" in content:
        print("❌ LLM_API_KEY у .env не задано або містить значення за замовчуванням.")
        return False

    print("✅ .env з LLM_API_KEY")
    return True


def check_context():
    ctx_path = ROOT / "context.md"
    if not ctx_path.exists():
        print("❌ context.md не знайдено.")
        return False
    size = ctx_path.stat().st_size
    print(f"✅ context.md ({size} байт)")
    return True


def check_api_connection():
    """Спробувати надіслати тестовий запит до API."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")

        import os
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            timeout=15,
        )
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            messages=[{"role": "user", "content": "Відповідай одним словом: працює?"}],
            max_tokens=10,
        )
        answer = resp.choices[0].message.content
        print(f"✅ API працює. Відповідь: {answer}")
        return True
    except Exception as e:
        print(f"❌ API помилка: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Перевірка середовища для ПР3")
    print("=" * 50)

    results = [
        check_python_version(),
        check_dependencies(),
        check_env_file(),
        check_context(),
    ]

    if all(results):
        print("\n--- Тест з'єднання з API ---")
        check_api_connection()

    print("=" * 50)
    if all(results):
        print("✅ Середовище готове! Запуск: uvicorn app.main:app --reload")
    else:
        print("❌ Є проблеми. Виправте їх і запустіть перевірку знову.")
