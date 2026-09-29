"""
evaluate.py — скрипт оцінювання якості детектора.

Прогоняє детектор на зображеннях з eval/images/,
порівнює результати з expected.json і виводить звіт.

Запуск:
  python eval/evaluate.py
  python eval/evaluate.py --confidence 0.5   # з іншим порогом
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Додаємо корінь проєкту до sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
from app.detector import detect, load_model


def load_expected(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["images"]


def evaluate_image(
    image_path: Path,
    expected_objects: list[dict],
    confidence: float,
) -> dict:
    """Порівняти результати детекції з очікуваними.

    Returns:
        dict з полями: file, detections, expected, tp, fp, fn, wrong_class, notes
    """
    image = Image.open(image_path).convert("RGB")
    result = detect(image, confidence=confidence)

    # Підрахунок фактичних класів
    detected_counts = Counter(d["class_name"] for d in result["detections"])

    # Підрахунок очікуваних класів
    expected_counts = Counter()
    for obj in expected_objects:
        expected_counts[obj["class"]] = obj["count"]

    # Аналіз
    tp = 0  # True Positives
    fp = 0  # False Positives (хибні спрацювання)
    fn = 0  # False Negatives (пропуски)
    notes = []

    all_classes = set(detected_counts.keys()) | set(expected_counts.keys())

    for cls in sorted(all_classes):
        det = detected_counts.get(cls, 0)
        exp = expected_counts.get(cls, 0)

        matched = min(det, exp)
        tp += matched

        if det > exp:
            extra = det - exp
            fp += extra
            if exp == 0:
                notes.append(f"  FP: '{cls}' — знайдено {det}, не очікувалося")
            else:
                notes.append(f"  FP: '{cls}' — знайдено {det}, очікувалося {exp} (+{extra} зайвих)")

        if det < exp:
            missed = exp - det
            fn += missed
            notes.append(f"  FN: '{cls}' — знайдено {det}, очікувалося {exp} (пропущено {missed})")

    return {
        "file": str(image_path.name),
        "inference_time_ms": result["inference_time_ms"],
        "detected_count": result["count"],
        "expected_total": sum(expected_counts.values()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "detections": result["detections"],
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description="Оцінювання якості детектора")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Поріг впевненості (за замовчуванням 0.25)",
    )
    args = parser.parse_args()

    eval_dir = Path(__file__).resolve().parent
    expected_path = eval_dir / "expected.json"

    if not expected_path.exists():
        print(f"❌ Не знайдено {expected_path}")
        sys.exit(1)

    expected_data = load_expected(expected_path)

    # Завантажити модель один раз
    print("Завантаження моделі...")
    load_model()

    print(f"\n{'='*60}")
    print(f"  Оцінювання якості (поріг = {args.confidence})")
    print(f"{'='*60}\n")

    total_tp = total_fp = total_fn = 0
    results_for_report = []

    for item in expected_data:
        image_path = eval_dir / item["file"]
        if not image_path.exists():
            print(f"⚠️  Файл {image_path} не знайдено — пропускаю.\n")
            continue

        report = evaluate_image(image_path, item["expected_objects"], args.confidence)
        results_for_report.append(report)
        total_tp += report["tp"]
        total_fp += report["fp"]
        total_fn += report["fn"]

        status = "✅" if report["fp"] == 0 and report["fn"] == 0 else "⚠️"
        print(f"{status} {report['file']} — {item.get('description', '')}")
        print(f"   Знайдено: {report['detected_count']}, очікувалося: {report['expected_total']}")
        print(f"   TP={report['tp']}  FP={report['fp']}  FN={report['fn']}  ({report['inference_time_ms']} мс)")
        for note in report["notes"]:
            print(note)
        print()

    # Загальний підсумок
    print(f"{'='*60}")
    print(f"  ПІДСУМОК")
    print(f"{'='*60}")
    print(f"  Зображень оцінено: {len(results_for_report)}")
    print(f"  True Positives:    {total_tp}")
    print(f"  False Positives:   {total_fp}")
    print(f"  False Negatives:   {total_fn}")
    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
        print(f"  Recall:            {recall:.2%}")
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
        print(f"  Precision:         {precision:.2%}")
    print()

    # ── Порівняння порогів ────────────────────────────────────
    if args.confidence == 0.25:
        print("💡 Порада: спробуйте різні пороги, щоб побачити як вони впливають:")
        print("   python eval/evaluate.py --confidence 0.1")
        print("   python eval/evaluate.py --confidence 0.5")
        print("   python eval/evaluate.py --confidence 0.7")


if __name__ == "__main__":
    main()
