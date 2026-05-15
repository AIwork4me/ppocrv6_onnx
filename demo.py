#!/usr/bin/env python3
"""PP-OCRv6 ONNX Runtime Demo — quick-start example."""

import sys
from pathlib import Path

import cv2
import numpy as np

from ppocrv6_onnx import PPOCRv6Onnx, OCRResult

# ── Configuration ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

DET_MODEL = MODEL_DIR / "PP-OCRv6_tiny_det_onnx" / "inference.onnx"
REC_MODEL = MODEL_DIR / "PP-OCRv6_tiny_rec_0515_onnx" / "inference.onnx"
CHAR_DICT = MODEL_DIR / "rec_char_dict.txt"

DEMO_IMAGE = ROOT / "assets" / "general_ocr_002.png"


def main() -> None:
    # ── Check models ──
    missing = []
    for path, name in [(DET_MODEL, "detection"), (REC_MODEL, "recognition"), (CHAR_DICT, "dictionary")]:
        if not path.exists():
            missing.append(f"  - {name}: {path}")
    if missing:
        print("Error: Model files not found. Please download models first:")
        print("\n".join(missing))
        print("\nDownload instructions: https://github.com/.../README.md#quick-start")
        sys.exit(1)

    # ── Check image ──
    image_path = DEMO_IMAGE if DEMO_IMAGE.exists() else None
    if image_path is None:
        test_dir = ROOT / "test_images"
        pngs = sorted(test_dir.glob("*.png")) if test_dir.exists() else []
        image_path = pngs[0] if pngs else None
    if image_path is None:
        print("Error: No test image found. Place a PNG in assets/ or test_images/.")
        sys.exit(1)

    print(f"Image: {image_path.name}")
    print(f"Det model: {DET_MODEL.name}")
    print(f"Rec model: {REC_MODEL.name}")
    print()

    # ── Load image ──
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        sys.exit(1)
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # ── Run OCR ──
    print("Running OCR...")
    with PPOCRv6Onnx(
        det_model_path=str(DET_MODEL),
        rec_model_path=str(REC_MODEL),
        rec_char_dict_path=str(CHAR_DICT),
    ) as ocr:
        results: list[OCRResult] = ocr(img)

    # ── Print results ──
    print(f"\nDetected {len(results)} text regions:\n")
    for i, r in enumerate(results):
        print(f"  [{i+1:2d}]  {r.text:<40s}  {r.score:.4f}")

    # ── Save visualization ──
    vis = img.copy()
    for r in results:
        pts = np.array(r.box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    out_path = Path(__file__).resolve().parent / "output_vis.png"
    cv2.imwrite(str(out_path), vis)
    print(f"\nVisualization saved to: {out_path}")


if __name__ == "__main__":
    main()