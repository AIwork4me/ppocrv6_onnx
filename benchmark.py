"""
Performance benchmark: pure ONNX Runtime vs PaddleX (ORT backend) vs PaddleX (Paddle backend).

Requires PaddleX to be installed for the PaddleX backends.
Pure ORT benchmark runs standalone.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# ── Try importing PaddleX (optional) ──
_paddlex_available = False
try:
    from paddlex import create_pipeline
    _paddlex_available = True
except ImportError:
    pass

# ── Our pure ORT module ──
sys.path.insert(0, str(ROOT))
from ppocrv6_onnx import PPOCRv6Onnx, OCRResult

WARMUP = 3
ITERATIONS = 10


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def bench_ort(ocr: PPOCRv6Onnx, img: np.ndarray) -> tuple[list[OCRResult], float, float]:
    """Pure ORT inference + benchmark."""
    for _ in range(WARMUP):
        ocr(img)
    times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        results = ocr(img)
        times.append(time.perf_counter() - t0)
    return results, float(np.mean(times)), float(np.std(times))


def bench_paddlex(pipeline: Any, img_path: str) -> tuple[list[dict], float, float]:
    """PaddleX inference + benchmark."""
    for _ in range(WARMUP):
        list(pipeline.predict(input=img_path))
    times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        output = list(pipeline.predict(input=img_path))
        times.append(time.perf_counter() - t0)

    results: list[dict] = []
    for res in output:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        for i in range(len(texts)):
            results.append({"text": texts[i], "score": float(scores[i])})
    return results, float(np.mean(times)), float(np.std(times))


def fmt_stat(mean_s: float, std_s: float) -> str:
    return f"{mean_s * 1000:.1f}ms ± {std_s * 1000:.1f}ms"


def compare_results(
    label_a: str, ra: list[Any], label_b: str, rb: list[Any],
) -> bool:
    """Compare two result lists. Items can be OCRResult or dict."""
    if len(ra) != len(rb):
        print(f"  {label_a} vs {label_b}: rows {len(ra)} ≠ {len(rb)}  ✗")
        return False
    diffs = 0
    score_diffs = []
    for i in range(len(ra)):
        ta = ra[i].text if hasattr(ra[i], "text") else ra[i]["text"]
        tb = rb[i].text if hasattr(rb[i], "text") else rb[i]["text"]
        sa = ra[i].score if hasattr(ra[i], "score") else ra[i]["score"]
        sb = rb[i].score if hasattr(rb[i], "score") else rb[i]["score"]
        if ta != tb:
            diffs += 1
        score_diffs.append(abs(sa - sb))
    sd = np.array(score_diffs)
    status = "✓" if diffs == 0 else f"✗ ({diffs} diff)"
    print(f"  {label_a} vs {label_b}:  text={status}  avg Δ={sd.mean():.2e}  max Δ={sd.max():.2e}")
    return diffs == 0


def main() -> None:
    # ── Find test images ──
    image_dir = ROOT / "test_images"
    demo_img = ROOT / "assets" / "general_ocr_002.png"
    test_files: list[Path] = []
    if demo_img.exists():
        test_files.append(demo_img)
    if image_dir.exists():
        test_files.extend(sorted(image_dir.glob("*.png")))
    if not test_files:
        print("No test images found in assets/ or test_images/")
        return

    print("=" * 100)
    label = "纯 ORT  vs  PaddleX(ORT)  vs  PaddleX(Paddle)" if _paddlex_available else "纯 ORT  (standalone)"
    print(f" PP-OCRv6 Benchmark: {label}")
    print("=" * 100)
    print(f" Images: {len(test_files)}   Warmup: {WARMUP}   Iters: {ITERATIONS}")
    print()

    # ── Init ORT ──
    det_dir = ROOT / "models" / "PP-OCRv6_tiny_det_onnx"
    rec_dir = ROOT / "models" / "PP-OCRv6_tiny_rec_0515_onnx"
    if not (det_dir / "inference.onnx").exists() or not (rec_dir / "inference.onnx").exists():
        print("Error: ONNX models not found. Download them first — see README.")
        sys.exit(1)

    ort_ocr = PPOCRv6Onnx(
        det_model_path=str(det_dir / "inference.onnx"),
        rec_model_path=str(rec_dir / "inference.onnx"),
        rec_char_dict_path=str(ROOT / "models" / "rec_char_dict.txt"),
    )

    # ── Init PaddleX backends (optional) ──
    px_ort = None
    px_paddle = None
    if _paddlex_available:
        ort_cfg = ROOT / "OCR_onnx.yaml"
        paddle_cfg = ROOT / "OCR_paddle.yaml"
        if ort_cfg.exists() and paddle_cfg.exists():
            px_ort = create_pipeline(pipeline=str(ort_cfg), engine="onnxruntime")
            px_paddle = create_pipeline(pipeline=str(paddle_cfg), engine="paddle")
        else:
            print("Note: OCR_onnx.yaml / OCR_paddle.yaml not found, skipping PaddleX backends.\n")
    else:
        print("Note: PaddleX not installed, running pure ORT benchmark only.\n")
        print("      To compare with PaddleX: pip install paddlex && python benchmark.py\n")

    # ── Per-image benchmark ──
    n_cols = 3 if px_ort else 1
    header = f" {'Image':<28s} {'#':>3s}  {'Pure ORT':<22s}"
    if px_ort:
        header += f"  {'PX(ORT)':<22s}  {'PX(Paddle)':<22s}"
    print(header)
    print("─" * (40 + n_cols * 24))

    all_ort: list[OCRResult] = []
    all_pxort: list[dict] = []
    all_pxpaddle: list[dict] = []
    ort_t: list[float] = []
    pxort_t: list[float] = []
    pxpaddle_t: list[float] = []

    for img_file in test_files:
        img_path = str(img_file)
        img = load_image(img_path)

        r1, m1, s1 = bench_ort(ort_ocr, img)
        all_ort.extend(r1)
        ort_t.append(m1)

        row = f" {img_file.name:<28s} {len(r1):>3d}  {fmt_stat(m1, s1):<22s}"

        if px_ort and px_paddle:
            r2, m2, s2 = bench_paddlex(px_ort, img_path)
            r3, m3, s3 = bench_paddlex(px_paddle, img_path)
            all_pxort.extend(r2)
            all_pxpaddle.extend(r3)
            pxort_t.append(m2)
            pxpaddle_t.append(m3)
            row += f"  {fmt_stat(m2, s2):<22s}  {fmt_stat(m3, s3):<22s}"

        print(row)

    # ── Precision comparison ──
    if px_ort:
        print(f"\n{'='*100}")
        compare_results("Pure ORT   ", all_ort, "PX(ORT)    ", all_pxort)
        compare_results("Pure ORT   ", all_ort, "PX(Paddle) ", all_pxpaddle)
        compare_results("PX(ORT)    ", all_pxort, "PX(Paddle) ", all_pxpaddle)
        print(f"\nTotal lines compared: {len(all_ort)}")

    # ── Speed summary ──
    print(f"\n{'─'*100}")
    o = np.array(ort_t) * 1000
    print(f"  Pure ORT        avg: {o.mean():.1f}ms  (range: {o.min():.1f} ~ {o.max():.1f})")
    if px_ort:
        po = np.array(pxort_t) * 1000
        pp = np.array(pxpaddle_t) * 1000
        print(f"  PaddleX(ORT)    avg: {po.mean():.1f}ms  (range: {po.min():.1f} ~ {po.max():.1f})  ({po.mean()/o.mean():.2f}x)")
        print(f"  PaddleX(Paddle) avg: {pp.mean():.1f}ms  (range: {pp.min():.1f} ~ {pp.max():.1f})  ({pp.mean()/o.mean():.2f}x)")
    print(f"  {'='*100}")

    ort_ocr.close()


if __name__ == "__main__":
    main()