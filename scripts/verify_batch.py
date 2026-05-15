"""
批量精度对比：纯 ONNXRuntime API vs PaddleX（多图片）。

Usage: cd to project root, then:
    python scripts/verify_batch.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ppocrv6_onnx import PPOCRv6Onnx, OCRResult

try:
    from paddlex import create_pipeline
except ImportError:
    print("Error: PaddleX is not installed. This script requires PaddleX.")
    sys.exit(1)


def main() -> None:
    yaml_path = ROOT / "OCR_onnx.yaml"
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found.")
        print("Generate it with: paddlex --get_pipeline_config OCR --save_path ./")
        sys.exit(1)

    # Find images
    image_files: list[Path] = []
    for d in [ROOT / "assets", ROOT / "test_images"]:
        if d.exists():
            image_files.extend(sorted(d.glob("*.png")))

    if not image_files:
        print("No PNG images found in assets/ or test_images/")
        return

    print(f"{'='*70}")
    print(f" 批量精度对比：纯 ONNXRuntime API vs PaddleX")
    print(f" 图片数量: {len(image_files)}")
    print(f"{'='*70}")

    ort_ocr = PPOCRv6Onnx(
        det_model_path=str(ROOT / "models" / "PP-OCRv6_tiny_det_onnx" / "inference.onnx"),
        rec_model_path=str(ROOT / "models" / "PP-OCRv6_tiny_rec_0515_onnx" / "inference.onnx"),
        rec_char_dict_path=str(ROOT / "models" / "rec_char_dict.txt"),
    )
    px_pipeline = create_pipeline(pipeline=str(yaml_path), engine="onnxruntime")

    all_aligned = True
    summary: list[dict] = []

    for img_file in image_files:
        img_path = str(img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"\n  ⚠ 无法读取: {img_file.name}")
            continue

        ort_results: list[OCRResult] = ort_ocr(img)
        px_output = px_pipeline.predict(input=img_path)
        px_results: list[dict] = []
        for res in px_output:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            for i in range(len(texts)):
                px_results.append({"text": texts[i], "score": float(scores[i])})

        n_ort, n_px = len(ort_results), len(px_results)
        count_match = n_ort == n_px
        if not count_match:
            all_aligned = False

        texts_diff = 0
        scores_list: list[float] = []
        for i in range(min(n_ort, n_px)):
            if ort_results[i].text != px_results[i]["text"]:
                texts_diff += 1
            scores_list.append(abs(ort_results[i].score - px_results[i]["score"]))

        avg_sd = np.mean(scores_list) if scores_list else 0.0
        max_sd = np.max(scores_list) if scores_list else 0.0
        img_aligned = count_match and texts_diff == 0

        status = "✓" if img_aligned else "✗"
        print(
            f"  {status} {img_file.name:<25s} "
            f"ORT={n_ort:>2d}  PaddleX={n_px:>2d}  "
            f"文本差异={texts_diff}  avg Δ={avg_sd:.2e}  max Δ={max_sd:.2e}"
        )

        if not img_aligned:
            all_aligned = False
            for i in range(max(n_ort, n_px)):
                o = ort_results[i] if i < n_ort else None
                p = px_results[i] if i < n_px else None
                if o and p and o.text != p["text"]:
                    print(f"         [{i}] ORT=\"{o.text}\"  vs  PX=\"{p['text']}\"")

        summary.append({
            "file": img_file.name,
            "ort_n": n_ort, "px_n": n_px,
            "text_diff": texts_diff,
            "aligned": img_aligned,
        })

    ort_ocr.close()

    print(f"\n{'='*70}")
    aligned_count = sum(1 for s in summary if s["aligned"])
    print(f"  结果: {aligned_count}/{len(summary)} 图片精度对齐")
    if all_aligned:
        print(f"  结论: ✓ 纯 ORT 与 PaddleX 精度在所有图片上完全对齐")
    else:
        print(f"  结论: ✗ 部分图片存在差异")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()