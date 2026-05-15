"""
PP-OCRv6 纯 ONNXRuntime API vs PaddleX 精度对比验证（需要 PaddleX）。

Usage: cd to project root, then:
    python scripts/verify_ort_vs_paddlex.py
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
    print("Error: PaddleX is not installed. This script requires PaddleX for comparison.")
    print("Install: pip install paddlepaddle && pip install paddlex")
    sys.exit(1)


def run_paddlex(image_path: str, yaml_path: str) -> list[dict]:
    pipeline = create_pipeline(pipeline=yaml_path, engine="onnxruntime")
    output = pipeline.predict(input=image_path)
    results = []
    for res in output:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        polys = res.get("rec_polys", [])
        for i in range(len(texts)):
            results.append({
                "text": texts[i],
                "score": float(scores[i]),
                "poly": np.array(polys[i]).tolist() if i < len(polys) else None,
            })
    return results


def run_pure_onnx(image_path: str) -> list[OCRResult]:
    with PPOCRv6Onnx(
        det_model_path=str(ROOT / "models" / "PP-OCRv6_tiny_det_onnx" / "inference.onnx"),
        rec_model_path=str(ROOT / "models" / "PP-OCRv6_tiny_rec_0515_onnx" / "inference.onnx"),
        rec_char_dict_path=str(ROOT / "models" / "rec_char_dict.txt"),
    ) as ocr:
        img = cv2.imread(image_path)
        return ocr(img)


def compare_results(paddlex_results: list[dict], onnx_results: list[OCRResult]) -> bool:
    print(f"\n{'='*70}")
    print(f" 精度对比验证: 纯 ONNXRuntime API vs PaddleX")
    print(f"{'='*70}")
    print(f"  PaddleX 识别文本数: {len(paddlex_results)}")
    print(f"  纯ORT    识别文本数: {len(onnx_results)}")

    if len(paddlex_results) != len(onnx_results):
        print(f"  ⚠ 检测文本行数不一致!")
        return False

    print(f"  ✓ 检测文本行数一致: {len(paddlex_results)} 行\n")

    all_text_match = True
    score_diffs: list[float] = []
    box_diffs: list[float] = []

    for i in range(len(paddlex_results)):
        p = paddlex_results[i]
        o = onnx_results[i]
        text_match = p["text"] == o.text
        score_diff = abs(p["score"] - o.score)
        score_diffs.append(score_diff)
        if not text_match:
            all_text_match = False
        if p.get("poly") is not None and o.box is not None:
            p_poly = np.array(p["poly"])
            o_box = np.array(o.box)
            if p_poly.shape == o_box.shape:
                box_diffs.append(float(np.mean(np.abs(p_poly.astype(float) - o_box.astype(float)))))
        if not text_match or score_diff >= 0.01:
            print(f"  [{i+1:2d}] PaddleX=\"{p['text']}\"  vs  ORT=\"{o.text}\"  Δ={score_diff:.8f}")

    print(f"\n  --- 统计 ---")
    print(f"  文本一致性: {all_text_match}")
    print(f"  平均置信度差值: {np.mean(score_diffs):.8f}")
    print(f"  最大置信度差值: {np.max(score_diffs):.8f}")
    if box_diffs:
        print(f"  平均检测框像素差: {np.mean(box_diffs):.4f}")
        print(f"  最大检测框像素差: {np.max(box_diffs):.4f}")

    print(f"\n{'='*70}")
    if all_text_match and max(score_diffs) < 0.001:
        print("  结论: ✓ 纯 ONNXRuntime API 与 PaddleX 推理精度完全对齐")
    else:
        print("  结论: ✗ 存在差异，需进一步排查")
    print(f"{'='*70}")
    return all_text_match and max(score_diffs) < 0.01


if __name__ == "__main__":
    # Find the OCR config file
    yaml_path = ROOT / "OCR_onnx.yaml"
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found. This config is needed for PaddleX comparison.")
        print("Generate it with: paddlex --get_pipeline_config OCR --save_path ./")
        sys.exit(1)

    test_image = str(ROOT / "assets" / "general_ocr_002.png")
    print("===== PP-OCRv6 纯 ONNXRuntime API vs PaddleX 精度对比 =====\n")
    print("[1/2] 运行 PaddleX 推理...")
    px_results = run_paddlex(test_image, str(yaml_path))
    print("[2/2] 运行纯 ONNXRuntime API 推理...")
    ort_results = run_pure_onnx(test_image)
    compare_results(px_results, ort_results)