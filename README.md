# PP-OCRv6 ONNX Runtime Inference

> **Pure ONNX Runtime** implementation of PP-OCRv6. Zero PaddlePaddle. Bit-exact accuracy. 3× faster.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/onnxruntime-1.26%2B-orange)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![中文](https://img.shields.io/badge/中文-README_zh-red)](README_zh.md)

## Why this exists

PaddleX works great — but it pulls in a **2 GB+ dependency chain** (PaddlePaddle + dozens of packages). If all you need is PP-OCRv6 inference with ONNX Runtime, you shouldn't have to install the entire Paddle ecosystem.

This repo gives you a **single-file, drop-in module** that matches PaddleX output **bit-for-bit** on the same ONNX Runtime backend — while being **3× faster** than PaddlePaddle native inference.

| | PaddleX (Paddle) | PaddleX (ORT) | **This repo (Pure ORT)** |
|---|:---:|:---:|:---:|
| Inference engine | PaddleInference | ONNX Runtime | ONNX Runtime |
| Requires PaddlePaddle | ✓ Yes (~2 GB) | ✓ Yes (~2 GB) | ✗ No |
| Package footprint | ~2 GB | ~2 GB | ~200 MB |
| Avg latency (M4) | 877 ms | 291 ms | **282 ms** |
| vs Paddle native | baseline | 3.0× faster | **3.1× faster** |
| Accuracy vs PaddleX | — | bit-exact | **bit-exact** |

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Benchmarks](#-benchmarks)
- [Precision Validation](#-precision-validation)
- [API Reference](#-api-reference)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Model Variants](#-model-variants)
- [Advanced: GPU / CoreML](#-advanced-gpu--coreml)

---

## Quick Start

### 1. Install dependencies

```bash
pip install onnxruntime opencv-python numpy pyclipper
```

That's it. No PaddlePaddle. No PaddleX.

### 2. Download models

```bash
# Detection model (ONNX)
wget -c https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_tiny_det_onnx.tar
tar xf PP-OCRv6_tiny_det_onnx.tar

# Recognition model (ONNX)
wget -c https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_tiny_rec_0515_onnx.tar
tar xf PP-OCRv6_tiny_rec_0515_onnx.tar
```

After extraction, your directory should look like:

```
PP-OCRv6_tiny_det_onnx/
├── inference.onnx
└── inference.yml

PP-OCRv6_tiny_rec_0515_onnx/
├── inference.onnx
└── inference.yml
```

### 3. Run inference

```python
import cv2
from ppocrv6_onnx import PPOCRv6Onnx, OCRResult

det_model = "PP-OCRv6_tiny_det_onnx/inference.onnx"
rec_model = "PP-OCRv6_tiny_rec_0515_onnx/inference.onnx"
char_dict = "models/rec_char_dict.txt"          # included in this repo

with PPOCRv6Onnx(det_model, rec_model, char_dict) as ocr:
    img = cv2.imread("your_image.png")
    results: list[OCRResult] = ocr(img)

    for r in results:
        print(f"{r.text}  ({r.score:.3f})")
```

### 4. Try the demo

```bash
python demo.py
```

---

## Benchmarks

Tested on **Apple M4**, 8 diverse images (Chinese, English, Japanese, handwriting, magazine, vertical text).

| Image | Lines | Pure ORT | PaddleX (ORT) | PaddleX (Paddle) |
|-------|:-----:|----------|---------------|-------------------|
| boarding pass | 30 | 162 ms | 163 ms | 527 ms |
| ancient Chinese | 12 | 112 ms | 113 ms | 343 ms |
| handwriting (CN) | 10 | 57 ms | 56 ms | 170 ms |
| handwriting (EN) | 11 | 69 ms | 76 ms | 209 ms |
| Japanese | 28 | 429 ms | 434 ms | 1,312 ms |
| magazine | 63 | 697 ms | 707 ms | 2,106 ms |
| vertical text | 76 | 563 ms | 575 ms | 1,711 ms |
| pinyin | 38 | 216 ms | 217 ms | 619 ms |
| **Average** | | **282 ms** | **291 ms** | **875 ms** |

> Run: `python benchmark.py`

---

## Precision Validation

Compared with PaddleX across **8 images / 268 text lines**:

| Metric | Value |
|--------|-------|
| Text mismatches | **0 / 268** |
| Avg confidence diff | 2.5 × 10⁻⁷ |
| Max confidence diff | 1.9 × 10⁻⁶ |
| Box coordinate diff | 0.0 px |

All results **bit-exact** with PaddleX when using the same ONNX Runtime backend.

> Verification scripts: `scripts/verify_ort_vs_paddlex.py`, `scripts/verify_batch.py`

---

## API Reference

### `PPOCRv6Onnx`

```python
class PPOCRv6Onnx:
    def __init__(
        self,
        det_model_path: str,           # Path to detection ONNX model
        rec_model_path: str,           # Path to recognition ONNX model
        rec_char_dict_path: str,       # Path to character dictionary
        *,
        det_thresh: float = 0.3,       # Detection binarization threshold
        det_box_thresh: float = 0.6,   # Detection box confidence threshold
        det_unclip_ratio: float = 1.5, # Box expansion ratio
        rec_batch_size: int = 6,       # Recognition batch size
        prefer_accelerator: bool = False,  # Enable CoreML / CUDA
    ) -> None: ...

    def __call__(self, img_bgr: np.ndarray) -> list[OCRResult]:
        """Run full OCR pipeline on a BGR image."""

    def detect(self, img_bgr: np.ndarray) -> tuple[np.ndarray, list[float]]:
        """Text detection only. Returns (boxes, scores)."""

    def recognize(self, img_list: list[np.ndarray]) -> tuple[list[str], list[float]]:
        """Text recognition only. Returns (texts, scores)."""

    def close(self) -> None:
        """Release resources. Idempotent."""
```

### `OCRResult`

```python
@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str                # Recognized text
    score: float             # Confidence [0, 1]
    box: list[list[int]]     # 4 corners [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
```

---

## Pipeline Architecture

```
Input Image (BGR)
    │
    ▼
[Text Detection]     DB model → Resize → Normalize → ONNX → DBPostProcess
    │
    ▼
[Box Sorting]        Top-to-bottom, left-to-right reading order
    │
    ▼
[Region Cropping]    Perspective transform for each text box
    │
    ▼
[Text Recognition]   CRNN model → Resize(H=48) → Normalize → ONNX → CTC Decode
    │
    ▼
OCRResult(text, score, box)[]
```

---

## Project Structure

```
ppocrv6_onnx/
├── ppocrv6_onnx.py          # Main module (~870 lines, all you need)
├── demo.py                  # Quick-start demo script
├── benchmark.py             # Performance comparison tool
├── pyproject.toml           # Project metadata & dependencies
├── README.md                # This file (English)
├── README_zh.md             # Chinese documentation
├── LICENSE                  # MIT License
├── models/                  # Downloaded ONNX models
│   └── rec_char_dict.txt    # 7180-character dictionary (included)
├── assets/                  # Demo image
├── test_images/             # Multi-lingual test suite
├── scripts/                 # Verification & dev tools
└── docs/                    # Methodology & guides
```

---

## Model Variants

PP-OCRv6 offers three sizes. Replace URLs accordingly:

| Size | Det Model | Rec Model |
|------|-----------|-----------|
| **tiny** | `PP-OCRv6_tiny_det_onnx.tar` | `PP-OCRv6_tiny_rec_0515_onnx.tar` |
| **small** | `PP-OCRv6_small_det_onnx.tar` | `PP-OCRv6_small_rec_0515_onnx.tar` |
| **medium** | `PP-OCRv6_medium_det_onnx.tar` | `PP-OCRv6_medium_rec_0515_onnx.tar` |

Base URL: `https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/`

---

## Advanced: GPU / CoreML

```python
# Apple Silicon CoreML acceleration
ocr = PPOCRv6Onnx(
    det_model_path=...,
    rec_model_path=...,
    rec_char_dict_path=...,
    prefer_accelerator=True,   # ⬅ enable hardware acceleration
)
```

> **Note:** Accelerators may produce slightly different confidence scores due to hardware-specific floating-point optimizations. Text results remain identical.

---

## Credits

- **PP-OCRv6** by Baidu PaddlePaddle team
- **Differentiable Binarization** — Liao et al.
- **CRNN + CTC** — Shi et al.
- Implementation inspired by [PaddleX](https://github.com/PaddlePaddle/PaddleX)

---

## License

MIT