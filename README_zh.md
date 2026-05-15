# PP-OCRv6 ONNX Runtime 推理

> **纯 ONNX Runtime** 实现 PP-OCRv6。零 PaddlePaddle 依赖。精度 bit-exact。速度提升 3 倍。

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/onnxruntime-1.26%2B-orange)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![English](https://img.shields.io/badge/English-README-blue)](README.md)

## 为什么用这个库

PaddleX 功能强大——但它需要 **2 GB+ 的依赖链**（PaddlePaddle + 数十个包）。如果你只需要用 ONNX Runtime 做 PP-OCRv6 推理，完全不需要安装整个 Paddle 生态。

本项目提供一个**单文件、即插即用**的模块，在同一 ONNX Runtime 后端上**逐字对齐** PaddleX 的识别结果，同时比 PaddlePaddle 原生推理**快 3 倍**。

| | PaddleX (Paddle) | PaddleX (ORT) | **本项目 (纯 ORT)** |
|---|:---:|:---:|:---:|
| 依赖 | PaddlePaddle + PaddleX | PaddlePaddle + PaddleX | onnxruntime + opencv + numpy + pyclipper |
| 平均耗时 (M4) | 877 ms | 291 ms | **282 ms** |
| vs Paddle 原生 | 基准 | 快 3.0× | **快 3.1×** |
| 精度 vs PaddleX | — | bit-exact | **bit-exact** |

---

## 目录

- [快速开始](#-快速开始)
- [性能基准](#-性能基准)
- [精度验证](#-精度验证)
- [API 参考](#-api-参考)
- [流水线架构](#-流水线架构)
- [项目结构](#-项目结构)
- [模型规格](#-模型规格)
- [高级用法：GPU / CoreML](#-高级用法gpu--coreml)

---

## 快速开始

### 1. 安装依赖

```bash
pip install onnxruntime opencv-python numpy pyclipper
```

仅此而已。不需要 PaddlePaddle。不需要 PaddleX。

### 2. 下载模型

```bash
# 检测模型（ONNX）
wget -c https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_tiny_det_onnx.tar
tar xf PP-OCRv6_tiny_det_onnx.tar

# 识别模型（ONNX）
wget -c https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_tiny_rec_0515_onnx.tar
tar xf PP-OCRv6_tiny_rec_0515_onnx.tar
```

解压后的目录结构：

```
PP-OCRv6_tiny_det_onnx/
├── inference.onnx
└── inference.yml

PP-OCRv6_tiny_rec_0515_onnx/
├── inference.onnx
└── inference.yml
```

### 3. 运行推理

```python
import cv2
from ppocrv6_onnx import PPOCRv6Onnx, OCRResult

det_model = "PP-OCRv6_tiny_det_onnx/inference.onnx"
rec_model = "PP-OCRv6_tiny_rec_0515_onnx/inference.onnx"
char_dict = "models/rec_char_dict.txt"          # 仓库自带

with PPOCRv6Onnx(det_model, rec_model, char_dict) as ocr:
    img = cv2.imread("your_image.png")
    results: list[OCRResult] = ocr(img)

    for r in results:
        print(f"{r.text}  ({r.score:.3f})")
```

### 4. 运行 Demo

```bash
python demo.py
```

---

## 性能基准

测试环境：**Apple M4**，8 张不同类型图片（中文、英文、日文、手写、杂志、竖排）。

| 图片 | 行数 | 纯 ORT | PaddleX (ORT) | PaddleX (Paddle) |
|------|:---:|--------|---------------|-------------------|
| 登机牌 | 30 | 162 ms | 163 ms | 527 ms |
| 古汉语 | 12 | 112 ms | 113 ms | 343 ms |
| 手写中文 | 10 | 57 ms | 56 ms | 170 ms |
| 手写英文 | 11 | 69 ms | 76 ms | 209 ms |
| 日文 | 28 | 429 ms | 434 ms | 1,312 ms |
| 杂志 | 63 | 697 ms | 707 ms | 2,106 ms |
| 竖排文字 | 76 | 563 ms | 575 ms | 1,711 ms |
| 拼音 | 38 | 216 ms | 217 ms | 619 ms |
| **平均** | | **282 ms** | **291 ms** | **875 ms** |

> 运行基准测试：`python benchmark.py`

---

## 精度验证

与 PaddleX 对比，8 张图片、268 行文本：

| 指标 | 数值 |
|------|-------|
| 文本不一致行数 | **0 / 268** |
| 平均置信度差值 | 2.5 × 10⁻⁷ |
| 最大置信度差值 | 1.9 × 10⁻⁶ |
| 检测框像素偏差 | 0.0 px |

使用同一 ONNX Runtime 后端时，与 PaddleX 完全 bit-exact 对齐。

> 验证脚本：`scripts/verify_ort_vs_paddlex.py`、`scripts/verify_batch.py`

---

## API 参考

### `PPOCRv6Onnx`

```python
class PPOCRv6Onnx:
    def __init__(
        self,
        det_model_path: str,           # 检测 ONNX 模型路径
        rec_model_path: str,           # 识别 ONNX 模型路径
        rec_char_dict_path: str,       # 字符字典路径
        *,
        det_thresh: float = 0.3,       # 检测二值化阈值
        det_box_thresh: float = 0.6,   # 检测框置信度阈值
        det_unclip_ratio: float = 1.5, # 文本框扩展比例
        rec_batch_size: int = 6,       # 识别批大小
        prefer_accelerator: bool = False,  # 启用 CoreML / CUDA
    ) -> None: ...

    def __call__(self, img_bgr: np.ndarray) -> list[OCRResult]:
        """对 BGR 图像执行完整 OCR 流程。"""

    def detect(self, img_bgr: np.ndarray) -> tuple[np.ndarray, list[float]]:
        """仅文本检测。返回 (boxes, scores)。"""

    def recognize(self, img_list: list[np.ndarray]) -> tuple[list[str], list[float]]:
        """仅文本识别。返回 (texts, scores)。"""

    def close(self) -> None:
        """释放资源，幂等。"""
```

### `OCRResult`

```python
@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str                # 识别文本
    score: float             # 置信度 [0, 1]
    box: list[list[int]]     # 四顶点坐标 [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
```

---

## 流水线架构

```
输入图片 (BGR)
    │
    ▼
[文本检测]      DB 模型 → 缩放 → 归一化 → ONNX 推理 → DB 后处理
    │
    ▼
[框排序]        按阅读顺序（从上到下、从左到右）
    │
    ▼
[区域裁剪]      透视变换逐个裁剪文本区域
    │
    ▼
[文本识别]      CRNN 模型 → 缩放(高48) → 归一化 → ONNX 推理 → CTC 解码
    │
    ▼
OCRResult(text, score, box)[]
```

---

## 项目结构

```
ppocrv6_onnx/
├── ppocrv6_onnx.py          # 主模块（~870 行，核心代码）
├── demo.py                  # 快速体验 Demo
├── benchmark.py             # 性能对比工具
├── pyproject.toml           # 项目元数据与依赖声明
├── README.md                # 英文文档
├── README_zh.md             # 本文档（中文）
├── LICENSE                  # MIT 许可证
├── models/                  # ONNX 模型目录（需下载）
│   └── rec_char_dict.txt    # 7180 字符字典（已包含）
├── assets/                  # 示例图片
├── test_images/             # 多语言测试图片集
└── scripts/                 # 验证与开发工具
```

---

## 模型规格

PP-OCRv6 提供三种规格，替换下载 URL 即可：

| 规格 | 检测模型 | 识别模型 |
|------|----------|----------|
| **tiny**（小） | `PP-OCRv6_tiny_det_onnx.tar` | `PP-OCRv6_tiny_rec_0515_onnx.tar` |
| **small**（中） | `PP-OCRv6_small_det_onnx.tar` | `PP-OCRv6_small_rec_0515_onnx.tar` |
| **medium**（大） | `PP-OCRv6_medium_det_onnx.tar` | `PP-OCRv6_medium_rec_0515_onnx.tar` |

基础 URL：`https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/`

---

## 高级用法：启用 GPU / CoreML 加速

```python
# Apple Silicon CoreML 加速
ocr = PPOCRv6Onnx(
    det_model_path=...,
    rec_model_path=...,
    rec_char_dict_path=...,
    prefer_accelerator=True,   # ⬅ 启用硬件加速
)
```

> **注意：** 加速器可能因硬件特定的浮点优化产生微小的置信度差异，文本结果保持不变。

---

## 致谢

- **PP-OCRv6** — 百度 PaddlePaddle 团队
- **Differentiable Binarization** — Liao et al.
- **CRNN + CTC** — Shi et al.
- 实现参考 [PaddleX](https://github.com/PaddlePaddle/PaddleX)

---

## 许可证

MIT