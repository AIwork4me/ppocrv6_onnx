# 模型适配方法论二：跨引擎数值漂移与其应对

> **场景**：将模型推理从任意框架迁移到不同的推理引擎（如 PaddleInference → ONNX Runtime、TensorRT → OpenVINO）。由于不同引擎对浮点运算的实现不同，**bit-exact 无法保证**。本文讨论如何在这种条件下证明适配成功。

---

## 为什么会出现数值漂移？

同一份 ONNX 模型权重，输入同一个预处理后的 tensor，不同引擎输出不同——原因：

| 来源 | 说明 | 典型数量级 |
|------|------|:---:|
| **卷积实现差异** | Intel MKL-DNN vs Apple BNNS vs NVIDIA cuDNN 内部对 `im2col + GEMM` 的瓦片大小、累加顺序不同 | 1e-5 |
| **激活函数精度** | `sigmoid(x) = 1/(1+exp(-x))` 在不同库中的多项式近似阶数不同 | 1e-6 |
| **算子融合** | 引擎可能将 `Conv + BN + ReLU` 融合成一个内核，改变中间精度 | 1e-6 |
| **FMA 行为** | `a*b + c` 是一次 FMA 还是 `tmp=a*b; tmp+c` 两次舍入，取决于硬件 | 1e-7 |
| **线程并行度** | 多线程下的浮点加法 `sum(a,b,c,d)` 的归约树结构不同，累加顺序不同 | 1e-7 |
| **内存对齐** | 不同的内存布局( NCHW vs NHWC )导致向量化路径不同 | 1e-6 |

**核心原因**：IEEE 754 保证单次运算的结果，但**不保证运算顺序**。不同引擎改变了顺序，累积误差就不同。

---

## 本次实测数据

Apple M4，同一 ONNX 模型权重，不同推理引擎：

```
PaddleInference（Paddle 原生）:
  "登机牌"  score=0.9996210336685181   ← Paddle 的浮点计算结果

ONNX Runtime（CPU provider）:
  "登机牌"  score=0.9996210336685181   ← ORT 的浮点计算结果，恰好一致

ONNX Runtime（CoreML provider）:
  "登机牌"  score=0.9996120333671569   ← CoreML 的浮点结果，差 ~1e-5
```

| 对比 | 文本 | 平均分差 | 最大分差 | 是否 bit-exact |
|------|:---:|:---:|:---:|:---:|
| PaddleInference vs ORT(CPU) | ✓ 一致 | 1.2e-5 | 2.6e-3 | ✗ |
| ORT(CPU) vs ORT(CoreML) | ✓ 一致 | ~1e-5 | ~1e-3 | ✗ |
| ORT(CPU) vs ORT(CPU) | ✓ 一致 | 0.0 | 0.0 | ✓ |

**关键观察**：
- 文本结果（基于 argmax / 阈值判断）是**鲁棒的**——1e-5 量级的概率差异不足以改变 argmax 结果
- 置信度分数（基于 softmax 概率值）对数值漂移敏感——这是正常的，**不能**作为验证失败的证据

---

## 应对策略

### 策略一：分层验证——分离"硬判定"和"软分数"

不要用置信度分数来判断是否正确。用 argmax 的结果来判断。

| 验证层 | 方法 | 容差 | 判定正确 |
|--------|------|:---:|----------|
| 检测框数量 | `len(boxes_before) == len(boxes_after)` | 0 | 完全相等 |
| 检测框坐标 | `abs(box_a - box_b).max()` | < 1 px | 像素级一致 |
| 识别文本 | `text_a == text_b` | 0 | 完全相等 |
| 置信度分数 | `abs(score_a - score_b)` | < 1e-3 | 显著一致 |

**正确做法**：
```python
def verify_cross_engine(ort_results, paddle_results, tolerance=1e-3):
    assert len(ort_results) == len(paddle_results), "Box count mismatch"

    for r1, r2 in zip(ort_results, paddle_results):
        assert r1.text == r2.text, f"Text mismatch: {r1.text} ≠ {r2.text}"
        score_diff = abs(r1.score - r2.score)
        assert score_diff < tolerance, f"Score diff {score_diff:.2e} > {tolerance}"

        box_diff = np.abs(np.array(r1.box) - np.array(r2.box)).max()
        assert box_diff < 1, f"Box diff {box_diff:.1f} px"

    print(f"✓ Cross-engine OK: {len(ort_results)} texts match, max score diff < {tolerance}")
```

### 策略二：多图统计——建立"正负基线"

单张图的分数差异可能是偶然的。用多张图片建立统计分布。

```python
# 跑 N 张图片，收集分数差异
diffs = []
for img in test_set:
    r_paddle = paddle_infer(img)
    r_ort = ort_infer(img)
    for r1, r2 in zip(r_paddle, r_ort):
        diffs.append(abs(r1.score - r2.score))

print(f"  Mean diff: {np.mean(diffs):.2e}")
print(f"  Max diff:  {np.max(diffs):.2e}")
print(f"  P99 diff:  {np.percentile(diffs, 99):.2e}")
```

**判定标准**：
- Max diff < 1e-2 → 跨引擎精度可接受
- Mean diff < 1e-4 → 精度优秀
- 任何 diff > 0.01 且文本不一致 → 可能存在引擎 bug，需排查

### 策略三：模型级验证——针对敏感模型的特殊处理

某些模型对数值漂移特别敏感：

| 模型类型 | 风险 | 应对 |
|----------|------|------|
| **DETR / Transformer** | Softmax + 自注意力对精度敏感 | 验证检测框 IoU > 0.99 |
| **Beam Search** | 分数微小变化可能导致 beam 路径分叉 | 验证 Top-1 beam 一致，Top-K 放宽 |
| **GAN** | 生成图像像素值可能整体偏移 | 验证 PSNR > 40dB |
| **LLM** | Token 级差异逐级放大 | 验证困惑度 (perplexity) 偏差 < 1% |
| **CTR 模型** | 概率值用于排序决策 | 验证 AUC 偏差 < 0.001 |

### 策略四：硬件锁版本——处理确定性

如果跨引擎差异不可接受（如金融、医疗场景），锁定推理环境：

```python
# 强制 ONNX Runtime 使用确定性计算
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"],
    provider_options=[{"arena_extend_strategy": "kSameAsRequested"}],
)

# 设置环境变量
import os
os.environ["ORT_DISABLE_ALL"] = "1"        # 禁用所有非确定性优化
os.environ["OMP_NUM_THREADS"] = "1"        # 单线程
```

---

## 判定矩阵：什么算"适配成功"

| 验证项目 | 同引擎 | 跨引擎 | 格雷区 |
|----------|:---:|:---:|:---:|
| 文本 100% 一致 | **必须** | **必须** | — |
| 检测框 < 1px 差 | **必须** | 强烈建议 | < 2px 可讨论 |
| 置信度 diff == 0 | **必须（bit-exact）** | 不要求 | — |
| 置信度 diff < 1e-3 | — | **必须** | 1e-3 ~ 1e-2 需人工审核 |
| 置信度 diff > 1e-2 | **失败** | **失败** | 排查引擎 bug |

---

## 一句话

**跨引擎没有 bit-exact。取而代之的是统计显著性：在足够多的测试样本上，argmax 结果一致，置信度分散在已知容差范围内，且不存在系统性偏差。**