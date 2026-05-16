# 模型适配方法论一：源框架 → 目标运行时（同引擎适配）

> **场景**：将模型推理从源框架（PaddleX / PaddlePaddle / PyTorch / TensorFlow）迁移到目标运行时（ONNX Runtime / TensorRT / OpenVINO），使用同一推理引擎，追求 **bit-exact** 输出。

---

## 核心公理

**模型的正确性 = 预处理 × 权重 × 后处理**。权重是共享的——同一个 ONNX 文件。只要预处理和后处理在数值上逐行对齐，输出就 bit-exact。

## 6 步法

### Step 1: 画出完整数据流

从源框架的 `pipeline.predict()` 入口开始，追踪每一层变换，直到输出。记录每一步的 shape、dtype、参数。

**产出**：一张 ASCII 流程图 + 每步的 (shape, dtype, params)。

```
Image (640,480,3) uint8
    │ Resize: limit_side_len=64, limit_type="min", round/32
    ▼
Resized (512,896,3) uint8
    │ Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], scale=1/255
    ▼
Normalized (512,896,3) float32
    │ ToCHW: transpose(2,0,1)
    ▼
Tensor (1,3,512,896) float32
    │ ONNX Inference
    ▼
Prob map (1,1,512,896) float32
    │ DBPostProcess: thresh=0.3, box_thresh=0.6, unclip_ratio=1.5
    ▼
Boxes (N,4,2) int16
```

### Step 2: 提取预处理参数

不要猜测。从源框架的配置文件和 `__init__` 代码中精确提取每一个参数。

**必须提取的信息**：

| 类别 | 参数 | 从哪里找 |
|------|------|----------|
| **图像读取** | 颜色格式 (BGR / RGB / Gray) | `ReadImage(format=...)` |
| **Resize** | 目标尺寸策略、对齐粒度、插值方法 | `DetResizeForTest.__init__` 或 `inference.yml` |
| **归一化** | mean、std、scale、公式 | `NormalizeImage.__init__` |
| **通道转换** | HWC → CHW、是否需要 cvtColor | `ToCHWImage` |
| **Batch** | padding 方向、填充值 | `ToBatch` 或 `np.pad` |

**反例（本次实践中踩过的坑）**：
- `NormalizeImage` 的 `order="chw"` 参数让它在内部先转 CHW 做归一化再转回 HWC，然后外层的 ToCHW 再转一次——两次转置相当于直接逐通道操作。如果不理解这个，你会多转一次。
- 识别模型的归一化是 `(x/255 - 0.5) / 0.5`，不是 ImageNet 标准归一化。两个模型在同一项目中用的归一化公式不同。

### Step 3: 验证预处理（L1 对齐）

**目标**：源框架和目标运行时产生的预处理 tensor 逐元素完全相等。

```python
# 在源框架中截取预处理输出
src_tensor = det_model.pre_tfs("image.png")  # shape (1,3,H,W)

# 在自己的实现中
my_tensor = my_det_preprocess("image.png")   # shape (1,3,H,W)

# 验证
diff = np.abs(src_tensor - my_tensor).max()
assert diff == 0.0, f"Preprocess mismatch: max diff = {diff}"
```

**如果 diff > 0**：
- 检查 resize 的顺序（先 ratio 还是先 pad）
- 检查 mean/std 的通道顺序（BGR 顺序 ≠ RGB 顺序）
- 检查归一化公式（`(x/255 - mean)/std` vs `(x - mean*255)/(std*255)`）
- 检查 int→float 转换的时机

### Step 4: 提取后处理参数

后处理往往比预处理复杂——有分支逻辑、有第三方库调用、有硬编码常量。

**必须提取的信息**：

| 后处理模块 | 关键参数 | 注意事项 |
|------------|----------|----------|
| DBPostProcess | thresh, box_thresh, unclip_ratio, min_size, max_candidates | `pyclipper.AddPath` 对 float32 和 int32 行为不同 |
| 框排序 | 排序策略 (y 优先 / x 优先)、同行判定阈值 | PaddleX 用 `abs(y[j+1] - y[j]) < 10` 判同行 |
| 区域裁剪 | 裁剪方式 (minAreaRect / 透视变换)、旋转判定 | 高宽比 ≥ 1.5 时 `np.rot90` |
| CTC Decode | blank token 位置、重复合并规则、字典映射 | `blank` 在 index 0，重复合并用 `seq[1:] != seq[:-1]` |

**反例（本次实践中踩过的坑）**：

```python
# 错误：强制转 int32 导致 pyclipper 用整数精度
box_i32 = box.astype(np.int32)
po.AddPath(box_i32, ...)   # 亚像素精度丢失！

# 正确：保持原始 dtype（float32）
po.AddPath(box, ...)        # float32 亚像素精度，与 PaddleX 一致
```

### Step 5: 端到端对齐（L2 + L3 验证）

**L2 — 单图验证**：
```
图片数: 1    文本行数一致: ✓    逐行文本: 30/30 一致    置信度差: 0.0
```

**L3 — 多图多样化验证**：
```
图片数: 8+   场景覆盖: 中文/英文/日文/手写/竖排/杂志
文本行总数: 268    差异: 0    结论: bit-exact 验证通过
```

**验证脚本模板**：
```python
def verify(ort_results, src_results):
    assert len(ort_results) == len(src_results)
    for r1, r2 in zip(ort_results, src_results):
        assert r1.text == r2.text, f"Text mismatch: {r1.text} vs {r2.text}"
        assert r1.score == r2.score, f"Score mismatch: {r1.score} vs {r2.score}"
    print("✓ Bit-exact verification passed")
```

### Step 6: 性能基线

同一硬件、同一引擎下，预期性能持平（±5%）。如果显著退化，检查：
- 不必要的 `np.array` 拷贝
- Python for 循环替代了 numpy 向量化操作
- 重复的 ONNX Session 查询（`get_inputs()[0].name` 应缓存）

---

## 检查清单

- [ ] 数据流图已画出，每步的 shape/dtype/params 已标注
- [ ] 预处理 L1 对齐：`max(|src - ort|) == 0.0`
- [ ] 所有后处理参数从源码提取，未猜测
- [ ] 第三方库调用行为已验证（dtype 敏感、返回值变化）
- [ ] L2 单图验证：文本 + 分数 100% 一致
- [ ] L3 多图验证：≥8 张多样图片，0 差异
- [ ] 性能基线：同引擎下 ±5% 以内

---

## 一句话

**适配 = 读懂 + 复现 + 对齐 + 证明。** 每一步都可以用 `max | diff | == 0.0` 量化验证。