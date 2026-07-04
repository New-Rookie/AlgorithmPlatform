# Assembly101 子集训练策略

本项目 V1 的目标不是追求 Assembly101 全量 SOTA，而是训练出一个**能进行有效推理的动作识别模型**，并验证单路/并发推理链路。

因此不建议使用极小子集，也不建议一开始全量训练。

---

## 1. 推荐数据规模

### Smoke Test（仅验证代码）
不推荐作为正式模型：

```text
视频数：5-10
类别数：2-3
clip数：200-500
用途：确认代码不报错
```

### Medium Subset（推荐第一版正式训练）
推荐作为当前默认训练规模：

```text
视频数：50-100
类别数：5-8
每类clip数：至少500，最好1000+
总clip数：5000-15000
用途：训练一个可用于推理演示的模型
```

### Large Subset（后续增强）
```text
视频数：200-500
类别数：10-20
总clip数：30000+
用途：提高泛化能力
```

---

## 2. 操作步骤

### Step 1：先查看标注分布

```bash
python scripts/inspect_assembly101_annotations.py \
  --annotations data/raw/assembly101/annotations.csv \
  --video-col video_path \
  --label-col action \
  --top-k 30
```

输出会显示每个动作类别的 clip 数和覆盖视频数。

---

### Step 2：选择 5-8 个高频且接近装配流程的类别

优先选择类似以下语义的动作：

```text
take / pick / grab
place / put / position
align
attach / insert / connect / assemble
screw / fix / tighten
inspect / check
```

具体类别名必须以 `inspect_assembly101_annotations.py` 输出为准。

---

### Step 3：生成 Medium Subset manifest

示例：

```bash
python scripts/prepare_assembly101.py \
  --videos-root data/raw/assembly101/videos \
  --annotations data/raw/assembly101/annotations.csv \
  --output-dir data/processed/assembly101 \
  --video-col video_path \
  --start-col start \
  --end-col end \
  --label-col action \
  --labels take,place,attach,insert,screw,inspect \
  --max-videos 80 \
  --max-rows 12000 \
  --balanced \
  --require-existing-videos
```

注意：`--labels` 必须改成真实标注文件中的类别名。

---

## 3. 正式训练命令

```bash
python scripts/train.py --config configs/train_assembly101_medium.yaml
```

输出：

```text
outputs/checkpoints/videomae_assembly_medium_v1/
```

---

## 4. 为什么不用太小子集

太小子集只能验证代码，不能训练出有效模型。典型问题：

```text
类别覆盖不足
模型只记住背景和个别视频
验证准确率虚高
换一个视频推理就失效
SOP状态机接收到的动作结果不稳定
```

---

## 5. 为什么不一开始全量

全量 Assembly101 会显著增加下载、预处理和训练成本。对于当前阶段，主要目标是先得到一个可用模型，并验证单推理和并发推理流程。

---

## 6. 当前建议结论

第一版正式模型建议使用：

```text
50-100 个视频
5-8 个动作类别
5000-15000 个动作clip
VideoMAE-B
训练 10-15 epoch
```

这比极小子集更有推理意义，也比全量训练更可控。
