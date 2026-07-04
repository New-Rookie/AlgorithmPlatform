# AlgorithmPlatform 使用指南（V1识别侧）

该版本为**视频动作识别 + SOP状态机 + 并发推理模拟系统**，基于 Assembly101 等公开数据集。

当前目标不是用极小样本验证代码，而是使用 Assembly101 的**中等子集**训练出一个具备基本推理价值的模型。

---

# 1. 环境准备

## 1.1 创建环境
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

## 1.2 先安装 PyTorch

PyTorch 需要按 CPU / CUDA 环境单独安装，不放在 requirements.txt 里。

CPU：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA CUDA 11.8：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 1.3 安装项目依赖
```bash
pip install -r requirements.txt
pip install -e .
```

---

# 2. 数据集准备（Assembly101）

## 2.1 数据结构
```text
data/
  raw/assembly101/
    videos/
    annotations.csv
  processed/assembly101/
    manifest.csv
    label_map.json
```

---

## 2.2 先检查标注分布

不要盲目全量训练，也不要用太小子集。先看有哪些动作类别、每类多少 clip。

```bash
python scripts/inspect_assembly101_annotations.py \
  --annotations data/raw/assembly101/annotations.csv \
  --video-col video_path \
  --label-col action \
  --top-k 30
```

输出：

```text
outputs/dataset/annotation_summary.csv
```

---

## 2.3 生成中等训练子集

推荐第一版正式训练规模：

```text
视频数：50-100
类别数：5-8
总clip数：5000-15000
```

示例命令：

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

注意：`--labels` 里的类别必须替换成你标注文件里的真实类别名。

生成：
- manifest.csv
- label_map.json

---

# 3. 训练模型（VideoMAE）

推荐用中等子集配置：

```bash
python scripts/train.py --config configs/train_assembly101_medium.yaml
```

输出模型：

```text
outputs/checkpoints/videomae_assembly_medium_v1/
```

如果只是验证代码是否能跑，可以使用轻量配置：

```bash
python scripts/train.py --config configs/train_assembly101.yaml
```

---

# 4. 单视频推理

先把 `configs/infer_single.yaml` 里的模型路径改成实际训练输出，例如：

```yaml
paths:
  model_dir: outputs/checkpoints/videomae_assembly_medium_v1
```

然后运行：

```bash
python scripts/infer_single.py --config configs/infer_single.yaml
```

输出结果：

```text
outputs/inference/single_predictions.csv
outputs/events/single_events.jsonl
outputs/clips/
```

---

# 5. 多路并发推理（模拟摄像头）

先把 `configs/infer_multi.yaml` 里的模型路径改成实际训练输出：

```yaml
paths:
  model_dir: outputs/checkpoints/videomae_assembly_medium_v1
```

然后配置多个本地视频：

```yaml
streams:
  - stream_id: stream_001
    video_path: data/demo/stream_001.mp4
    station_id: station_001
    employee_id: emp_001
    sop_id: sop_basic
```

运行：

```bash
python scripts/infer_multi_sim.py --config configs/infer_multi.yaml
```

说明：
- 每个视频 = 一个模拟摄像头流
- 每个流绑定 station + employee + SOP
- 每个流维护独立 FSM
- 当前版本用多线程模拟并发

输出：
- multi_events.jsonl
- 后续可扩展 multi_predictions.csv

---

# 6. 系统核心机制

## 6.1 动作识别
- VideoMAE
- 3秒滑动窗口
- 0.5秒步长

## 6.2 SOP状态机
- 每个工位独立FSM
- 判断步骤顺序 / 漏步骤 / 错步骤

## 6.3 并发推理
- 多视频流模拟
- 共享模型
- 每路独立状态机

## 6.4 异常检测
- DEVIATION：流程错误
- UNCERTAIN：低置信度
- POSSIBLE SKIP：疑似跳步骤

---

# 7. 当前能力边界

## 可以做
- 基于公开数据集训练动作分类模型
- 单视频动作识别推理
- 多视频并发推理模拟
- SOP流程识别（伪SOP）
- 异常事件记录

## 不支持
- 真实工厂机箱精细识别
- 真实工具/零件检测
- 工业上线级准确率

---

# 8. 运行建议顺序

```text
Step 1：inspect_assembly101_annotations.py
Step 2：prepare_assembly101.py 生成中等子集
Step 3：train.py 训练 VideoMAE
Step 4：infer_single.py 单视频推理
Step 5：infer_multi_sim.py 多路模拟并发推理
```

---

# 9. 常见问题

## Q1：视频路径找不到
检查 `manifest.csv` 中 `video_path` 是否指向真实本地视频。

## Q2：显存不足
降低 `batch_size` 或 `num_frames`，或者用梯度累积。

## Q3：推理很慢
VideoMAE 是相对重的模型。当前优先保证模型有效性，后续再考虑 X3D/TSM 轻量化。

## Q4：子集多大合适
不要用过小子集。第一版建议 50-100 个视频、5-8 个动作类别、5000-15000 个 clip。

---

# 10. 总结

该系统是一个：

> 视频动作识别 + SOP状态机 + 并发流处理 的工业原型系统
