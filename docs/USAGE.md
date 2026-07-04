# AlgorithmPlatform 使用指南（V1识别侧）

该版本为**视频动作识别 + SOP状态机 + 并发推理模拟系统**，基于 Assembly101 等公开数据集。

---

# 1. 环境准备

## 1.1 创建环境
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

## 1.2 安装依赖
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
    annotations/
  processed/assembly101/
    manifest.csv
    label_map.json
```

## 2.2 转换数据集

将官方标注转为统一格式：

```bash
python scripts/prepare_assembly101.py \
  --videos-root data/raw/assembly101/videos \
  --annotations data/raw/assembly101/annotations.csv \
  --output-dir data/processed/assembly101
```

生成：
- manifest.csv
- label_map.json

---

# 3. 训练模型（VideoMAE）

```bash
python scripts/train.py --config configs/train_assembly101.yaml
```

输出模型：
```
outputs/checkpoints/videomae_assembly_v1/
```

---

# 4. 单视频推理

## 运行方式
```bash
python scripts/infer_single.py --config configs/infer_single.yaml
```

## 输出结果
```text
outputs/inference/single_predictions.csv
outputs/events/single_events.jsonl
outputs/clips/
```

---

# 5. 多路并发推理（模拟摄像头）

## 运行方式
```bash
python scripts/infer_multi_sim.py --config configs/infer_multi.yaml
```

## 说明
- 每个视频 = 一个“模拟摄像头流”
- 每个流绑定 station + employee + SOP
- 多线程模拟并发

输出：
- multi_predictions.csv
- multi_events.jsonl
- clips/

---

# 6. 系统核心机制

## 6.1 动作识别
- VideoMAE（滑动窗口3秒）
- 每0.5秒推理一次

## 6.2 SOP状态机
- 每个工位独立FSM
- 判断步骤顺序 / 漏步骤 / 错步骤

## 6.3 并发推理
- 多视频流模拟
- 共享模型
- batch inference（逻辑层）

## 6.4 异常检测
- DEVIATION（流程错误）
- UNCERTAIN（低置信度）
- POSSIBLE SKIP（跳步骤）

---

# 7. 当前能力边界

## 可以做
- 动作分类（take / place / screw / inspect）
- SOP流程识别（伪SOP）
- 多视频并发推理模拟
- 异常事件记录

## 不支持（当前版本）
- 真实工厂机箱精细识别
- 工具级高精度检测
- 实时工业部署

---

# 8. 运行建议顺序

```text
Step 1：prepare_assembly101.py
Step 2：train.py
Step 3：infer_single.py
Step 4：infer_multi_sim.py
```

---

# 9. 常见问题

## Q1：视频路径找不到
检查 manifest.csv 中 video_path 是否正确

## Q2：显存不足
降低 batch_size 或 num_frames

## Q3：推理很慢
这是 VideoMAE正常情况，可换 X3D/TSM

---

# 10. 总结

该系统是一个：

> 视频动作识别 + SOP状态机 + 并发流处理 的工业原型系统
