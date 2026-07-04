# AlgorithmPlatform

基于 Python 的装配动作识别与并发推理平台原型。

当前 V1 聚焦 **识别侧**，不接入大语言模型：

- 使用 Assembly101 作为主数据集；
- 使用 VideoMAE 训练短时动作识别模型；
- 使用滑动窗口 + SOP 状态机进行实时动作切分；
- 使用本地视频文件模拟多路摄像头，实现并发推理；
- 支持单视频推理、多路并发推理、异常事件输出和证据片段保存。

> 说明：Assembly101 数据集体量较大，且下载方式可能随官方维护调整。本仓库提供数据目录规范、manifest 转换器和训练/推理入口，不在仓库内保存原始数据。

## 1. 安装

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
pip install -e .
```

## 2. 数据准备

建议目录：

```text
data/
  raw/assembly101/
    videos/
    annotations/
  processed/assembly101/
    manifest.csv
    label_map.json
```

先按 Assembly101 官方页面申请/下载数据，然后执行：

```bash
python scripts/prepare_assembly101.py \
  --videos-root data/raw/assembly101/videos \
  --annotations data/raw/assembly101/annotations/actions.csv \
  --output-dir data/processed/assembly101 \
  --video-col video_path \
  --start-col start \
  --end-col end \
  --label-col action
```

如果官方标注列名不同，修改命令里的列名即可。脚本会输出统一格式 `manifest.csv`。

统一 manifest 格式：

```csv
video_path,start,end,label,split,station_id,sop_id,employee_id
/path/to/video.mp4,1.20,4.80,take,train,station_001,sop_basic,emp_001
```

## 3. 训练

```bash
python scripts/train.py --config configs/train_assembly101.yaml
```

输出：

```text
outputs/checkpoints/videomae_assembly_v1/
  config.json
  model.safetensors / pytorch_model.bin
  preprocessor_config.json
  label_map.json
  train_metrics.jsonl
```

## 4. 单视频推理

```bash
python scripts/infer_single.py --config configs/infer_single.yaml
```

输出：

```text
outputs/inference/single_predictions.csv
outputs/events/single_events.jsonl
outputs/clips/
```

## 5. 多路并发推理模拟

```bash
python scripts/infer_multi_sim.py --config configs/infer_multi.yaml
```

多路视频流在 `configs/infer_multi.yaml` 中配置。每一路视频会绑定到一个工位、员工和 SOP；调度器会把多路窗口合成 batch，再送入共享动作识别模型推理。

## 6. V1 算法定版

- 动作识别：VideoMAE-B，HuggingFace Transformers 实现；
- 在线切分：滑动窗口 + 置信度平滑 + 每工位 FSM；
- 并发推理：多视频源模拟 + batch inference；
- SOP 判断：配置化状态机；
- 后续扩展：YOLO/RT-DETR 目标检测、MS-TCN/ASFormer 离线长视频分割、RTMPose 姿态风险识别。

## 7. 当前边界

V1 可以验证完整识别平台和并发推理链路，但不能直接用于真实机箱装配上线。真实上线前必须采集公司现场数据，并重新标注/微调模型。
