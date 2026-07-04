# Assembly101 数据集下载指南（重要）

⚠️ 不需要全量下载（3.89TB）。当前项目只需要**中等子集训练数据**。

---

# 1. 推荐方案（Hugging Face）⭐

官方已将数据迁移至 Hugging Face：

Dataset: cvml-nus/assembly101

特点：
- 支持按需下载
- 不需要 Google Drive 权限脚本
- 更适合做子集实验

---

## 1.1 登录
```bash
pip install huggingface_hub
huggingface-cli login
```

---

## 1.2 只下载 annotations（必须）
```bash
huggingface-cli download cvml-nus/assembly101 \
  --repo-type dataset \
  --local-dir data/raw/assembly101 \
  --include "annotations/*"
```

---

## 1.3 按需下载部分视频（关键）

先选择 recording：
```bash
python scripts/select_assembly101_recordings.py \
  --annotations data/raw/assembly101/annotations.csv \
  --labels take,place,attach \
  --max-recordings 80
```

得到：
```text
outputs/dataset/selected_recordings.txt
```

---

然后按 recording 下载（示例）：
```bash
huggingface-cli download cvml-nus/assembly101 \
  --repo-type dataset \
  --local-dir data/raw/assembly101 \
  --include "recordings/<recording_name>/*"
```

⚠️ 只下载 selected_recordings.txt 里的 recording

---

# 2. 备选方案（Google Drive 官方脚本）

如果你已经拿到 Drive 权限：

## 2.1 下载单个视频
```bash
python download.py --videos <recording_name>
```

---

## 2.2 下载部分视角
```bash
python download.py --videos <recording_name> --views fixed
```

可选：
- all
- fixed
- egocentric
- v1~v8
- e1~e4

---

# 3. 强烈建议（工程策略）

❌ 不要下载全部数据
❌ 不要一次性解压全量

✔ 推荐结构：

```text
data/raw/assembly101/
  annotations/
  recordings/
      (80个左右录制片段)
```

---

# 4. 推荐最终规模（用于训练）

```text
80 recordings
5–8 actions
5k–15k clips
```

---

# 5. 总结

你现在的目标不是“下载数据集”，而是：

> 构建一个可训练的子集数据管道

全量数据 ≠ 更好模型
合适子集 = 可运行系统
