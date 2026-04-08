# label-02631 — VGG16 动物图像分类

基于 VGG16 预训练模型的 4 类动物（cat、dog、tiger、lion）图像分类项目。
支持 Docker 一键运行，中文图表无乱码，训练完成后自动评估和调优。

---

## 快速开始：Docker 一键运行（推荐）

只需一条命令，零手动配置：

```bash
# 构建镜像并启动完整流水线
docker-compose up --build

# 或后台运行
docker-compose up --build -d
docker logs -f animal-classifier
```

流水线自动执行：数据准备 → VGG16 训练(15轮) → 评估 → 调优(如需) → 退出

训练完成后容器自动退出，所有结果保存在本地 `./outputs/` 目录。

> 镜像已内置 660 张初始数据集图片（4 类各 165 张），无需额外准备数据。
> VGG16 预训练权重（~528MB）在构建镜像时预下载并缓存，首次构建需要等待下载，后续构建和运行无需重复下载。
> 图表中文标题、标签使用 Noto Sans CJK 字体渲染，不会出现方块乱码。
> Docker 容器内自动使用单进程数据加载（`num_workers=0`），并配置 2GB 共享内存，避免多进程加载崩溃。

### Docker 内运行单元测试

```bash
docker compose --profile test run --rm test
```

预期结果：`206 passed`，无 failed。

### 清理环境

```bash
docker-compose down
# 如需清理训练产物
# rm -rf outputs/
```

---

## 本地运行

```bash
# 1. 创建虚拟环境
python3 -m venv venv && source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 一键运行完整流水线
python run_pipeline.py
```

分步运行：

```bash
# 数据准备（从验证集抽取 1/3 到测试集）
python prepare_data.py                          # 使用已有数据
python prepare_data.py --allow-synthetic        # 无数据时生成合成数据集

# 训练
python train.py --epochs 15 --tag baseline

# 评估（替换为实际输出目录）
python evaluate.py --model outputs/baseline_XXXXXXXX_XXXXXX/best_model.pth --test-dir dataset/test

# 调优（准确率 < 85% 时）
python retrain.py --baseline-dir outputs/baseline_XXXXXXXX_XXXXXX
```

### 本地运行测试

```bash
python -m pytest tests/ -v
```

---

## 质检验收指南

本节面向质检人员，提供完整的验收测试流程。

### 一、环境准备

```bash
docker --version          # 需要 Docker 20.10+
docker compose version    # 需要 Docker Compose v2+
```

### 二、一键启动验证

```bash
docker-compose up --build
```

日志中应依次出现以下阶段：

| 阶段 | 关键日志 | 说明 |
|------|---------|------|
| 环境检查 | `依赖检查通过` | Python 版本和依赖包正常 |
| 数据准备 | `数据集统计摘要` + 四类数量表格 | 4 类动物数据就绪 |
| 模型训练 | `Epoch [15/15]` | 完成 15 轮训练 |
| 模型评估 | `总体准确率 (Accuracy)` | 输出准确率/召回率/F1 |
| 混淆分析 | `混淆矩阵分析报告已保存` | 生成分析报告 |
| 调优判断 | `准确率 >= 85%` 或 `开始调优` | 根据准确率决定是否调优 |
| 流水线完成 | `流水线执行完毕` | 全部步骤完成 |

### 三、运行单元测试

```bash
docker compose --profile test run --rm test
```

预期结果：`206 passed`，无 failed。

### 四、检查交付产物

流水线完成后，检查 `./outputs/` 目录：

```bash
ls outputs/baseline_*/
```

交付产物清单：

| 文件 | 说明 | 验收要点 |
|------|------|---------|
| `dataset/dataset_summary.txt` | 数据集各文件夹图片数量统计 | 包含 4 类动物，train/val/test 三个分区 |
| `best_model.pth` | 最佳模型权重 | 文件大小 > 0 |
| `training_curves.png` | 训练损失和准确率曲线 | 图中包含 train/val 两条曲线，中文标题无乱码 |
| `confusion_matrix.png` | 混淆矩阵可视化 | 包含原始数值和归一化两张子图，中文标签无乱码 |
| `per_class_metrics.png` | 每类精确率/召回率/F1 对比 | 4 类动物均有柱状图 |
| `test_metrics.json` | 评估指标 JSON | 包含 accuracy、confusion_analysis 字段 |
| `confusion_analysis.txt` | 混淆矩阵分析报告 | 包含每类准确率、混淆对、原因分析、改进建议 |
| `gradcam/` | Grad-CAM 热力图 | 错误样本的模型关注区域可视化 |
| `errors/` | 错误分类样本 | 子目录格式为 `{真实类}_as_{预测类}/` |

### 五、关键指标验证

```bash
# 在容器内查看（或直接查看 outputs 目录下的 JSON）
cat outputs/baseline_*/test_metrics.json | python3 -m json.tool
```

关注字段：`accuracy`、`precision_macro`、`recall_macro`、`f1_macro`。

### 六、数据集完整性验证

```bash
cat dataset/dataset_summary.txt
```

预期数据分布：

| 类别 | 训练集 | 验证集 | 测试集（从 val 抽取 1/3） |
|------|--------|--------|--------------------------|
| cat | 120 | 30 | 15 |
| dog | 120 | 30 | 15 |
| tiger | 120 | 30 | 15 |
| lion | 120 | 30 | 15 |
| 合计 | 480 | 120 | 60 |

### 七、中文字体验证

打开 `outputs/baseline_*/training_curves.png` 和 `confusion_matrix.png`，确认：
- 图表标题、坐标轴标签为正常中文（非方块/问号）
- 负号 `-` 正常显示（非方块）

---

## 项目结构

```
├── prepare_data.py      # 数据准备（生成数据集 + 抽取测试集）
├── train.py             # VGG16 模型训练
├── evaluate.py          # 模型评估（准确率/召回率/F1/混淆矩阵/Grad-CAM）
├── retrain.py           # 调优重训练
├── run_pipeline.py      # 一键运行完整流水线
├── log_config.py        # 统一日志配置
├── font_config.py       # Matplotlib 中文字体配置（防乱码）
├── requirements.txt     # Python 依赖
├── Dockerfile           # 多阶段构建，内置中文字体 + 数据集 + 预训练权重
├── docker-compose.yml   # Docker Compose 编排（含测试服务，配置 2GB 共享内存）
├── tests/               # 测试用例（206 个）
│   ├── conftest.py
│   ├── test_prepare_data.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_retrain.py
│   └── test_pipeline.py
├── dataset/             # 数据集（660 张图片，已内置于镜像）
│   ├── train/           # 训练集（480 张）
│   ├── val/             # 验证集（120 张）
│   ├── test/            # 测试集（60 张，从 val 抽取 1/3）
│   └── dataset_summary.txt
└── outputs/             # 训练输出（运行后生成）
```

## 模型架构

- 基础模型：VGG16（ImageNet 预训练权重）
- 分类头：4096 → 256 → 4
- 冻结策略：冻结特征提取层，解冻最后 4 层微调

## 调优策略

若基线准确率 < 85%，自动按以下顺序尝试：

1. 降低学习率 (0.001 → 0.0005) + CosineAnnealing + 20 轮
2. 切换 SGD + 高动量 + 25 轮
3. 解冻更多层 + 极低学习率 + 30 轮

目标：准确率至少提升 2%（即使最终未达 85%，提升 ≥ 2% 即视为调优成功）。

## Services

| 服务 | 说明 | 启动方式 |
|------|------|---------|
| animal-classifier | 完整训练流水线 | `docker-compose up --build` |
| test | 单元测试（206 个用例） | `docker compose --profile test run --rm test` |

> 本项目为 ML 训练任务，非 Web 服务，无对外端口。容器完成后自动退出。

## 测试账号

本项目为机器学习训练项目，无需登录账号。

## CLI 参数参考

#### prepare_data.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-root` | `dataset` | 数据集根目录 |
| `--seed` | `42` | 随机种子 |
| `--test-ratio` | `0.333` | 从验证集抽取到测试集的比例 |
| `--allow-synthetic` | `False` | 无数据时生成合成数据集 |

#### train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | `15` | 训练轮数 |
| `--lr` | `0.001` | 学习率 |
| `--batch-size` | `32` | 批大小 |
| `--optimizer` | `adam` | 优化器（adam/sgd） |
| `--scheduler` | `step` | 学习率调度器（step/cosine/none） |
| `--tag` | `baseline` | 实验标签 |
| `--output-dir` | `outputs` | 输出目录 |
| `--config` | - | JSON 配置文件路径 |
| `--no-pretrained` | `False` | 不使用预训练权重 |

#### evaluate.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (必填) | 模型文件路径 |
| `--test-dir` | `dataset/test` | 测试集目录 |
| `--output-dir` | 模型同目录 | 输出目录 |
| `--batch-size` | `32` | 评估批大小 |
| `--num-workers` | `2`（本地）/ `0`（Docker） | 数据加载线程数（Docker 内自动设为 0） |

#### retrain.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--baseline-dir` | (必填) | 基线模型输出目录 |
| `--target-improvement` | `2.0` | 目标准确率提升百分比 |
| `--max-attempts` | `3` | 最大尝试策略数 |

#### run_pipeline.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--accuracy-threshold` | `85.0` | 准确率达标阈值 |
| `--min-improvement` | `2.0` | 调优最低提升幅度 |
| `--epochs` | `15` | 基线训练轮数 |

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| `FileNotFoundError: 数据集目录不存在` | 先运行 `python prepare_data.py --allow-synthetic` |
| `模型文件不存在` | 检查 `outputs/` 目录下是否有 `.pth` 文件 |
| `测试集为空` | 确认 `dataset/test/` 下各类别有图片 |
| `GPU 显存不足` | 减小 `--batch-size`（如 8 或 16） |
| `依赖检查未通过` | 运行 `pip install -r requirements.txt` |
| 测试超时 | 分文件运行：`python -m pytest tests/test_xxx.py` |
| 图表中文乱码 | Docker 内已自动配置；本地需安装中文字体（如 Noto Sans CJK） |
| Docker 首次构建慢 | VGG16 预训练权重约 528MB，首次需下载；后续构建使用 Docker 缓存 |
| Docker 内预训练权重下载失败 | 自动回退为随机初始化权重继续训练，或本地加 `--no-pretrained` |
| Docker 内 DataLoader 崩溃 | 已配置 `shm_size: 2gb` 和 `num_workers=0`，通常不会出现 |

## 题目内容

数据准备：在现有训练/验证集基础上，再从验证集每类里抽 1/3 图片放到新建的 test 文件夹当测试集，最终数据集包含 4 类动物（cat、dog、tiger、lion）。

模型训练与评估：用 VGG 模型训练至少 15 轮，训练后增加召回率、F1 分数等指标评估，还要生成测试结果的混淆矩阵并分析原因。

调优要求：若测试准确率 ≥ 85% 则完成；若 ≤ 85% 需调整参数重新训练，使准确率至少提升 2%。同时需说明每个新文件夹的图片数量。
