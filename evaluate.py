"""
VGG16 动物图像分类 - 评估脚本
功能：
1. 在测试集上评估模型
2. 计算准确率、召回率、精确率、F1 分数
3. 生成混淆矩阵并分析原因
4. 输出详细的分类报告
"""

import os
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np

from log_config import setup_logger
import font_config  # noqa: F401 — 确保中文字体配置生效

logger = setup_logger(__name__, "evaluation.log")

IMG_SIZE = 224


def load_model(model_path: str, device: torch.device) -> tuple:
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    class_to_idx = checkpoint.get("class_to_idx", {})
    dropout = config.get("dropout", 0.5)

    # 动态确定类别数：从 class_to_idx 读取，回退到 4
    num_classes = len(class_to_idx) if class_to_idx else 4

    model = models.vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(4096, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    val_acc = checkpoint.get("val_accuracy", "unknown")
    logger.info(f"模型加载成功 - Epoch: {epoch}, Val Acc: {val_acc}")

    return model, class_to_idx, config


def get_test_transform():
    """测试集数据变换"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_all(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple:
    """对测试集进行预测"""
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    """计算详细的评估指标"""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    for i, cls_name in enumerate(class_names):
        metrics["per_class"][cls_name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(np.sum(y_true == i)),
        }

    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """绘制混淆矩阵"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 原始数值混淆矩阵
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title("Confusion Matrix (Counts)")

    # 归一化混淆矩阵
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"混淆矩阵已保存: {output_path}")


def generate_confusion_matrix_and_analyze(
    cm: np.ndarray,
    class_names: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    output_dir: Path,
) -> str:
    """
    生成混淆矩阵并分析原因（对应 Prompt 要求："生成测试结果的混淆矩阵并分析原因"）。

    本函数将"生成混淆矩阵"和"分析混淆原因"合并为一个不可分割的操作：
    1. 生成混淆矩阵可视化图片 (confusion_matrix.png)
    2. 自动分析混淆原因：
       - 统计每类分类准确率，识别最弱类别
       - 找出主要混淆对（如"cat 最常被误判为 tiger，占比 X%"）
       - 基于预测置信度分析误判原因（高/中/低确信度误判的不同诊断）
       - 逐样本错误明细（Top-2 预测类别、置信度差距）
       - 生物学背景辅助解释（如"猫和老虎同属猫科，纹理相似"）
       - 整体诊断评级与针对性改进建议
    3. 输出分析报告到 confusion_analysis.txt 文件和控制台

    Args:
        cm: 混淆矩阵 (numpy array)
        class_names: 类别名称列表
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率
        output_dir: 输出目录

    Returns:
        str: 混淆矩阵原因分析报告文本
    """
    # Step 1: 生成混淆矩阵可视化
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    # Step 2: 分析混淆原因（基于预测数据的动态分析）
    analysis = analyze_confusion_matrix(cm, class_names, y_true, y_pred, y_probs)

    # Step 3: 在控制台输出分析报告
    logger.info("")
    logger.info("=" * 60)
    logger.info("混淆矩阵原因分析")
    logger.info("=" * 60)
    logger.info(f"\n{analysis}")

    # Step 4: 保存分析报告到文件
    analysis_path = output_dir / "confusion_analysis.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis)
    logger.info(f"混淆矩阵分析报告已保存: {analysis_path}")

    return analysis


def plot_per_class_metrics(metrics: dict, class_names: list, output_path: Path):
    """绘制每类指标对比图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precisions = [metrics["per_class"][c]["precision"] for c in class_names]
    recalls = [metrics["per_class"][c]["recall"] for c in class_names]
    f1s = [metrics["per_class"][c]["f1"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precisions, width, label="Precision", color="#4CAF50")
    bars2 = ax.bar(x, recalls, width, label="Recall", color="#2196F3")
    bars3 = ax.bar(x + width, f1s, width, label="F1 Score", color="#FF9800")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"每类指标图已保存: {output_path}")


def analyze_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
    y_probs: np.ndarray = None,
) -> str:
    """
    动态分析混淆矩阵。
    当提供预测概率时，会分析错误样本的置信度分布，给出数据驱动的洞察。
    """
    analysis_lines = []
    analysis_lines.append("=" * 60)
    analysis_lines.append("混淆矩阵分析报告")
    analysis_lines.append("=" * 60)

    n = len(class_names)
    row_sums = cm.sum(axis=1)
    # 安全归一化：避免除零
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    cm_normalized = cm.astype("float") / safe_sums[:, np.newaxis]

    # ── 1. 每类分类准确率 ──
    analysis_lines.append("\n1. 每类分类准确率:")
    worst_class_idx = -1
    worst_class_acc = 1.0
    for i, cls in enumerate(class_names):
        acc = cm_normalized[i, i]
        total = int(row_sums[i])
        correct = cm[i, i]
        analysis_lines.append(f"   {cls}: {acc:.2%} ({correct}/{total})")
        if total > 0 and acc < worst_class_acc:
            worst_class_acc = acc
            worst_class_idx = i

    if worst_class_idx >= 0:
        analysis_lines.append(
            f"   → 最弱类别: {class_names[worst_class_idx]} ({worst_class_acc:.2%})"
        )

    # ── 2. 主要混淆对 ──
    analysis_lines.append("\n2. 主要混淆对 (错误率 > 5%):")
    confusion_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm_normalized[i, j] > 0.05:
                confusion_pairs.append(
                    (i, j, class_names[i], class_names[j], cm_normalized[i, j], cm[i, j])
                )
    confusion_pairs.sort(key=lambda x: x[4], reverse=True)

    if not confusion_pairs:
        analysis_lines.append("   无显著混淆对 (所有错误率 < 5%)")
    else:
        for _, _, true_cls, pred_cls, rate, count in confusion_pairs:
            analysis_lines.append(f"   {true_cls} → {pred_cls}: {rate:.2%} ({count} 张)")

    # ── 3. 动态原因分析（基于实际预测数据） ──
    analysis_lines.append("\n3. 混淆原因分析:")

    if confusion_pairs and y_true is not None and y_pred is not None and y_probs is not None:
        for true_idx, pred_idx, true_cls, pred_cls, rate, count in confusion_pairs:
            analysis_lines.append(f"\n   ● {true_cls} 被误判为 {pred_cls} ({rate:.2%}, {count} 张):")

            # 找出这些错误样本
            error_mask = (y_true == true_idx) & (y_pred == pred_idx)
            error_indices = np.where(error_mask)[0]
            error_probs = y_probs[error_mask]

            if len(error_probs) > 0:
                # 错误样本对「错误类别」的平均置信度
                avg_wrong_conf = float(np.mean(error_probs[:, pred_idx]))
                max_wrong_conf = float(np.max(error_probs[:, pred_idx]))
                min_wrong_conf = float(np.min(error_probs[:, pred_idx]))

                # 错误样本对「正确类别」的平均置信度
                avg_true_conf = float(np.mean(error_probs[:, true_idx]))

                # 置信度差距
                conf_gap = avg_wrong_conf - avg_true_conf

                analysis_lines.append(
                    f"     - 错误类别平均置信度: {avg_wrong_conf:.2%} "
                    f"(范围: {min_wrong_conf:.2%} ~ {max_wrong_conf:.2%})"
                )
                analysis_lines.append(
                    f"     - 正确类别平均置信度: {avg_true_conf:.2%}"
                )
                analysis_lines.append(
                    f"     - 置信度差距: {conf_gap:+.2%}"
                )

                # ── 逐样本错误明细 ──
                analysis_lines.append(f"     - 错误样本明细 (共 {len(error_indices)} 张):")
                for rank, sample_idx in enumerate(error_indices):
                    sample_prob = y_probs[sample_idx]
                    wrong_conf = float(sample_prob[pred_idx])
                    true_conf = float(sample_prob[true_idx])
                    # 找出 top-2 预测类别
                    top2_idx = np.argsort(sample_prob)[::-1][:2]
                    top2_str = ", ".join(
                        f"{class_names[k]}={sample_prob[k]:.2%}" for k in top2_idx
                    )
                    analysis_lines.append(
                        f"       样本#{sample_idx}: "
                        f"预测→{pred_cls}({wrong_conf:.2%}), "
                        f"真实→{true_cls}({true_conf:.2%}) "
                        f"[Top2: {top2_str}]"
                    )
                    # 对每个样本给出具体观察
                    margin = wrong_conf - true_conf
                    if margin > 0.5:
                        analysis_lines.append(
                            f"         → 高确信误判(差距{margin:.0%})，"
                            f"该样本可能具有强烈的 {pred_cls} 视觉特征"
                        )
                    elif margin > 0.2:
                        analysis_lines.append(
                            f"         → 中等确信误判(差距{margin:.0%})，"
                            f"两类特征在该样本上竞争激烈"
                        )
                    else:
                        analysis_lines.append(
                            f"         → 低确信误判(差距{margin:.0%})，"
                            f"模型对该样本缺乏判别信心，建议人工复核标注"
                        )
                    # 限制输出量：超过 10 个样本时截断
                    if rank >= 9 and len(error_indices) > 10:
                        analysis_lines.append(
                            f"       ... 省略剩余 {len(error_indices) - 10} 张"
                        )
                        break

                # 根据置信度模式给出动态诊断
                if avg_wrong_conf > 0.7:
                    analysis_lines.append(
                        f"     - 诊断: 模型对错误类别高度自信 (>{avg_wrong_conf:.0%})，"
                        f"说明两类在特征空间中高度重叠，模型学到的判别特征不足以区分。"
                    )
                    analysis_lines.append(
                        f"     - 建议: 增加这两类的训练样本，或引入对比学习/Triplet Loss 拉开类间距离。"
                    )
                elif avg_wrong_conf > 0.4:
                    analysis_lines.append(
                        f"     - 诊断: 模型对错误类别中等自信 ({avg_wrong_conf:.0%})，"
                        f"存在决策边界模糊区域，部分样本特征不够显著。"
                    )
                    analysis_lines.append(
                        f"     - 建议: 加强数据增强多样性，或使用 Label Smoothing 软化决策边界。"
                    )
                else:
                    analysis_lines.append(
                        f"     - 诊断: 模型对错误类别置信度较低 ({avg_wrong_conf:.0%})，"
                        f"属于边缘误判，正确类别置信度也不高，可能是样本质量问题。"
                    )
                    analysis_lines.append(
                        f"     - 建议: 检查这些样本是否存在标注错误或图片质量问题。"
                    )

            # 补充生物学背景（作为辅助信息，非主要分析）
            bio_context = _get_biological_context(true_cls, pred_cls)
            if bio_context:
                analysis_lines.append(f"     - 生物学背景: {bio_context}")

    elif confusion_pairs:
        # 没有概率数据时，基于混淆矩阵的统计特征分析
        for _, _, true_cls, pred_cls, rate, count in confusion_pairs:
            analysis_lines.append(f"\n   ● {true_cls} 被误判为 {pred_cls} ({rate:.2%}):")

            # 检查是否双向混淆
            reverse_rate = cm_normalized[
                class_names.index(pred_cls), class_names.index(true_cls)
            ]
            if reverse_rate > 0.05:
                analysis_lines.append(
                    f"     - 发现双向混淆: {pred_cls}→{true_cls} 也有 {reverse_rate:.2%} 错误率"
                )
                analysis_lines.append(
                    f"     - 诊断: 两类存在系统性特征重叠，非单向偏差。"
                )
                analysis_lines.append(
                    f"     - 建议: 需要同时增强两类的区分性特征学习。"
                )
            else:
                analysis_lines.append(
                    f"     - 单向混淆: 仅 {true_cls} 易被误判为 {pred_cls}，反向无显著混淆。"
                )
                analysis_lines.append(
                    f"     - 诊断: {true_cls} 类可能存在部分样本与 {pred_cls} 视觉相似。"
                )
                analysis_lines.append(
                    f"     - 建议: 增加 {true_cls} 类的训练样本多样性。"
                )

            bio_context = _get_biological_context(true_cls, pred_cls)
            if bio_context:
                analysis_lines.append(f"     - 生物学背景: {bio_context}")
    else:
        analysis_lines.append("   模型分类效果良好，无显著混淆。")

    # ── 4. 整体诊断与改进建议 ──
    analysis_lines.append("\n4. 整体诊断与改进建议:")

    overall_acc = np.trace(cm) / max(cm.sum(), 1)
    analysis_lines.append(f"   整体准确率: {overall_acc:.2%}")

    if overall_acc >= 0.95:
        analysis_lines.append("   评级: 优秀 — 模型表现出色，可考虑部署。")
    elif overall_acc >= 0.85:
        analysis_lines.append("   评级: 良好 — 达到基本要求，仍有优化空间。")
    elif overall_acc >= 0.70:
        analysis_lines.append("   评级: 一般 — 需要进一步调优。")
    else:
        analysis_lines.append("   评级: 较差 — 模型需要重大改进。")

    if confusion_pairs:
        # 根据混淆对数量和严重程度给出针对性建议
        severe_pairs = [p for p in confusion_pairs if p[4] > 0.15]
        moderate_pairs = [p for p in confusion_pairs if 0.05 < p[4] <= 0.15]

        if severe_pairs:
            analysis_lines.append(f"   发现 {len(severe_pairs)} 个严重混淆对 (>15%):")
            for _, _, tc, pc, r, _ in severe_pairs:
                analysis_lines.append(f"     - {tc} ↔ {pc}: {r:.2%}")
            analysis_lines.append("   优先建议:")
            analysis_lines.append("     1. 针对严重混淆类别增加训练数据")
            analysis_lines.append("     2. 使用 Focal Loss 替代 CrossEntropyLoss")
            analysis_lines.append("     3. 考虑更深的网络架构 (ResNet50, EfficientNet)")

        if moderate_pairs:
            analysis_lines.append(f"   发现 {len(moderate_pairs)} 个中等混淆对 (5-15%):")
            analysis_lines.append("   次要建议:")
            analysis_lines.append("     1. 增强数据增强策略 (MixUp, CutMix)")
            analysis_lines.append("     2. 使用注意力机制 (SE-Net, CBAM) 关注区分性特征")
    else:
        analysis_lines.append("   当前模型无显著混淆，可考虑:")
        analysis_lines.append("     1. 在更大规模数据集上验证泛化能力")
        analysis_lines.append("     2. 测试对抗样本的鲁棒性")

    return "\n".join(analysis_lines)


def _get_biological_context(cls_a: str, cls_b: str) -> str:
    """获取两个动物类别之间的生物学背景信息（辅助参考）"""
    contexts = {
        frozenset({"cat", "tiger"}): "同属猫科(Felidae)，共享面部结构和条纹花纹基因",
        frozenset({"cat", "lion"}): "同属猫科(Felidae)，面部骨骼结构相似",
        frozenset({"tiger", "lion"}): "同属猫科大型猫亚科(Pantherinae)，体型和轮廓接近",
        frozenset({"dog", "lion"}): "部分犬种(金毛、藏獒)毛色和颈部鬃毛与狮子相似",
        frozenset({"cat", "dog"}): "作为常见宠物，部分品种在体型和毛色上有交叉",
        frozenset({"dog", "tiger"}): "部分犬种的斑纹花色与虎纹有视觉相似性",
    }
    return contexts.get(frozenset({cls_a, cls_b}), "")


def generate_gradcam(
    model: nn.Module,
    test_dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_dir: Path,
    device: torch.device,
    max_samples: int = 20,
) -> int:
    """
    对错误分类样本生成 Grad-CAM 热力图，叠加到原图上保存。
    使用 VGG16 最后一个卷积层 (features[-1]) 作为目标层。

    Args:
        model: VGG16 模型
        test_dataset: ImageFolder 测试集
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        output_dir: 输出目录
        device: 计算设备
        max_samples: 最多生成的热力图数量

    Returns:
        int: 生成的热力图数量
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # 找出错误样本索引
    error_indices = np.where(y_true != y_pred)[0]
    if len(error_indices) == 0:
        logger.info("无错误分类样本，跳过 Grad-CAM 生成。")
        return 0

    # 限制数量
    if len(error_indices) > max_samples:
        error_indices = error_indices[:max_samples]

    model.eval()

    # 注册 hook 捕获最后卷积层的激活和梯度
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    # VGG16 最后一个卷积层
    target_layer = None
    for layer in reversed(list(model.features.children())):
        if isinstance(layer, nn.Conv2d):
            target_layer = layer
            break

    if target_layer is None:
        logger.warning("未找到卷积层，跳过 Grad-CAM。")
        return 0

    # 临时禁用 inplace ReLU（避免与 backward hook 冲突）
    original_inplace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) and module.inplace:
            original_inplace[name] = True
            module.inplace = False

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    transform = get_test_transform()
    saved_count = 0

    try:
        for idx in error_indices:
            img_path = Path(test_dataset.samples[idx][0])
            true_cls = class_names[y_true[idx]]
            pred_cls = class_names[y_pred[idx]]

            # 加载原图
            try:
                orig_img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # 前向传播
            input_tensor = transform(orig_img).unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)

            output = model(input_tensor)
            pred_class_idx = output.argmax(dim=1).item()

            # 反向传播：对预测类别的得分求梯度
            model.zero_grad()
            output[0, pred_class_idx].backward()

            if "value" not in activations or "value" not in gradients:
                continue

            # 计算 Grad-CAM
            grads = gradients["value"].detach()   # [1, C, H, W]
            acts = activations["value"].detach()   # [1, C, H, W]
            weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP
            cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
            cam = torch.relu(cam)
            cam = cam.detach().squeeze().cpu().numpy()

            # 归一化到 [0, 1]
            if cam.max() > 0:
                cam = cam / cam.max()

            # 上采样到原图尺寸
            orig_w, orig_h = orig_img.size
            cam_resized = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (orig_w, orig_h), Image.BILINEAR
                )
            ) / 255.0

            # 绘制叠加图
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(orig_img)
            axes[0].set_title(f"原图\n真实: {true_cls}")
            axes[0].axis("off")

            axes[1].imshow(cam_resized, cmap="jet")
            axes[1].set_title(f"Grad-CAM\n预测: {pred_cls}")
            axes[1].axis("off")

            axes[2].imshow(orig_img)
            axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)
            axes[2].set_title(f"叠加\n{true_cls}→{pred_cls}")
            axes[2].axis("off")

            plt.tight_layout()
            save_name = f"gradcam_{idx}_{true_cls}_as_{pred_cls}.png"
            plt.savefig(gradcam_dir / save_name, dpi=100, bbox_inches="tight")
            plt.close()
            saved_count += 1

    finally:
        handle_fwd.remove()
        handle_bwd.remove()
        # 恢复 inplace ReLU 设置
        for name, module in model.named_modules():
            if name in original_inplace:
                module.inplace = True

    if saved_count > 0:
        logger.info(f"已生成 {saved_count} 张 Grad-CAM 热力图到: {gradcam_dir}")
    return saved_count


def save_error_samples(
    test_dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_dir: Path,
) -> int:
    """
    保存被错误分类的图片样本到 output_dir/errors/{true_class}_as_{pred_class}/。
    用于辅助人工分析误判原因。

    Returns:
        int: 保存的错误样本数量
    """
    import shutil

    errors_dir = output_dir / "errors"
    saved_count = 0

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            true_cls = class_names[y_true[i]]
            pred_cls = class_names[y_pred[i]]
            error_subdir = errors_dir / f"{true_cls}_as_{pred_cls}"
            error_subdir.mkdir(parents=True, exist_ok=True)

            # ImageFolder.samples 是 (path, class_idx) 的列表
            src_path = Path(test_dataset.samples[i][0])
            dst_path = error_subdir / src_path.name
            shutil.copy2(str(src_path), str(dst_path))
            saved_count += 1

    if saved_count > 0:
        logger.info(f"已保存 {saved_count} 张错误分类样本到: {errors_dir}")
    else:
        logger.info("无错误分类样本，errors 目录未创建。")

    return saved_count


def evaluate(
    model_path: str,
    test_dir: str,
    output_dir: str = None,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """主评估流程

    Args:
        model_path: 模型文件路径
        test_dir: 测试集目录
        output_dir: 输出目录（默认与模型同目录）
        batch_size: 批大小（默认 32）
        num_workers: 数据加载线程数（默认 2）
    """
    import time as _time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 确定输出目录
    model_path = Path(model_path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    logger.info(f"加载模型: {model_path}")
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"模型文件大小: {model_size_mb:.1f} MB")

    try:
        model, class_to_idx, config = load_model(str(model_path), device)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

    # 构建类别名称列表（按索引排序）
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    logger.info(f"类别: {class_names}")

    # 加载测试集
    logger.info(f"测试集目录: {test_dir}")
    if not Path(test_dir).exists():
        logger.error(f"测试集目录不存在: {test_dir}")
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")

    try:
        test_dataset = datasets.ImageFolder(
            root=test_dir,
            transform=get_test_transform(),
        )
    except Exception as e:
        logger.error(f"测试集加载失败: {e}")
        raise

    if len(test_dataset) == 0:
        logger.error("测试集为空，无法评估。")
        raise ValueError("测试集为空")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    logger.info(f"测试集样本数: {len(test_dataset)}")
    logger.info(f"测试集类别映射: {test_dataset.class_to_idx}")
    logger.info(f"评估参数: batch_size={batch_size}, num_workers={num_workers}")

    # 预测
    logger.info("开始预测...")
    predict_start = _time.time()
    y_pred, y_true, y_probs = predict_all(model, test_loader, device)
    predict_time = _time.time() - predict_start
    logger.info(f"预测完成，耗时: {predict_time:.1f}s ({len(y_true)} 样本)")

    # 计算指标
    metrics = compute_metrics(y_true, y_pred, class_names)

    # 打印结果
    logger.info("=" * 60)
    logger.info("测试集评估结果")
    logger.info("=" * 60)
    logger.info(f"总体准确率 (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"宏平均精确率 (Precision): {metrics['precision_macro']:.4f}")
    logger.info(f"宏平均召回率 (Recall):    {metrics['recall_macro']:.4f}")
    logger.info(f"宏平均 F1 分数 (F1):      {metrics['f1_macro']:.4f}")
    logger.info("")
    logger.info("详细分类报告:")
    logger.info(f"\n{metrics['classification_report']}")

    # 混淆矩阵
    cm = np.array(metrics["confusion_matrix"])

    # ── 生成混淆矩阵并分析原因（Prompt 核心需求） ──
    analysis = generate_confusion_matrix_and_analyze(
        cm=cm,
        class_names=class_names,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        output_dir=output_dir,
    )

    # 绘制每类指标
    plot_per_class_metrics(metrics, class_names, output_dir / "per_class_metrics.png")

    # 保存错误分类样本（辅助人工分析误判原因）
    error_count = save_error_samples(test_dataset, y_true, y_pred, class_names, output_dir)

    # 生成 Grad-CAM 热力图（对错误样本进行可视化原因分析）
    try:
        gradcam_count = generate_gradcam(
            model, test_dataset, y_true, y_pred, class_names, output_dir, device
        )
    except Exception as e:
        logger.warning(f"Grad-CAM 生成失败（不影响主流程）: {e}")
        gradcam_count = 0

    # 保存指标（含混淆分析作为正式交付物）
    metrics_to_save = {k: v for k, v in metrics.items() if k != "classification_report"}
    metrics_to_save["classification_report_text"] = metrics["classification_report"]
    metrics_to_save["confusion_analysis"] = analysis
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

    logger.info(f"\n评估结果已保存到: {output_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="VGG16 动物图像分类 - 评估")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径 (.pth)")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="测试集目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小 (默认 32)")
    parser.add_argument("--num-workers", type=int, default=0 if os.environ.get("IN_DOCKER") == "1" else 2, help="数据加载线程数 (默认 2)")
    args = parser.parse_args()

    metrics = evaluate(
        args.model,
        args.test_dir,
        args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    accuracy = metrics["accuracy"] * 100
    logger.info(f"\n最终测试准确率: {accuracy:.2f}%")

    if accuracy >= 85:
        logger.info("✅ 准确率 >= 85%，达标！")
    else:
        logger.info(f"⚠️ 准确率 < 85%，需要调优。当前: {accuracy:.2f}%")
        logger.info("请运行 retrain.py 进行参数调优。")


if __name__ == "__main__":
    main()
