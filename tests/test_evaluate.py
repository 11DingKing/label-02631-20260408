"""
evaluate.py 测试用例
覆盖：
- load_model: 模型加载
- get_test_transform: 测试变换
- predict_all: 批量预测
- compute_metrics: 指标计算
- plot_confusion_matrix: 混淆矩阵绘制
- plot_per_class_metrics: 每类指标绘制
- analyze_confusion_matrix: 混淆矩阵分析
- evaluate (集成): 完整评估流程
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from evaluate import (
    load_model,
    get_test_transform,
    predict_all,
    compute_metrics,
    plot_confusion_matrix,
    plot_per_class_metrics,
    analyze_confusion_matrix,
    _get_biological_context,
    save_error_samples,
    generate_gradcam,
    evaluate,
    IMG_SIZE,
)

# 默认 4 类（从数据集动态确定，测试中使用固定值）
NUM_CLASSES = 4


# ── get_test_transform ─────────────────────────────────

class TestGetTestTransform:
    def test_returns_compose(self):
        t = get_test_transform()
        from torchvision import transforms
        assert isinstance(t, transforms.Compose)

    def test_output_shape(self):
        from PIL import Image
        t = get_test_transform()
        img = Image.new("RGB", (500, 300), "blue")
        tensor = t(img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_no_random_augmentation(self):
        """测试变换不应有随机增强"""
        t = get_test_transform()
        types = [type(tr).__name__ for tr in t.transforms]
        assert "RandomHorizontalFlip" not in types
        assert "RandomRotation" not in types
        assert "RandomCrop" not in types


# ── load_model ─────────────────────────────────────────

class TestLoadModel:
    def test_loads_successfully(self, tiny_model_checkpoint):
        ckpt_path, _, expected_class_to_idx = tiny_model_checkpoint
        device = torch.device("cpu")
        model, class_to_idx, config = load_model(str(ckpt_path), device)

        assert model is not None
        assert class_to_idx == expected_class_to_idx
        assert isinstance(config, dict)

    def test_model_in_eval_mode(self, tiny_model_checkpoint):
        ckpt_path, _, _ = tiny_model_checkpoint
        device = torch.device("cpu")
        model, _, _ = load_model(str(ckpt_path), device)
        assert not model.training

    def test_model_output_shape(self, tiny_model_checkpoint):
        ckpt_path, _, _ = tiny_model_checkpoint
        device = torch.device("cpu")
        model, _, _ = load_model(str(ckpt_path), device)

        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_invalid_path_raises(self):
        device = torch.device("cpu")
        with pytest.raises(Exception):
            load_model("/nonexistent/model.pth", device)


# ── predict_all ────────────────────────────────────────

class TestPredictAll:
    @pytest.fixture
    def simple_model_and_loader(self, tiny_model_checkpoint):
        ckpt_path, _, _ = tiny_model_checkpoint
        device = torch.device("cpu")
        model, _, _ = load_model(str(ckpt_path), device)

        x = torch.randn(8, 3, IMG_SIZE, IMG_SIZE)
        y = torch.randint(0, NUM_CLASSES, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        return model, loader, device

    def test_returns_arrays(self, simple_model_and_loader):
        model, loader, device = simple_model_and_loader
        preds, labels, probs = predict_all(model, loader, device)
        assert isinstance(preds, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(probs, np.ndarray)

    def test_correct_lengths(self, simple_model_and_loader):
        model, loader, device = simple_model_and_loader
        preds, labels, probs = predict_all(model, loader, device)
        assert len(preds) == 8
        assert len(labels) == 8
        assert len(probs) == 8

    def test_preds_in_valid_range(self, simple_model_and_loader):
        model, loader, device = simple_model_and_loader
        preds, _, _ = predict_all(model, loader, device)
        assert all(0 <= p < NUM_CLASSES for p in preds)

    def test_probs_sum_to_one(self, simple_model_and_loader):
        model, loader, device = simple_model_and_loader
        _, _, probs = predict_all(model, loader, device)
        for row in probs:
            assert abs(row.sum() - 1.0) < 1e-5


# ── compute_metrics ────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        class_names = ["cat", "dog", "lion", "tiger"]
        metrics = compute_metrics(y_true, y_pred, class_names)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        y_pred = np.array([1, 1, 0, 0, 3, 3, 2, 2])
        class_names = ["cat", "dog", "lion", "tiger"]
        metrics = compute_metrics(y_true, y_pred, class_names)

        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 1, 0])
        class_names = ["cat", "dog", "lion", "tiger"]
        metrics = compute_metrics(y_true, y_pred, class_names)

        cm = metrics["confusion_matrix"]
        assert len(cm) == 4
        assert len(cm[0]) == 4

    def test_per_class_metrics_present(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        class_names = ["cat", "dog", "lion", "tiger"]
        metrics = compute_metrics(y_true, y_pred, class_names)

        for cls in class_names:
            assert cls in metrics["per_class"]
            assert "precision" in metrics["per_class"][cls]
            assert "recall" in metrics["per_class"][cls]
            assert "f1" in metrics["per_class"][cls]
            assert "support" in metrics["per_class"][cls]

    def test_classification_report_present(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        class_names = ["cat", "dog", "lion", "tiger"]
        metrics = compute_metrics(y_true, y_pred, class_names)

        assert "classification_report" in metrics
        assert isinstance(metrics["classification_report"], str)
        assert len(metrics["classification_report"]) > 0

    def test_metrics_values_in_range(self):
        """所有指标应在 [0, 1] 范围内"""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 3, 1, 1, 0, 0])
        class_names = ["cat", "dog", "lion", "tiger"]
        metrics = compute_metrics(y_true, y_pred, class_names)

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision_macro"] <= 1
        assert 0 <= metrics["recall_macro"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1


# ── plot_confusion_matrix ──────────────────────────────

class TestPlotConfusionMatrix:
    def test_saves_file(self, tmp_path):
        cm = np.array([[10, 2, 0, 1], [1, 8, 1, 0], [0, 1, 9, 2], [1, 0, 1, 7]])
        class_names = ["cat", "dog", "lion", "tiger"]
        output_path = tmp_path / "cm.png"
        plot_confusion_matrix(cm, class_names, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_handles_zero_row(self, tmp_path):
        """某类无样本时不应崩溃"""
        cm = np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 5]])
        class_names = ["cat", "dog", "lion", "tiger"]
        output_path = tmp_path / "cm_zero.png"
        plot_confusion_matrix(cm, class_names, output_path)
        assert output_path.exists()


# ── plot_per_class_metrics ─────────────────────────────

class TestPlotPerClassMetrics:
    def test_saves_file(self, tmp_path):
        metrics = {
            "per_class": {
                "cat": {"precision": 0.9, "recall": 0.85, "f1": 0.87},
                "dog": {"precision": 0.8, "recall": 0.9, "f1": 0.85},
                "lion": {"precision": 0.75, "recall": 0.7, "f1": 0.72},
                "tiger": {"precision": 0.95, "recall": 0.92, "f1": 0.93},
            }
        }
        class_names = ["cat", "dog", "lion", "tiger"]
        output_path = tmp_path / "metrics.png"
        plot_per_class_metrics(metrics, class_names, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


# ── analyze_confusion_matrix ──────────────────────────

class TestAnalyzeConfusionMatrix:
    def test_returns_string(self):
        cm = np.array([[10, 2, 0, 1], [1, 8, 1, 0], [0, 1, 9, 2], [1, 0, 1, 7]])
        class_names = ["cat", "dog", "lion", "tiger"]
        analysis = analyze_confusion_matrix(cm, class_names)
        assert isinstance(analysis, str)
        assert len(analysis) > 0

    def test_contains_sections(self):
        cm = np.array([[10, 2, 0, 1], [1, 8, 1, 0], [0, 1, 9, 2], [1, 0, 1, 7]])
        class_names = ["cat", "dog", "lion", "tiger"]
        analysis = analyze_confusion_matrix(cm, class_names)
        assert "每类分类准确率" in analysis
        assert "混淆原因分析" in analysis
        assert "整体诊断与改进建议" in analysis

    def test_perfect_matrix(self):
        """完美分类时应报告无混淆"""
        cm = np.diag([10, 10, 10, 10])
        class_names = ["cat", "dog", "lion", "tiger"]
        analysis = analyze_confusion_matrix(cm, class_names)
        assert "无显著混淆对" in analysis

    def test_identifies_confusion_pairs(self):
        """应识别出主要混淆对"""
        cm = np.array([[5, 5, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
        class_names = ["cat", "dog", "lion", "tiger"]
        analysis = analyze_confusion_matrix(cm, class_names)
        assert "cat" in analysis
        assert "dog" in analysis

    def test_handles_zero_row(self):
        """某类无样本时不应崩溃"""
        cm = np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 5]])
        class_names = ["cat", "dog", "lion", "tiger"]
        analysis = analyze_confusion_matrix(cm, class_names)
        assert isinstance(analysis, str)

    def test_dynamic_analysis_with_probs(self):
        """传入概率数据时应输出置信度分析和逐样本明细"""
        cm = np.array([[5, 5, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
        class_names = ["cat", "dog", "lion", "tiger"]

        # 模拟: 5 个 cat 被正确分类, 5 个 cat 被误判为 dog
        y_true = np.array([0]*10 + [1]*10 + [2]*10 + [3]*10)
        y_pred = np.array([0]*5 + [1]*5 + [1]*10 + [2]*10 + [3]*10)
        # 构造概率: 误判样本对 dog 有较高置信度
        y_probs = np.zeros((40, 4))
        # 正确的 cat
        for i in range(5):
            y_probs[i] = [0.8, 0.1, 0.05, 0.05]
        # 误判的 cat -> dog
        for i in range(5, 10):
            y_probs[i] = [0.2, 0.6, 0.1, 0.1]
        # 正确的 dog
        for i in range(10, 20):
            y_probs[i] = [0.1, 0.8, 0.05, 0.05]
        # 正确的 lion
        for i in range(20, 30):
            y_probs[i] = [0.05, 0.05, 0.8, 0.1]
        # 正确的 tiger
        for i in range(30, 40):
            y_probs[i] = [0.05, 0.05, 0.1, 0.8]

        analysis = analyze_confusion_matrix(cm, class_names, y_true, y_pred, y_probs)

        # 应包含置信度分析
        assert "置信度" in analysis
        assert "诊断" in analysis
        assert "建议" in analysis
        # 应包含逐样本错误明细
        assert "错误样本明细" in analysis
        assert "样本#" in analysis
        assert "Top2" in analysis

    def test_dynamic_analysis_high_confidence_error(self):
        """高置信度误判应给出'特征空间重叠'诊断和逐样本观察"""
        cm = np.array([[5, 5, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
        class_names = ["cat", "dog", "lion", "tiger"]

        y_true = np.array([0]*10 + [1]*10 + [2]*10 + [3]*10)
        y_pred = np.array([0]*5 + [1]*5 + [1]*10 + [2]*10 + [3]*10)
        y_probs = np.zeros((40, 4))
        for i in range(5):
            y_probs[i] = [0.9, 0.05, 0.03, 0.02]
        # 高置信度误判
        for i in range(5, 10):
            y_probs[i] = [0.1, 0.8, 0.05, 0.05]
        for i in range(10, 20):
            y_probs[i] = [0.05, 0.85, 0.05, 0.05]
        for i in range(20, 30):
            y_probs[i] = [0.03, 0.02, 0.9, 0.05]
        for i in range(30, 40):
            y_probs[i] = [0.02, 0.03, 0.05, 0.9]

        analysis = analyze_confusion_matrix(cm, class_names, y_true, y_pred, y_probs)
        assert "高度自信" in analysis or "特征空间" in analysis
        # 逐样本观察应包含高确信误判描述
        assert "高确信误判" in analysis or "视觉特征" in analysis

    def test_dynamic_analysis_low_confidence_error(self):
        """低置信度误判应给出'边缘误判'诊断和人工复核建议"""
        cm = np.array([[8, 2, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
        class_names = ["cat", "dog", "lion", "tiger"]

        y_true = np.array([0]*10 + [1]*10 + [2]*10 + [3]*10)
        y_pred = np.array([0]*8 + [1]*2 + [1]*10 + [2]*10 + [3]*10)
        y_probs = np.zeros((40, 4))
        for i in range(8):
            y_probs[i] = [0.7, 0.15, 0.1, 0.05]
        # 低置信度误判
        for i in range(8, 10):
            y_probs[i] = [0.3, 0.35, 0.2, 0.15]
        for i in range(10, 20):
            y_probs[i] = [0.1, 0.7, 0.1, 0.1]
        for i in range(20, 30):
            y_probs[i] = [0.1, 0.1, 0.7, 0.1]
        for i in range(30, 40):
            y_probs[i] = [0.1, 0.1, 0.1, 0.7]

        analysis = analyze_confusion_matrix(cm, class_names, y_true, y_pred, y_probs)
        assert "边缘误判" in analysis or "置信度较低" in analysis
        # 逐样本观察应包含低确信描述
        assert "低确信误判" in analysis or "人工复核" in analysis

    def test_bidirectional_confusion_detected(self):
        """无概率数据时应检测双向混淆"""
        # cat->dog 和 dog->cat 都有显著混淆
        cm = np.array([[6, 4, 0, 0], [3, 7, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
        class_names = ["cat", "dog", "lion", "tiger"]
        analysis = analyze_confusion_matrix(cm, class_names)
        assert "双向混淆" in analysis

    def test_biological_context_included(self):
        """混淆分析应包含生物学背景"""
        cm = np.array([[5, 0, 0, 5], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
        class_names = ["cat", "dog", "lion", "tiger"]
        y_true = np.array([0]*10 + [1]*10 + [2]*10 + [3]*10)
        y_pred = np.array([0]*5 + [3]*5 + [1]*10 + [2]*10 + [3]*10)
        y_probs = np.zeros((40, 4))
        for i in range(5):
            y_probs[i] = [0.8, 0.05, 0.05, 0.1]
        for i in range(5, 10):
            y_probs[i] = [0.2, 0.05, 0.05, 0.7]
        for i in range(10, 40):
            y_probs[i, y_true[i]] = 0.8
            for j in range(4):
                if j != y_true[i]:
                    y_probs[i, j] = 0.2 / 3

        analysis = analyze_confusion_matrix(cm, class_names, y_true, y_pred, y_probs)
        assert "猫科" in analysis or "Felidae" in analysis

    def test_overall_rating(self):
        """应包含整体评级"""
        cm = np.diag([10, 10, 10, 10])
        class_names = ["cat", "dog", "lion", "tiger"]
        analysis = analyze_confusion_matrix(cm, class_names)
        assert "评级" in analysis

    def test_worst_class_identified(self):
        """应识别最弱类别"""
        cm = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [5, 2, 1, 2]])
        class_names = ["cat", "dog", "lion", "tiger"]
        analysis = analyze_confusion_matrix(cm, class_names)
        assert "最弱类别" in analysis
        assert "tiger" in analysis

    def test_sample_detail_truncation(self):
        """超过 10 个错误样本时应截断输出"""
        # 20 个 cat 中 15 个被误判为 dog
        cm = np.array([[5, 15, 0, 0], [0, 20, 0, 0], [0, 0, 20, 0], [0, 0, 0, 20]])
        class_names = ["cat", "dog", "lion", "tiger"]

        y_true = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20)
        y_pred = np.array([0]*5 + [1]*15 + [1]*20 + [2]*20 + [3]*20)
        y_probs = np.zeros((80, 4))
        for i in range(5):
            y_probs[i] = [0.8, 0.1, 0.05, 0.05]
        for i in range(5, 20):
            y_probs[i] = [0.2, 0.6, 0.1, 0.1]
        for i in range(20, 40):
            y_probs[i] = [0.1, 0.8, 0.05, 0.05]
        for i in range(40, 60):
            y_probs[i] = [0.05, 0.05, 0.8, 0.1]
        for i in range(60, 80):
            y_probs[i] = [0.05, 0.05, 0.1, 0.8]

        analysis = analyze_confusion_matrix(cm, class_names, y_true, y_pred, y_probs)
        assert "省略剩余" in analysis


# ── save_error_samples ─────────────────────────────────

class TestSaveErrorSamples:
    def test_saves_misclassified_images(self, tmp_path):
        """错误分类的图片应被保存到 errors/ 目录"""
        from PIL import Image
        from types import SimpleNamespace

        # 创建模拟测试集目录
        test_dir = tmp_path / "test_images"
        for cls in ["cat", "dog"]:
            d = test_dir / cls
            d.mkdir(parents=True)
            for i in range(3):
                Image.new("RGB", (10, 10)).save(d / f"{cls}_{i}.jpg")

        # 模拟 ImageFolder.samples: [(path, class_idx), ...]
        samples = []
        for i in range(3):
            samples.append((str(test_dir / "cat" / f"cat_{i}.jpg"), 0))
        for i in range(3):
            samples.append((str(test_dir / "dog" / f"dog_{i}.jpg"), 1))

        mock_dataset = SimpleNamespace(samples=samples)

        # 模拟: cat_0 被误判为 dog, 其余正确
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 0, 0, 1, 1, 1])
        class_names = ["cat", "dog"]

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = save_error_samples(mock_dataset, y_true, y_pred, class_names, output_dir)
        assert count == 1
        error_dir = output_dir / "errors" / "cat_as_dog"
        assert error_dir.exists()
        assert len(list(error_dir.glob("*.jpg"))) == 1

    def test_no_errors_no_directory(self, tmp_path):
        """无错误分类时不应创建 errors 目录"""
        from types import SimpleNamespace

        mock_dataset = SimpleNamespace(samples=[("/fake/a.jpg", 0), ("/fake/b.jpg", 1)])
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        class_names = ["cat", "dog"]

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = save_error_samples(mock_dataset, y_true, y_pred, class_names, output_dir)
        assert count == 0
        assert not (output_dir / "errors").exists()

    def test_multiple_confusion_pairs(self, tmp_path):
        """多个混淆对应分别保存到不同子目录"""
        from PIL import Image
        from types import SimpleNamespace

        test_dir = tmp_path / "test_images"
        for cls in ["cat", "dog", "lion"]:
            d = test_dir / cls
            d.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (10, 10)).save(d / f"{cls}_{i}.jpg")

        samples = []
        for cls in ["cat", "dog", "lion"]:
            for i in range(2):
                idx = {"cat": 0, "dog": 1, "lion": 2}[cls]
                samples.append((str(test_dir / cls / f"{cls}_{i}.jpg"), idx))

        mock_dataset = SimpleNamespace(samples=samples)

        # cat_0 -> dog, dog_0 -> lion
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 0, 2, 1, 2, 2])
        class_names = ["cat", "dog", "lion"]

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = save_error_samples(mock_dataset, y_true, y_pred, class_names, output_dir)
        assert count == 2
        assert (output_dir / "errors" / "cat_as_dog").exists()
        assert (output_dir / "errors" / "dog_as_lion").exists()


# ── evaluate (集成测试) ────────────────────────────────

class TestEvaluateIntegration:
    def test_full_evaluation_flow(self, tiny_model_checkpoint, tmp_path):
        ckpt_path, dataset_path, _ = tiny_model_checkpoint
        output_dir = tmp_path / "eval_output"

        metrics = evaluate(
            model_path=str(ckpt_path),
            test_dir=str(dataset_path / "test"),
            output_dir=str(output_dir),
        )

        # 验证返回的指标
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics

        # 验证输出文件
        assert (output_dir / "confusion_matrix.png").exists()
        assert (output_dir / "per_class_metrics.png").exists()
        assert (output_dir / "test_metrics.json").exists()
        assert (output_dir / "confusion_analysis.txt").exists()

        # 验证 confusion_analysis.txt 内容非空且包含关键章节
        analysis_content = (output_dir / "confusion_analysis.txt").read_text(encoding="utf-8")
        assert len(analysis_content) > 0
        assert "混淆矩阵分析报告" in analysis_content
        assert "每类分类准确率" in analysis_content
        assert "混淆原因分析" in analysis_content
        assert "整体诊断与改进建议" in analysis_content

        # 验证 errors 目录（未训练模型大概率有错误分类）
        # 不强制要求有错误，但如果有，目录结构应正确
        errors_dir = output_dir / "errors"
        if errors_dir.exists():
            for subdir in errors_dir.iterdir():
                assert "_as_" in subdir.name
                assert any(subdir.glob("*.jpg"))

        # 验证 Grad-CAM 目录（未训练模型大概率有错误分类）
        gradcam_dir = output_dir / "gradcam"
        if errors_dir.exists():
            assert gradcam_dir.exists()
            png_files = list(gradcam_dir.glob("*.png"))
            assert len(png_files) > 0
            for f in png_files:
                assert "gradcam_" in f.name

        # 验证 JSON 可解析
        with open(output_dir / "test_metrics.json") as f:
            saved = json.load(f)
        assert "accuracy" in saved
        assert "confusion_analysis" in saved
        assert len(saved["confusion_analysis"]) > 0

        # 验证 confusion_analysis.txt 与 test_metrics.json 中的内容一致
        assert saved["confusion_analysis"] == analysis_content

    def test_metrics_values_reasonable(self, tiny_model_checkpoint, tmp_path):
        """指标值应在合理范围内"""
        ckpt_path, dataset_path, _ = tiny_model_checkpoint
        output_dir = tmp_path / "eval_check"

        metrics = evaluate(
            model_path=str(ckpt_path),
            test_dir=str(dataset_path / "test"),
            output_dir=str(output_dir),
        )

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision_macro"] <= 1
        assert 0 <= metrics["recall_macro"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1


# ── generate_gradcam ───────────────────────────────────

class TestGenerateGradcam:
    def test_generates_heatmaps_for_errors(self, tiny_model_checkpoint, tmp_path):
        """有错误分类时应生成 Grad-CAM 热力图"""
        ckpt_path, dataset_path, _ = tiny_model_checkpoint
        device = torch.device("cpu")
        model, class_to_idx, _ = load_model(str(ckpt_path), device)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

        from torchvision import datasets
        test_dataset = datasets.ImageFolder(
            root=str(dataset_path / "test"),
            transform=get_test_transform(),
        )

        # 模拟预测结果：全部错误
        n = len(test_dataset)
        y_true = np.array([s[1] for s in test_dataset.samples])
        y_pred = np.array([(s[1] + 1) % len(class_names) for s in test_dataset.samples])

        output_dir = tmp_path / "gradcam_output"
        output_dir.mkdir()

        count = generate_gradcam(
            model, test_dataset, y_true, y_pred, class_names, output_dir, device
        )

        assert count > 0
        gradcam_dir = output_dir / "gradcam"
        assert gradcam_dir.exists()
        assert len(list(gradcam_dir.glob("*.png"))) == count

    def test_no_errors_no_gradcam(self, tiny_model_checkpoint, tmp_path):
        """无错误分类时不应生成热力图"""
        ckpt_path, dataset_path, _ = tiny_model_checkpoint
        device = torch.device("cpu")
        model, class_to_idx, _ = load_model(str(ckpt_path), device)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

        from torchvision import datasets
        test_dataset = datasets.ImageFolder(
            root=str(dataset_path / "test"),
            transform=get_test_transform(),
        )

        # 全部正确
        y_true = np.array([s[1] for s in test_dataset.samples])
        y_pred = y_true.copy()

        output_dir = tmp_path / "gradcam_output"
        output_dir.mkdir()

        count = generate_gradcam(
            model, test_dataset, y_true, y_pred, class_names, output_dir, device
        )
        assert count == 0

    def test_max_samples_limit(self, tiny_model_checkpoint, tmp_path):
        """应尊重 max_samples 限制"""
        ckpt_path, dataset_path, _ = tiny_model_checkpoint
        device = torch.device("cpu")
        model, class_to_idx, _ = load_model(str(ckpt_path), device)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

        from torchvision import datasets
        test_dataset = datasets.ImageFolder(
            root=str(dataset_path / "test"),
            transform=get_test_transform(),
        )

        # 全部错误
        y_true = np.array([s[1] for s in test_dataset.samples])
        y_pred = np.array([(s[1] + 1) % len(class_names) for s in test_dataset.samples])

        output_dir = tmp_path / "gradcam_output"
        output_dir.mkdir()

        count = generate_gradcam(
            model, test_dataset, y_true, y_pred, class_names, output_dir, device,
            max_samples=2,
        )
        assert count <= 2


# ── 错误处理与边界条件测试 ─────────────────────────────

class TestEvaluateErrorHandling:
    def test_nonexistent_model_file(self, tmp_path):
        """模型文件不存在时应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="模型文件不存在"):
            evaluate(
                model_path=str(tmp_path / "nonexistent.pth"),
                test_dir=str(tmp_path),
            )

    def test_nonexistent_test_dir(self, tiny_model_checkpoint, tmp_path):
        """测试集目录不存在时应抛出 FileNotFoundError"""
        ckpt_path, _, _ = tiny_model_checkpoint
        with pytest.raises(FileNotFoundError, match="测试集目录不存在"):
            evaluate(
                model_path=str(ckpt_path),
                test_dir=str(tmp_path / "nonexistent_test"),
            )

    def test_corrupted_model_file(self, tmp_path):
        """损坏的模型文件应抛出异常"""
        bad_model = tmp_path / "bad_model.pth"
        bad_model.write_text("this is not a valid model file")
        with pytest.raises(Exception):
            evaluate(
                model_path=str(bad_model),
                test_dir=str(tmp_path),
            )

    def test_empty_test_dataset(self, tiny_model_checkpoint, tmp_path):
        """空测试集应抛出 ValueError"""
        ckpt_path, _, _ = tiny_model_checkpoint
        empty_test = tmp_path / "empty_test"
        # ImageFolder 需要至少一个子目录
        (empty_test / "cat").mkdir(parents=True)

        with pytest.raises(Exception):
            evaluate(
                model_path=str(ckpt_path),
                test_dir=str(empty_test),
            )

    def test_custom_batch_size_and_workers(self, tiny_model_checkpoint, tmp_path):
        """自定义 batch_size 和 num_workers 应正常工作"""
        ckpt_path, dataset_path, _ = tiny_model_checkpoint
        output_dir = tmp_path / "eval_custom"

        metrics = evaluate(
            model_path=str(ckpt_path),
            test_dir=str(dataset_path / "test"),
            output_dir=str(output_dir),
            batch_size=2,
            num_workers=0,
        )
        assert "accuracy" in metrics


class TestComputeMetricsEdgeCases:
    def test_single_class(self):
        """只有一个类别时应正常计算"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        class_names = ["cat"]
        metrics = compute_metrics(y_true, y_pred, class_names)
        assert metrics["accuracy"] == 1.0

    def test_single_sample(self):
        """只有一个样本时应正常计算"""
        y_true = np.array([0])
        y_pred = np.array([0])
        class_names = ["cat"]
        metrics = compute_metrics(y_true, y_pred, class_names)
        assert metrics["accuracy"] == 1.0

    def test_all_same_prediction(self):
        """所有样本预测为同一类时应正常计算"""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 0, 0, 0])
        class_names = ["cat", "dog", "lion", "tiger"]
        metrics = compute_metrics(y_true, y_pred, class_names)
        assert metrics["accuracy"] == 0.25
