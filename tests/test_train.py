"""
train.py 测试用例
覆盖：
- get_default_config: 默认配置完整性
- get_data_transforms: 数据变换管道
- build_model: VGG16 模型构建与冻结策略
- build_optimizer: 优化器和调度器构建
- train_one_epoch: 单轮训练
- validate: 验证
- save_training_curves: 训练曲线保存
- train (集成): 完整训练流程
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from train import (
    get_default_config,
    get_data_transforms,
    build_model,
    build_optimizer,
    build_dataloaders,
    train_one_epoch,
    validate,
    save_training_curves,
    IMG_SIZE,
)

# 默认 4 类（从数据集动态确定，测试中使用固定值）
NUM_CLASSES = 4


# ── get_default_config ─────────────────────────────────

class TestGetDefaultConfig:
    def test_returns_dict(self):
        config = get_default_config()
        assert isinstance(config, dict)

    def test_required_keys_present(self):
        config = get_default_config()
        required = [
            "dataset_root", "output_dir", "batch_size", "num_epochs",
            "learning_rate", "weight_decay", "optimizer", "scheduler",
            "num_workers", "pretrained", "freeze_features", "dropout", "seed",
        ]
        for key in required:
            assert key in config, f"缺少必要配置项: {key}"

    def test_default_epochs_at_least_15(self):
        config = get_default_config()
        assert config["num_epochs"] >= 15

    def test_learning_rate_positive(self):
        config = get_default_config()
        assert config["learning_rate"] > 0

    def test_batch_size_positive(self):
        config = get_default_config()
        assert config["batch_size"] > 0


# ── get_data_transforms ───────────────────────────────

class TestGetDataTransforms:
    def test_train_transform_is_compose(self):
        t = get_data_transforms(is_training=True)
        assert isinstance(t, transforms.Compose)

    def test_val_transform_is_compose(self):
        t = get_data_transforms(is_training=False)
        assert isinstance(t, transforms.Compose)

    def test_train_has_augmentation(self):
        """训练变换应包含数据增强"""
        t = get_data_transforms(is_training=True)
        transform_types = [type(tr).__name__ for tr in t.transforms]
        assert "RandomHorizontalFlip" in transform_types
        assert "RandomRotation" in transform_types
        assert "ColorJitter" in transform_types

    def test_val_no_augmentation(self):
        """验证变换不应包含随机增强"""
        t = get_data_transforms(is_training=False)
        transform_types = [type(tr).__name__ for tr in t.transforms]
        assert "RandomHorizontalFlip" not in transform_types
        assert "RandomRotation" not in transform_types

    def test_both_have_normalize(self):
        """两者都应有归一化"""
        for is_train in [True, False]:
            t = get_data_transforms(is_training=is_train)
            transform_types = [type(tr).__name__ for tr in t.transforms]
            assert "Normalize" in transform_types

    def test_train_output_shape(self):
        """训练变换输出应为 (3, 224, 224)"""
        from PIL import Image
        t = get_data_transforms(is_training=True)
        img = Image.new("RGB", (300, 300), "red")
        tensor = t(img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_val_output_shape(self):
        """验证变换输出应为 (3, 224, 224)"""
        from PIL import Image
        t = get_data_transforms(is_training=False)
        img = Image.new("RGB", (300, 300), "red")
        tensor = t(img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)


# ── build_model ────────────────────────────────────────

class TestBuildModel:
    def test_output_classes(self):
        """模型输出应为 4 类"""
        config = get_default_config()
        config["pretrained"] = False  # 加速测试
        device = torch.device("cpu")
        model = build_model(config, device, NUM_CLASSES)
        # 检查最后一层输出维度
        last_layer = model.classifier[-1]
        assert last_layer.out_features == NUM_CLASSES

    def test_freeze_features(self):
        """冻结特征层时，大部分参数不可训练"""
        config = get_default_config()
        config["pretrained"] = False
        config["freeze_features"] = True
        config["unfreeze_last_n"] = 0
        device = torch.device("cpu")
        model = build_model(config, device, NUM_CLASSES)

        frozen = sum(1 for p in model.features.parameters() if not p.requires_grad)
        assert frozen > 0

    def test_unfreeze_last_n(self):
        """解冻最后 N 层时，这些层应可训练"""
        config = get_default_config()
        config["pretrained"] = False
        config["freeze_features"] = True
        config["unfreeze_last_n"] = 4
        device = torch.device("cpu")
        model = build_model(config, device, NUM_CLASSES)

        # 最后几层应有可训练参数
        features_list = list(model.features.children())
        for layer in features_list[-4:]:
            for p in layer.parameters():
                assert p.requires_grad is True

    def test_no_freeze(self):
        """不冻结时所有参数可训练"""
        config = get_default_config()
        config["pretrained"] = False
        config["freeze_features"] = False
        device = torch.device("cpu")
        model = build_model(config, device, NUM_CLASSES)

        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable

    def test_classifier_structure(self):
        """分类头结构: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear"""
        config = get_default_config()
        config["pretrained"] = False
        device = torch.device("cpu")
        model = build_model(config, device, NUM_CLASSES)

        classifier = model.classifier
        assert isinstance(classifier[0], nn.Linear)
        assert isinstance(classifier[1], nn.ReLU)
        assert isinstance(classifier[2], nn.Dropout)
        assert isinstance(classifier[3], nn.Linear)
        assert isinstance(classifier[4], nn.ReLU)
        assert isinstance(classifier[5], nn.Dropout)
        assert isinstance(classifier[6], nn.Linear)

    def test_forward_pass(self):
        """前向传播应正常工作"""
        config = get_default_config()
        config["pretrained"] = False
        device = torch.device("cpu")
        model = build_model(config, device, NUM_CLASSES)
        model.eval()

        x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, NUM_CLASSES)

    def test_dynamic_num_classes(self):
        """模型应支持不同类别数"""
        config = get_default_config()
        config["pretrained"] = False
        device = torch.device("cpu")
        model = build_model(config, device, num_classes=5)
        last_layer = model.classifier[-1]
        assert last_layer.out_features == 5


# ── build_optimizer ────────────────────────────────────

class TestBuildOptimizer:
    def _get_model(self):
        config = get_default_config()
        config["pretrained"] = False
        return build_model(config, torch.device("cpu"), NUM_CLASSES)

    def test_adam_optimizer(self):
        model = self._get_model()
        config = get_default_config()
        config["optimizer"] = "adam"
        optimizer, scheduler = build_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_sgd_optimizer(self):
        model = self._get_model()
        config = get_default_config()
        config["optimizer"] = "sgd"
        optimizer, scheduler = build_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_invalid_optimizer_raises(self):
        model = self._get_model()
        config = get_default_config()
        config["optimizer"] = "invalid"
        with pytest.raises(ValueError, match="不支持的优化器"):
            build_optimizer(model, config)

    def test_step_scheduler(self):
        model = self._get_model()
        config = get_default_config()
        config["scheduler"] = "step"
        _, scheduler = build_optimizer(model, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_cosine_scheduler(self):
        model = self._get_model()
        config = get_default_config()
        config["scheduler"] = "cosine"
        _, scheduler = build_optimizer(model, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_no_scheduler(self):
        model = self._get_model()
        config = get_default_config()
        config["scheduler"] = "none"
        _, scheduler = build_optimizer(model, config)
        assert scheduler is None

    def test_learning_rate_applied(self):
        model = self._get_model()
        config = get_default_config()
        config["learning_rate"] = 0.0042
        optimizer, _ = build_optimizer(model, config)
        assert optimizer.param_groups[0]["lr"] == 0.0042


# ── train_one_epoch & validate ─────────────────────────

class TestTrainAndValidate:
    @pytest.fixture
    def dummy_loader(self):
        """创建一个假的 DataLoader"""
        x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)
        y = torch.randint(0, NUM_CLASSES, (16,))
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=4)

    @pytest.fixture
    def model_and_optimizer(self):
        config = get_default_config()
        config["pretrained"] = False
        device = torch.device("cpu")
        model = build_model(config, device, NUM_CLASSES)
        optimizer, _ = build_optimizer(model, config)
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, criterion, device

    def test_train_one_epoch_returns_metrics(self, dummy_loader, model_and_optimizer):
        model, optimizer, criterion, device = model_and_optimizer
        metrics = train_one_epoch(model, dummy_loader, criterion, optimizer, device, 1)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], float)
        assert 0 <= metrics["accuracy"] <= 100

    def test_train_one_epoch_loss_is_finite(self, dummy_loader, model_and_optimizer):
        model, optimizer, criterion, device = model_and_optimizer
        metrics = train_one_epoch(model, dummy_loader, criterion, optimizer, device, 1)
        assert metrics["loss"] > 0
        assert not float("inf") == metrics["loss"]

    def test_validate_returns_metrics(self, dummy_loader, model_and_optimizer):
        model, _, criterion, device = model_and_optimizer
        metrics = validate(model, dummy_loader, criterion, device)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], float)
        assert 0 <= metrics["accuracy"] <= 100

    def test_validate_no_grad(self, dummy_loader, model_and_optimizer):
        """验证时不应计算梯度"""
        model, _, criterion, device = model_and_optimizer
        model.eval()
        validate(model, dummy_loader, criterion, device)
        # 如果有梯度泄漏，参数的 grad 会被设置
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_training_improves_loss(self, dummy_loader, model_and_optimizer):
        """多轮训练后 loss 应该有变化（不一定下降，但不应完全不变）"""
        model, optimizer, criterion, device = model_and_optimizer
        losses = []
        for epoch in range(3):
            metrics = train_one_epoch(model, dummy_loader, criterion, optimizer, device, epoch)
            losses.append(metrics["loss"])
        # 至少有一些变化
        assert not all(l == losses[0] for l in losses)


# ── save_training_curves ───────────────────────────────

class TestSaveTrainingCurves:
    def test_saves_png(self, tmp_path):
        history = {
            "train_loss": [1.5, 1.2, 0.9],
            "val_loss": [1.6, 1.3, 1.0],
            "train_acc": [30, 50, 70],
            "val_acc": [28, 45, 65],
        }
        save_training_curves(history, tmp_path)
        assert (tmp_path / "training_curves.png").exists()
        assert (tmp_path / "training_curves.png").stat().st_size > 0

    def test_single_epoch(self, tmp_path):
        """单轮数据也应能正常绘图"""
        history = {
            "train_loss": [1.5],
            "val_loss": [1.6],
            "train_acc": [30],
            "val_acc": [28],
        }
        save_training_curves(history, tmp_path)
        assert (tmp_path / "training_curves.png").exists()


# ── train (集成测试) ───────────────────────────────────

class TestTrainIntegration:
    def test_full_training_flow(self, tmp_dataset):
        """完整训练流程：2 轮快速训练"""
        from train import train

        config = get_default_config()
        config["dataset_root"] = str(tmp_dataset)
        config["output_dir"] = str(tmp_dataset / "outputs")
        config["num_epochs"] = 2
        config["batch_size"] = 4
        config["num_workers"] = 0
        config["pretrained"] = False
        config["tag"] = "test"
        config["seed"] = 42

        output_dir, best_acc, test_metrics = train(config)

        # 验证输出文件
        assert output_dir.exists()
        assert (output_dir / "best_model.pth").exists()
        assert (output_dir / "final_model.pth").exists()
        assert (output_dir / "history.json").exists()
        assert (output_dir / "config.json").exists()
        assert (output_dir / "class_mapping.json").exists()
        assert (output_dir / "training_curves.png").exists()

        # 验证 history 内容
        with open(output_dir / "history.json") as f:
            history = json.load(f)
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert len(history["train_acc"]) == 2
        assert len(history["val_acc"]) == 2

        # 验证 config 保存
        with open(output_dir / "config.json") as f:
            saved_config = json.load(f)
        assert saved_config["num_epochs"] == 2

        # 验证 class_mapping
        with open(output_dir / "class_mapping.json") as f:
            mapping = json.load(f)
        assert "class_to_idx" in mapping
        assert len(mapping["class_to_idx"]) == NUM_CLASSES

        # 验证 best_acc 是合理的百分比
        assert 0 <= best_acc <= 100

    def test_model_checkpoint_loadable(self, tmp_dataset):
        """保存的模型应能正常加载"""
        from train import train

        config = get_default_config()
        config["dataset_root"] = str(tmp_dataset)
        config["output_dir"] = str(tmp_dataset / "outputs")
        config["num_epochs"] = 1
        config["batch_size"] = 4
        config["num_workers"] = 0
        config["pretrained"] = False
        config["tag"] = "loadtest"

        output_dir, _, _ = train(config)

        checkpoint = torch.load(
            output_dir / "best_model.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "val_accuracy" in checkpoint
        assert "config" in checkpoint
        assert "class_to_idx" in checkpoint


# ── build_dataloaders 类别白名单过滤 ───────────────────

class TestBuildDataloadersClassFilter:
    def test_filters_extra_classes(self, tmp_dataset):
        """数据目录中存在白名单外的类别时应被过滤"""
        from PIL import Image
        # 添加一个不在白名单中的类别
        elephant_dir = tmp_dataset / "train" / "elephant"
        elephant_dir.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (64, 64), "gray").save(elephant_dir / f"elephant_{i:04d}.jpg")
        elephant_val = tmp_dataset / "val" / "elephant"
        elephant_val.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (64, 64), "gray").save(elephant_val / f"elephant_v_{i:04d}.jpg")

        # 写入 classes.json 只包含 4 类
        from prepare_data import save_classes_json
        save_classes_json(tmp_dataset, ["cat", "dog", "tiger", "lion"])

        config = get_default_config()
        config["dataset_root"] = str(tmp_dataset)
        config["batch_size"] = 4
        config["num_workers"] = 0

        train_loader, val_loader, test_loader, class_to_idx, num_classes = build_dataloaders(config)

        # elephant 不应出现在类别映射中
        assert "elephant" not in class_to_idx
        assert num_classes == 4

    def test_loads_all_whitelisted_classes(self, tmp_dataset):
        """白名单中的所有类别都应被加载"""
        from prepare_data import save_classes_json
        save_classes_json(tmp_dataset, ["cat", "dog", "tiger", "lion"])

        config = get_default_config()
        config["dataset_root"] = str(tmp_dataset)
        config["batch_size"] = 4
        config["num_workers"] = 0

        train_loader, val_loader, test_loader, class_to_idx, num_classes = build_dataloaders(config)

        assert set(class_to_idx.keys()) == {"cat", "dog", "tiger", "lion"}
        assert num_classes == 4


# ── 错误处理与边界条件测试 ─────────────────────────────

class TestTrainErrorHandling:
    def test_train_with_nonexistent_dataset(self, tmp_path):
        """数据集目录不存在时应抛出 FileNotFoundError"""
        from train import train

        config = get_default_config()
        config["dataset_root"] = str(tmp_path / "nonexistent")
        config["output_dir"] = str(tmp_path / "outputs")
        config["num_epochs"] = 1
        config["batch_size"] = 4
        config["num_workers"] = 0
        config["pretrained"] = False
        config["tag"] = "error_test"

        with pytest.raises(FileNotFoundError):
            train(config)

    def test_train_with_empty_dataset(self, tmp_path):
        """数据集目录存在但为空时应抛出异常"""
        from train import train

        # 创建空目录结构
        (tmp_path / "train" / "cat").mkdir(parents=True)
        (tmp_path / "val" / "cat").mkdir(parents=True)

        config = get_default_config()
        config["dataset_root"] = str(tmp_path)
        config["output_dir"] = str(tmp_path / "outputs")
        config["num_epochs"] = 1
        config["batch_size"] = 4
        config["num_workers"] = 0
        config["pretrained"] = False
        config["tag"] = "empty_test"

        with pytest.raises(Exception):
            train(config)

    def test_invalid_optimizer_in_config(self, tmp_dataset):
        """无效优化器配置应抛出 ValueError"""
        from train import train

        config = get_default_config()
        config["dataset_root"] = str(tmp_dataset)
        config["output_dir"] = str(tmp_dataset / "outputs")
        config["num_epochs"] = 1
        config["batch_size"] = 4
        config["num_workers"] = 0
        config["pretrained"] = False
        config["optimizer"] = "invalid_optimizer"
        config["tag"] = "invalid_opt"

        with pytest.raises(ValueError, match="不支持的优化器"):
            train(config)

    def test_zero_epochs(self, tmp_dataset):
        """0 轮训练应正常完成（不进入循环）"""
        from train import train

        config = get_default_config()
        config["dataset_root"] = str(tmp_dataset)
        config["output_dir"] = str(tmp_dataset / "outputs")
        config["num_epochs"] = 0
        config["batch_size"] = 4
        config["num_workers"] = 0
        config["pretrained"] = False
        config["tag"] = "zero_epochs"

        output_dir, best_acc, _ = train(config)
        assert output_dir.exists()
        assert best_acc == 0.0  # 没有训练，最佳准确率为 0
