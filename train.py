"""
VGG16 动物图像分类 - 训练脚本
功能：
1. 加载预训练 VGG16 模型，修改分类头适配 4 类动物
2. 训练至少 15 轮
3. 记录训练/验证 loss 和 accuracy
4. 保存最佳模型和训练曲线
"""

import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from log_config import setup_logger
import font_config  # noqa: F401 — 确保中文字体配置生效

logger = setup_logger(__name__, "training.log")

# ── 配置 ──────────────────────────────────────────────
IMG_SIZE = 224


def get_default_config():
    """获取默认训练配置"""
    return {
        "dataset_root": "dataset",
        "output_dir": "outputs",
        "batch_size": 32,
        "num_epochs": 15,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "optimizer": "adam",        # adam / sgd
        "scheduler": "step",        # step / cosine / none
        "step_size": 5,
        "gamma": 0.5,
        "num_workers": 0 if os.environ.get("IN_DOCKER") == "1" else 2,
        "pretrained": True,
        "freeze_features": True,    # 冻结 VGG 特征提取层
        "unfreeze_last_n": 4,       # 解冻最后 N 层
        "dropout": 0.5,
        "seed": 42,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="VGG16 动物图像分类训练")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "sgd"])
    parser.add_argument("--scheduler", type=str, default=None, choices=["step", "cosine", "none"])
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="baseline", help="实验标签")
    return parser.parse_args()


def build_config(args):
    """合并默认配置和命令行参数"""
    config = get_default_config()

    # 从 JSON 文件加载
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            file_config = json.load(f)
        config.update(file_config)
        logger.info(f"从 {args.config} 加载配置")

    # 命令行参数覆盖
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.optimizer is not None:
        config["optimizer"] = args.optimizer
    if args.scheduler is not None:
        config["scheduler"] = args.scheduler
    if args.no_pretrained:
        config["pretrained"] = False
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    config["tag"] = args.tag
    return config


def get_data_transforms(is_training: bool):
    """获取数据变换"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])





def build_dataloaders(config: dict):
    """构建数据加载器，仅加载白名单中的类别
    
    加载 train/val/test 三个数据集：
    - train: 使用训练增强（随机裁剪、翻转、旋转、颜色抖动）
    - val/test: 使用验证/测试变换（仅调整大小和归一化）
    """
    dataset_root = Path(config["dataset_root"])

    # ── 类别白名单过滤 ──
    # 从 classes.json 读取允许的类别，防止意外加载多余类别
    allowed_classes = None
    classes_file = dataset_root / "classes.json"
    if classes_file.exists():
        with open(classes_file, encoding="utf-8") as f:
            classes_data = json.load(f)
        allowed_classes = set(classes_data.get("classes", []))
        logger.info(f"类别白名单: {sorted(allowed_classes)}")

    train_root = str(dataset_root / "train")
    val_root = str(dataset_root / "val")
    test_root = str(dataset_root / "test")

    def _build_filtered_dataset(root: str, transform, allowed: set = None):
        """构建仅包含白名单类别的 ImageFolder 数据集"""
        if allowed is None:
            return datasets.ImageFolder(root=root, transform=transform)

        # 自定义 find_classes：仅返回白名单中的类别
        class FilteredImageFolder(datasets.ImageFolder):
            def find_classes(self, directory):
                classes_list, class_to_idx = super().find_classes(directory)
                filtered = [c for c in classes_list if c in allowed]
                new_idx = {c: i for i, c in enumerate(filtered)}
                return filtered, new_idx

        return FilteredImageFolder(root=root, transform=transform)

    train_dataset = _build_filtered_dataset(
        train_root, get_data_transforms(is_training=True), allowed_classes,
    )
    val_dataset = _build_filtered_dataset(
        val_root, get_data_transforms(is_training=False), allowed_classes,
    )
    
    # 加载测试集（如果存在）
    test_dataset = None
    test_loader = None
    if Path(test_root).exists():
        try:
            test_dataset = _build_filtered_dataset(
                test_root, get_data_transforms(is_training=False), allowed_classes,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
                pin_memory=True,
            )
            logger.info(f"测试集样本数: {len(test_dataset)}")
        except Exception as e:
            logger.warning(f"测试集加载失败（将跳过测试集评估）: {e}")
    else:
        logger.warning(f"测试集目录不存在: {test_root}，将跳过测试集评估")

    # 日志：检查是否有白名单外的目录存在
    if allowed_classes is not None:
        all_dirs = {
            d.name for d in Path(train_root).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        }
        unexpected = all_dirs - allowed_classes
        if unexpected:
            logger.warning(
                f"数据集中发现白名单外的类别目录: {unexpected}，已排除。"
            )

    logger.info(f"训练集类别映射: {train_dataset.class_to_idx}")
    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"验证集样本数: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx, len(train_dataset.classes)



def build_model(config: dict, device: torch.device, num_classes: int = 4) -> nn.Module:
    """构建 VGG16 模型"""
    logger.info(f"构建 VGG16 模型 (pretrained={config['pretrained']}, num_classes={num_classes})")

    if config["pretrained"]:
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            model = models.vgg16(weights=weights)
        except Exception as e:
            logger.warning(f"预训练权重下载失败: {e}")
            logger.warning("回退为随机初始化权重，训练仍可继续但效果可能下降。")
            model = models.vgg16(weights=None)
    else:
        model = models.vgg16(weights=None)

    # 冻结特征提取层
    if config["freeze_features"]:
        for param in model.features.parameters():
            param.requires_grad = False

        # 解冻最后 N 层
        if config["unfreeze_last_n"] > 0:
            features_list = list(model.features.children())
            for layer in features_list[-config["unfreeze_last_n"]:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(f"解冻特征提取层最后 {config['unfreeze_last_n']} 层")

    # 替换分类头
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=config["dropout"]),
        nn.Linear(4096, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=config["dropout"]),
        nn.Linear(256, num_classes),
    )

    model = model.to(device)

    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    return model


def build_optimizer(model: nn.Module, config: dict):
    """构建优化器和学习率调度器"""
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(
            trainable_params,
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"不支持的优化器: {config['optimizer']}")

    scheduler = None
    if config["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["step_size"],
            gamma=config["gamma"],
        )
    elif config["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"],
            eta_min=1e-6,
        )

    logger.info(f"优化器: {config['optimizer']}, 学习率: {config['learning_rate']}")
    if scheduler:
        logger.info(f"学习率调度器: {config['scheduler']}")

    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"  Epoch {epoch} Batch {batch_idx+1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return {"loss": epoch_loss, "accuracy": epoch_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return {"loss": epoch_loss, "accuracy": epoch_acc}


def save_training_curves(history: dict, output_dir: Path):
    """保存训练曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss 曲线
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"训练曲线已保存: {output_dir / 'training_curves.png'}")


def train(config: dict):
    """主训练流程"""
    # 设置随机种子
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output_dir"]) / f"{config['tag']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 数据加载
    logger.info("加载数据集...")
    data_load_start = time.time()
    try:
        train_loader, val_loader, test_loader, class_to_idx, num_classes = build_dataloaders(config)
    except FileNotFoundError as e:
        logger.error(f"数据集目录不存在: {e}")
        logger.error("请先运行 prepare_data.py 准备数据集。")
        raise
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise
    data_load_time = time.time() - data_load_start
    logger.info(f"数据加载完成，耗时: {data_load_time:.1f}s")

    # 保存类别映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

    # 模型构建
    try:
        model = build_model(config, device, num_classes)
    except Exception as e:
        logger.error(f"模型构建失败: {e}")
        raise

    # 损失函数、优化器
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer(model, config)

    # 训练历史
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    val_metrics = {"loss": 0.0, "accuracy": 0.0}  # 默认值，防止 0 轮训练时未绑定

    logger.info("=" * 60)
    logger.info(f"开始训练 - {config['num_epochs']} 轮")
    logger.info("=" * 60)

    start_time = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # 训练
        try:
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        except RuntimeError as e:
            logger.error(f"Epoch {epoch} 训练出错: {e}")
            if "out of memory" in str(e).lower():
                logger.error("GPU 显存不足，建议减小 batch_size 或使用 CPU。")
            raise

        # 验证
        val_metrics = validate(model, val_loader, criterion, device)

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 记录历史
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch:>2}/{config['num_epochs']}] "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # 保存最佳模型
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": config,
                "class_to_idx": class_to_idx,
            }, output_dir / "best_model.pth")
            logger.info(f"  ★ 新最佳模型已保存 (Val Acc: {best_val_acc:.2f}%)")

    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"训练完成！总耗时: {total_time:.1f}s")
    logger.info(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    logger.info(f"模型保存路径: {output_dir}")
    logger.info("=" * 60)

    # 保存训练历史
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 保存训练曲线
    try:
        save_training_curves(history, output_dir)
    except Exception as e:
        logger.warning(f"训练曲线保存失败（不影响主流程）: {e}")

    # 保存最终模型
    torch.save({
        "epoch": config["num_epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_metrics["accuracy"],
        "config": config,
        "class_to_idx": class_to_idx,
    }, output_dir / "final_model.pth")

    # ── 测试集评估 ──
    test_metrics = None
    if test_loader is not None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("开始测试集评估")
        logger.info("=" * 60)
        
        # 加载最佳模型进行测试
        best_model_path = output_dir / "best_model.pth"
        if best_model_path.exists():
            logger.info(f"加载最佳模型进行测试: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
        
        test_metrics = validate(model, test_loader, criterion, device)
        logger.info(f"测试集结果 - Loss: {test_metrics['loss']:.4f} Acc: {test_metrics['accuracy']:.2f}%")
        
        # 保存测试集指标
        test_results = {
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
        }
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"测试集结果已保存: {output_dir / 'test_results.json'}")
        
        # 对比验证集和测试集准确率
        acc_diff = best_val_acc - test_metrics["accuracy"]
        if abs(acc_diff) > 5.0:
            logger.warning(
                f"⚠️ 验证集准确率 ({best_val_acc:.2f}%) 与测试集准确率 ({test_metrics['accuracy']:.2f}%) "
                f"差距较大 ({acc_diff:+.2f}%)，可能存在过拟合或数据泄露问题"
            )
        else:
            logger.info(
                f"✓ 验证集与测试集准确率差距合理 ({acc_diff:+.2f}%)"
            )
    else:
        logger.warning("测试集不可用，跳过测试集评估")

    return output_dir, best_val_acc, test_metrics


def main():
    args = parse_args()
    config = build_config(args)

    logger.info("训练配置:")
    for k, v in sorted(config.items()):
        logger.info(f"  {k}: {v}")

    output_dir, best_acc, test_metrics = train(config)

    logger.info(f"\n训练结果保存在: {output_dir}")
    logger.info(f"最佳验证准确率: {best_acc:.2f}%")
    if test_metrics is not None:
        logger.info(f"测试集准确率: {test_metrics['accuracy']:.2f}%")

    return output_dir, best_acc, test_metrics


if __name__ == "__main__":
    main()
