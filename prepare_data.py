"""
数据准备脚本
功能：
1. 下载/检查 4 类动物数据集 (cat, dog, tiger, lion)
2. 从验证集每类抽取 1/3 图片到测试集
3. 统计并报告每个文件夹的图片数量
"""

import os
import sys
import shutil
import random
import logging
import argparse
from pathlib import Path
from typing import Dict

from log_config import setup_logger

# 磁盘空间预检查最低要求（字节）
MIN_DISK_SPACE_BYTES = 50 * 1024 * 1024  # 50MB

logger = setup_logger(__name__, "data_prepare.log")

# ── 默认配置（可通过 CLI 参数覆盖） ──────────────────
DATASET_ROOT = Path("dataset")
CLASSES = ["cat", "dog", "tiger", "lion"]
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"
RANDOM_SEED = 42
TEST_RATIO = 1 / 3  # 从验证集每类抽取 1/3

# 每类最少需要的图片数量
MIN_IMAGES_PER_CLASS_TRAIN = 80
MIN_IMAGES_PER_CLASS_VAL = 30

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def discover_classes(dataset_root: Path) -> list:
    """
    从数据集目录结构中动态发现类别名称。
    优先读取 classes.json，否则从 train/ 目录扫描子文件夹。
    最终结果会与 CLASSES 白名单取交集，过滤掉非指定类别，防止数据污染。
    """
    classes_file = dataset_root / "classes.json"
    if classes_file.exists():
        with open(classes_file, encoding="utf-8") as f:
            import json
            data = json.load(f)
        raw = data["classes"]
    elif (dataset_root / "train").exists():
        # 从 train 目录扫描
        raw = sorted(
            d.name for d in (dataset_root / "train").iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    elif (dataset_root / "val").exists():
        # 从 val 目录扫描
        raw = sorted(
            d.name for d in (dataset_root / "val").iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    else:
        return list(CLASSES)  # 最终回退到默认值

    # ── 白名单校验：仅保留 Prompt 指定的 4 类动物 ──
    allowed = set(CLASSES)
    unexpected = [c for c in raw if c not in allowed]
    if unexpected:
        logger.warning(
            f"发现非指定类别目录: {unexpected}，已自动过滤。"
            f"Prompt 仅要求 4 类: {CLASSES}"
        )
    validated = [c for c in raw if c in allowed]

    if not validated:
        logger.warning("过滤后无有效类别，回退到默认值。")
        return list(CLASSES)

    return validated


def save_classes_json(dataset_root: Path, classes: list):
    """保存类别元数据到 classes.json，作为单一数据源"""
    import json
    classes_file = dataset_root / "classes.json"
    with open(classes_file, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "num_classes": len(classes)}, f, indent=2, ensure_ascii=False)
    logger.info(f"类别元数据已保存: {classes_file} ({len(classes)} 类: {classes})")


def is_image_file(filepath: Path) -> bool:
    """判断文件是否为支持的图片格式"""
    return filepath.suffix.lower() in SUPPORTED_EXTENSIONS


def count_images(directory: Path, classes: list = None) -> Dict[str, int]:
    """统计目录下每个类别的图片数量"""
    if classes is None:
        classes = CLASSES
    counts = {}
    if not directory.exists():
        return counts
    for cls_dir in sorted(directory.iterdir()):
        if cls_dir.is_dir() and cls_dir.name in classes:
            count = sum(1 for f in cls_dir.iterdir() if is_image_file(f))
            counts[cls_dir.name] = count
    return counts




def download_sample_dataset(
    dataset_root: Path = None,
    classes: list = None,
    min_train: int = None,
    min_val: int = None,
    allow_synthetic: bool = False,
):
    """
    检查数据集是否完整。
    仅当 allow_synthetic=True 且完全无数据时，才生成合成数据集。
    否则在无数据时报错并提示用户配置路径。
    """
    dataset_root = dataset_root or DATASET_ROOT
    classes = classes or CLASSES
    min_train = min_train if min_train is not None else MIN_IMAGES_PER_CLASS_TRAIN
    min_val = min_val if min_val is not None else MIN_IMAGES_PER_CLASS_VAL

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    logger.info("检查数据集是否已存在...")

    train_counts = count_images(train_dir, classes)
    val_counts = count_images(val_dir, classes)

    # 检查是否已有任何真实数据
    has_any_data = any(train_counts.get(cls, 0) > 0 for cls in classes) or \
                   any(val_counts.get(cls, 0) > 0 for cls in classes)

    if has_any_data:
        # ── 硬性校验：所有必需类别必须存在 ──
        missing_classes = []
        for cls in classes:
            t = train_counts.get(cls, 0)
            v = val_counts.get(cls, 0)
            if t == 0 and v == 0:
                missing_classes.append(cls)

        if missing_classes:
            logger.error(
                f"数据集缺少以下必需类别: {missing_classes}。"
                f"Prompt 要求最终数据集包含全部 {len(classes)} 类: {classes}。"
            )
            logger.error("请补充缺失类别的数据后重新运行。")
            return False

        # 已有数据且所有类别存在，尊重原貌，不混入合成数据
        all_classes_ready = True
        for cls in classes:
            if train_counts.get(cls, 0) < min_train or val_counts.get(cls, 0) < min_val:
                all_classes_ready = False

        if all_classes_ready:
            logger.info("数据集已存在且完整，跳过下载。")
        else:
            logger.warning(
                "数据集已存在但部分类别数量低于建议阈值 "
                f"(train>={min_train}, val>={min_val})。"
            )
            logger.warning("为尊重现有数据分布，不自动补充合成数据。如需补充请手动操作。")
            for cls in classes:
                t = train_counts.get(cls, 0)
                v = val_counts.get(cls, 0)
                if t < min_train or v < min_val:
                    logger.warning(f"  {cls}: train={t}, val={v}")
        return True

    # 完全没有数据
    if not allow_synthetic:
        logger.error("未检测到任何数据，且未启用合成数据生成。")
        logger.error(f"请将真实图片放入以下目录结构后重新运行:")
        for cls in classes:
            logger.error(f"  {dataset_root / 'train' / cls}/")
            logger.error(f"  {dataset_root / 'val' / cls}/")
        logger.error("或使用 --allow-synthetic 参数启用合成数据集生成（仅用于演示/测试）。")
        return False

    logger.warning("⚠️ 未检测到任何数据，生成合成数据集用于训练演示...")
    logger.warning("⚠️ 合成数据仅用于验证流水线功能，不代表真实分类效果。")
    return generate_synthetic_dataset(dataset_root=dataset_root)



def _draw_pattern(draw, img_size, pattern, accent, random_module):
    """在图片上绘制类别特征纹理"""
    import math

    if pattern == "stripes":
        stripe_width = random_module.randint(3, 6)
        offset = random_module.randint(0, 10)
        for i in range(offset, img_size, stripe_width * 3):
            stripe_color = tuple(
                max(0, min(255, c + random_module.randint(-10, 10))) for c in accent
            )
            draw.line(
                [(i, 0), (i + random_module.randint(-20, 20), img_size)],
                fill=stripe_color,
                width=stripe_width,
            )
    elif pattern == "spots":
        num_spots = random_module.randint(8, 20)
        for _ in range(num_spots):
            cx = random_module.randint(20, img_size - 20)
            cy = random_module.randint(20, img_size - 20)
            r = random_module.randint(8, 25)
            spot_color = tuple(
                max(0, min(255, c + random_module.randint(-20, 20))) for c in accent
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=spot_color)
    elif pattern == "tiger_stripes":
        num_stripes = random_module.randint(6, 12)
        for _ in range(num_stripes):
            x_start = random_module.randint(0, img_size)
            stripe_color = tuple(
                max(0, min(255, c + random_module.randint(-10, 10))) for c in accent
            )
            points = []
            y = 0
            x = x_start
            while y < img_size:
                x += random_module.randint(-15, 15)
                points.append((x, y))
                y += random_module.randint(10, 30)
            if len(points) >= 2:
                draw.line(
                    points, fill=stripe_color, width=random_module.randint(5, 12)
                )
    elif pattern == "mane":
        cx, cy = img_size // 2, img_size // 2
        num_rays = random_module.randint(20, 40)
        for _ in range(num_rays):
            angle = random_module.uniform(0, 2 * math.pi)
            length = random_module.randint(40, 100)
            ex = cx + int(length * math.cos(angle))
            ey = cy + int(length * math.sin(angle))
            mane_color = tuple(
                max(0, min(255, c + random_module.randint(-20, 20))) for c in accent
            )
            draw.line(
                [(cx, cy), (ex, ey)],
                fill=mane_color,
                width=random_module.randint(2, 5),
            )


def _draw_shape(draw, img_size, shape, accent, random_module):
    """在图片上绘制类别特征形状"""
    cx, cy = img_size // 2, img_size // 2

    if shape == "triangle_ears":
        size = random_module.randint(25, 40)
        draw.polygon(
            [(cx - 50, cy - 30), (cx - 30, cy - 30 - size), (cx - 10, cy - 30)],
            fill=accent,
        )
        draw.polygon(
            [(cx + 10, cy - 30), (cx + 30, cy - 30 - size), (cx + 50, cy - 30)],
            fill=accent,
        )
    elif shape == "floppy_ears":
        draw.ellipse([cx - 70, cy - 20, cx - 30, cy + 50], fill=accent)
        draw.ellipse([cx + 30, cy - 20, cx + 70, cy + 50], fill=accent)
    elif shape == "round_face":
        r = random_module.randint(35, 50)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=accent, width=3)
    elif shape == "mane_circle":
        r = random_module.randint(40, 55)
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r], outline=accent, width=4
        )


def generate_synthetic_dataset(dataset_root: Path = None):
    """
    生成合成图片数据集用于训练演示。
    每类生成不同特征的图片以确保模型可以学习区分。
    """
    dataset_root = dataset_root or DATASET_ROOT

    try:
        from PIL import Image, ImageDraw, ImageFilter
        import numpy as np
    except ImportError:
        logger.error("需要安装 Pillow 和 numpy: pip install Pillow numpy")
        return False

    # 每类的视觉特征配置
    class_configs = {
        "cat": {
            "base_colors": [(200, 180, 160), (180, 160, 140), (220, 200, 180)],
            "accent_color": (100, 200, 100),
            "pattern": "stripes",
            "shape": "triangle_ears",
            "train_count": 120,
            "val_count": 45,
        },
        "dog": {
            "base_colors": [(160, 120, 80), (140, 100, 60), (180, 140, 100)],
            "accent_color": (200, 100, 100),
            "pattern": "spots",
            "shape": "floppy_ears",
            "train_count": 120,
            "val_count": 45,
        },
        "tiger": {
            "base_colors": [(240, 180, 50), (220, 160, 30), (255, 200, 70)],
            "accent_color": (40, 40, 40),
            "pattern": "tiger_stripes",
            "shape": "round_face",
            "train_count": 120,
            "val_count": 45,
        },
        "lion": {
            "base_colors": [(210, 170, 100), (190, 150, 80), (230, 190, 120)],
            "accent_color": (150, 100, 50),
            "pattern": "mane",
            "shape": "mane_circle",
            "train_count": 120,
            "val_count": 45,
        },
    }

    img_size = 224

    def create_animal_image(
        cls_name: str, config: dict, idx: int, split: str, rng: random.Random
    ) -> "Image.Image":
        """生成具有类别特征的合成图片
        
        Args:
            cls_name: 类别名称
            config: 类别配置
            idx: 图片索引
            split: 数据集划分 (train/val)
            rng: 独立的随机数生成器，确保不同split使用不同的随机状态
        """
        base_color = list(rng.choice(config["base_colors"]))
        for i in range(3):
            base_color[i] = max(0, min(255, base_color[i] + rng.randint(-20, 20)))
        base_color = tuple(base_color)

        img = Image.new("RGB", (img_size, img_size), base_color)
        draw = ImageDraw.Draw(img)

        # 添加背景纹理
        for _ in range(rng.randint(50, 150)):
            x = rng.randint(0, img_size - 1)
            y = rng.randint(0, img_size - 1)
            noise_color = tuple(
                max(0, min(255, c + rng.randint(-30, 30))) for c in base_color
            )
            draw.point((x, y), fill=noise_color)

        # 绘制纹理和形状
        _draw_pattern(draw, img_size, config["pattern"], config["accent_color"], rng)
        _draw_shape(draw, img_size, config["shape"], config["accent_color"], rng)

        # 添加随机变换增加多样性
        if rng.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() > 0.7:
            angle = rng.randint(-15, 15)
            img = img.rotate(angle, fillcolor=base_color)
        if rng.random() > 0.6:
            img = img.filter(
                ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0))
            )

        return img

    # 生成数据集
    for cls_name, config in class_configs.items():
        for split, count_key in [("train", "train_count"), ("val", "val_count")]:
            split_dir = dataset_root / split / cls_name
            split_dir.mkdir(parents=True, exist_ok=True)

            existing = (
                sum(1 for f in split_dir.iterdir() if is_image_file(f))
                if split_dir.exists()
                else 0
            )
            needed = config[count_key]

            if existing >= needed:
                logger.info(f"  {split}/{cls_name}: 已有 {existing} 张，跳过")
                continue

            # 为每个类别和split使用独立的随机种子，确保train和val的随机状态完全独立
            # 这样可以防止训练集和验证集的图片特征过于相似
            split_seed = RANDOM_SEED + hash(cls_name) % 1000 + (100 if split == "val" else 0)
            rng = random.Random(split_seed)
            
            logger.info(f"  生成 {split}/{cls_name}: {needed} 张图片 (seed={split_seed})...")
            for i in range(needed):
                img = create_animal_image(cls_name, config, i, split, rng)
                img.save(
                    split_dir / f"{cls_name}_{split}_{i:04d}.jpg", "JPEG", quality=90
                )

    logger.info("合成数据集生成完成。")
    return True


def create_test_set(
    val_dir: Path = None,
    test_dir: Path = None,
    classes: list = None,
    seed: int = None,
    test_ratio: float = None,
) -> Dict[str, Dict[str, int]]:
    """
    从验证集每类抽取 1/3 图片到测试集。
    使用移动操作（非复制），确保测试集与验证集不重叠。

    Returns:
        dict: 每类的抽取统计 {cls: {"original": N, "extracted": M, "remaining": R}}
    """
    val_dir = val_dir or VAL_DIR
    test_dir = test_dir or TEST_DIR
    classes = classes or CLASSES
    seed = seed if seed is not None else RANDOM_SEED
    test_ratio = test_ratio if test_ratio is not None else TEST_RATIO

    logger.info("=" * 60)
    logger.info("开始创建测试集 (从验证集每类抽取 1/3)")
    logger.info("=" * 60)

    random.seed(seed)
    stats = {}

    # ── 磁盘空间与权限预校验 ──
    try:
        disk = shutil.disk_usage(str(val_dir) if val_dir.exists() else str(test_dir.parent))
        free_mb = disk.free / (1024 * 1024)
        if free_mb < 100:
            logger.error(f"磁盘剩余空间不足: {free_mb:.1f} MB (需要至少 100 MB)")
            return stats
        logger.info(f"磁盘剩余空间: {free_mb:.0f} MB")
    except OSError as e:
        logger.warning(f"无法检查磁盘空间: {e}，继续执行...")

    # 测试目标目录写入权限
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        probe = test_dir / ".write_probe"
        probe.touch()
        probe.unlink()
    except PermissionError:
        logger.error(f"目标目录无写入权限: {test_dir}")
        return stats
    except OSError as e:
        logger.error(f"目标目录访问异常: {test_dir} — {e}")
        return stats

    for cls in classes:
        val_cls_dir = val_dir / cls
        test_cls_dir = test_dir / cls

        if not val_cls_dir.exists():
            logger.error(f"验证集目录不存在: {val_cls_dir}")
            stats[cls] = {"original": 0, "extracted": 0, "remaining": 0}
            continue

        test_cls_dir.mkdir(parents=True, exist_ok=True)

        # 幂等性检查：若测试集该类已有图片，跳过
        existing_test = sum(1 for f in test_cls_dir.iterdir() if is_image_file(f))
        if existing_test > 0:
            logger.info(f"  [{cls}] 测试集已有 {existing_test} 张图片，跳过抽取")
            stats[cls] = {"original": 0, "extracted": 0, "remaining": 0, "skipped": True}
            continue

        val_images = sorted([f for f in val_cls_dir.iterdir() if is_image_file(f)])
        total_val = len(val_images)

        if total_val == 0:
            logger.warning(f"验证集 {cls} 类别无图片，跳过")
            stats[cls] = {"original": 0, "extracted": 0, "remaining": 0}
            continue

        # 计算需要抽取的数量 (1/3)
        num_to_extract = max(1, int(total_val * test_ratio))

        # ── 验证集剩余样本量保护 ──
        remaining_after = total_val - num_to_extract
        MIN_VAL_REMAINING = 2  # 验证集每类至少保留 2 张，确保验证环节可用
        if remaining_after < MIN_VAL_REMAINING:
            logger.warning(
                f"  [{cls}] 验证集仅有 {total_val} 张，抽取 {num_to_extract} 张后"
                f"仅剩 {remaining_after} 张，低于最低保留量 {MIN_VAL_REMAINING}。"
            )
            # 调整抽取数量，确保验证集至少保留 MIN_VAL_REMAINING 张
            num_to_extract = max(1, total_val - MIN_VAL_REMAINING)
            if num_to_extract <= 0:
                logger.error(
                    f"  [{cls}] 验证集样本量过少 ({total_val} 张)，"
                    f"无法同时满足测试集抽取和验证集最低保留量。跳过该类。"
                )
                stats[cls] = {"original": total_val, "extracted": 0, "remaining": total_val}
                continue
            logger.warning(f"  [{cls}] 已调整抽取数量为 {num_to_extract} 张，保留 {total_val - num_to_extract} 张用于验证。")

        selected = random.sample(val_images, num_to_extract)

        moved_count = 0
        move_failed = False
        for img_path in selected:
            dest = test_cls_dir / img_path.name
            try:
                shutil.move(str(img_path), str(dest))
                moved_count += 1
            except (OSError, shutil.Error) as e:
                logger.error(f"  [{cls}] 移动文件失败 {img_path.name}: {e}")
                # 回滚已移动的文件
                logger.warning(f"  [{cls}] 正在回滚已移动的 {moved_count} 个文件...")
                for rollback_file in test_cls_dir.iterdir():
                    if is_image_file(rollback_file):
                        try:
                            shutil.move(str(rollback_file), str(val_cls_dir / rollback_file.name))
                        except Exception:
                            pass
                stats[cls] = {
                    "original": total_val,
                    "extracted": 0,
                    "remaining": total_val,
                    "error": "move_failed",
                }
                move_failed = True
                break

        if move_failed:
            continue

        remaining_val = total_val - moved_count
        stats[cls] = {
            "original": total_val,
            "extracted": moved_count,
            "remaining": remaining_val,
        }
        logger.info(
            f"  [{cls}] 验证集原有: {total_val} 张 -> "
            f"抽取到测试集: {moved_count} 张, "
            f"验证集剩余: {remaining_val} 张"
        )

    logger.info("测试集创建完成。")
    return stats


def print_dataset_summary(
    dataset_root: Path = None, classes: list = None
) -> Dict[str, Dict[str, int]]:
    """打印数据集统计摘要，返回统计数据"""
    dataset_root = dataset_root or DATASET_ROOT
    classes = classes or CLASSES

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    logger.info("=" * 60)
    logger.info("数据集统计摘要")
    logger.info("=" * 60)

    splits = {"train": train_dir, "val": val_dir, "test": test_dir}
    total_all = 0

    summary_lines = []
    summary_lines.append(
        f"{'类别':<10} {'训练集':>8} {'验证集':>8} {'测试集':>8} {'合计':>8}"
    )
    summary_lines.append("-" * 50)

    class_totals = {cls: {"train": 0, "val": 0, "test": 0} for cls in classes}

    for split_name, split_dir in splits.items():
        counts = count_images(split_dir, classes)
        for cls in classes:
            class_totals[cls][split_name] = counts.get(cls, 0)

    for cls in classes:
        t = class_totals[cls]
        row_total = t["train"] + t["val"] + t["test"]
        total_all += row_total
        summary_lines.append(
            f"{cls:<10} {t['train']:>8} {t['val']:>8} {t['test']:>8} {row_total:>8}"
        )

    summary_lines.append("-" * 50)
    train_total = sum(class_totals[c]["train"] for c in classes)
    val_total = sum(class_totals[c]["val"] for c in classes)
    test_total = sum(class_totals[c]["test"] for c in classes)
    summary_lines.append(
        f"{'合计':<10} {train_total:>8} {val_total:>8} {test_total:>8} {total_all:>8}"
    )

    for line in summary_lines:
        logger.info(line)

    # 写入 dataset_summary.txt 作为正式交付物
    summary_file = dataset_root / "dataset_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("数据集统计摘要\n")
        f.write("=" * 50 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
    logger.info(f"数据集统计已保存: {summary_file}")

    return class_totals


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="动物图像分类 - 数据准备")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="数据集根目录路径 (默认: dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="从验证集抽取到测试集的比例 (默认: 1/3)",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=False,
        help="当数据集不存在时，自动生成合成数据集用于演示",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 从 CLI 参数构建配置，回退到模块级默认值
    dataset_root = Path(args.dataset_root) if args.dataset_root else DATASET_ROOT
    seed = args.seed if args.seed is not None else RANDOM_SEED
    test_ratio = args.test_ratio if args.test_ratio is not None else TEST_RATIO

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    logger.info("=" * 60)
    logger.info("动物图像分类 - 数据准备")
    logger.info(f"数据集路径: {dataset_root.resolve()}")
    logger.info("=" * 60)

    # Step 1: 检查/准备数据集
    success = download_sample_dataset(
        dataset_root=dataset_root,
        allow_synthetic=args.allow_synthetic,
    )
    if not success:
        logger.error("数据集准备失败。")
        sys.exit(1)

    # Step 1.5: 保存类别元数据 (单一数据源)
    classes = discover_classes(dataset_root)
    save_classes_json(dataset_root, classes)

    # Step 2: 创建测试集
    test_counts = count_images(test_dir, classes)
    if all(test_counts.get(cls, 0) > 0 for cls in classes):
        logger.info("测试集已存在，跳过创建。")
    else:
        create_test_set(
            val_dir=val_dir,
            test_dir=test_dir,
            classes=classes,
            seed=seed,
            test_ratio=test_ratio,
        )

    # Step 3: 打印统计摘要
    print_dataset_summary(dataset_root=dataset_root, classes=classes)

    logger.info("数据准备完成！可以开始训练模型。")


if __name__ == "__main__":
    main()
