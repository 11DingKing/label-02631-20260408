"""
共享 pytest fixtures
"""

import shutil
import random
from pathlib import Path

import pytest
from PIL import Image

CLASSES = ["cat", "dog", "tiger", "lion"]


@pytest.fixture
def tmp_dataset(tmp_path):
    """
    创建一个最小化的临时数据集用于测试。
    train: 每类 4 张, val: 每类 6 张 (抽取 1/3 = 2 张到 test)
    """
    for split, count in [("train", 4), ("val", 6)]:
        for cls in CLASSES:
            cls_dir = tmp_path / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(count):
                # 每类用不同颜色，确保图片内容不同
                color_map = {"cat": "red", "dog": "blue", "tiger": "orange", "lion": "yellow"}
                img = Image.new("RGB", (64, 64), color_map[cls])
                img.save(cls_dir / f"{cls}_{split}_{i:04d}.jpg")
    return tmp_path


@pytest.fixture
def tmp_dataset_with_test(tmp_dataset):
    """
    创建包含 test 集的临时数据集。
    train: 每类 4 张, val: 每类 4 张, test: 每类 2 张
    """
    for cls in CLASSES:
        test_dir = tmp_dataset / "test" / cls
        test_dir.mkdir(parents=True)
        for i in range(2):
            color_map = {"cat": "red", "dog": "blue", "tiger": "orange", "lion": "yellow"}
            img = Image.new("RGB", (64, 64), color_map[cls])
            img.save(test_dir / f"{cls}_test_{i:04d}.jpg")
    return tmp_dataset


@pytest.fixture
def tiny_model_checkpoint(tmp_path, tmp_dataset_with_test):
    """
    创建一个微型 VGG16 checkpoint 用于评估测试。
    使用未训练的模型，仅验证流程正确性。
    动态确定 num_classes 从 class_to_idx。
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    class_to_idx = {"cat": 0, "dog": 1, "lion": 2, "tiger": 3}
    num_classes = len(class_to_idx)

    model = models.vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes),
    )

    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_accuracy": 25.0,
        "config": {"dropout": 0.5},
        "class_to_idx": class_to_idx,
    }

    ckpt_path = tmp_path / "test_model.pth"
    torch.save(checkpoint, ckpt_path)
    return ckpt_path, tmp_dataset_with_test, class_to_idx
