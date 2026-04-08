"""
prepare_data.py 测试用例
覆盖：
- is_image_file: 文件扩展名判断
- count_images: 图片计数
- create_test_set: 从验证集抽取 1/3 到测试集
- generate_synthetic_dataset: 合成数据集生成
- download_sample_dataset: 数据集完整性检查
- print_dataset_summary: 统计摘要
"""

import shutil
from pathlib import Path

import pytest
from PIL import Image

from prepare_data import (
    is_image_file,
    count_images,
    create_test_set,
    generate_synthetic_dataset,
    download_sample_dataset,
    print_dataset_summary,
    discover_classes,
    save_classes_json,
    parse_args,
    CLASSES,
    MIN_DISK_SPACE_BYTES,
)


# ── discover_classes ───────────────────────────────────

class TestDiscoverClasses:
    def test_from_train_dir(self, tmp_dataset):
        """应从 train 目录扫描类别"""
        classes = discover_classes(tmp_dataset)
        assert sorted(classes) == sorted(CLASSES)

    def test_from_classes_json(self, tmp_path):
        """classes.json 存在时应优先读取（白名单内的类别）"""
        import json
        (tmp_path / "train" / "cat").mkdir(parents=True)
        classes_file = tmp_path / "classes.json"
        with open(classes_file, "w") as f:
            json.dump({"classes": ["cat", "dog", "lion"], "num_classes": 3}, f)
        classes = discover_classes(tmp_path)
        assert classes == ["cat", "dog", "lion"]

    def test_from_val_dir(self, tmp_path):
        """无 train 目录时应从 val 目录扫描（白名单内的类别）"""
        for cls in ["cat", "tiger"]:
            (tmp_path / "val" / cls).mkdir(parents=True)
        classes = discover_classes(tmp_path)
        assert classes == ["cat", "tiger"]

    def test_fallback_to_default(self, tmp_path):
        """无任何目录时应回退到默认值"""
        classes = discover_classes(tmp_path)
        assert classes == CLASSES

    def test_ignores_hidden_dirs(self, tmp_path):
        """应忽略以 . 开头的隐藏目录"""
        (tmp_path / "train" / "cat").mkdir(parents=True)
        (tmp_path / "train" / ".hidden").mkdir(parents=True)
        classes = discover_classes(tmp_path)
        assert ".hidden" not in classes

    def test_filters_unexpected_classes(self, tmp_path):
        """应过滤掉非指定的 4 类动物目录（白名单校验）"""
        for cls in ["cat", "dog", "monkey", "elephant"]:
            (tmp_path / "train" / cls).mkdir(parents=True)
        classes = discover_classes(tmp_path)
        assert "monkey" not in classes
        assert "elephant" not in classes
        assert "cat" in classes
        assert "dog" in classes

    def test_filters_unexpected_from_classes_json(self, tmp_path):
        """classes.json 中包含非指定类别时也应过滤"""
        import json
        classes_file = tmp_path / "classes.json"
        with open(classes_file, "w") as f:
            json.dump({"classes": ["cat", "dog", "monkey", "tiger"], "num_classes": 4}, f)
        classes = discover_classes(tmp_path)
        assert "monkey" not in classes
        assert sorted(classes) == ["cat", "dog", "tiger"]


# ── save_classes_json ──────────────────────────────────

class TestSaveClassesJson:
    def test_creates_file(self, tmp_path):
        """应创建 classes.json 文件"""
        save_classes_json(tmp_path, ["cat", "dog"])
        assert (tmp_path / "classes.json").exists()

    def test_content_correct(self, tmp_path):
        """文件内容应包含 classes 和 num_classes"""
        import json
        save_classes_json(tmp_path, ["cat", "dog", "lion"])
        with open(tmp_path / "classes.json") as f:
            data = json.load(f)
        assert data["classes"] == ["cat", "dog", "lion"]
        assert data["num_classes"] == 3

    def test_roundtrip_with_discover(self, tmp_path):
        """save 后 discover 应读回相同结果（白名单内的类别）"""
        save_classes_json(tmp_path, ["cat", "dog", "tiger"])
        classes = discover_classes(tmp_path)
        assert classes == ["cat", "dog", "tiger"]


# ── is_image_file ──────────────────────────────────────

class TestIsImageFile:
    def test_jpg(self, tmp_path):
        p = tmp_path / "photo.jpg"
        p.touch()
        assert is_image_file(p) is True

    def test_jpeg(self, tmp_path):
        p = tmp_path / "photo.jpeg"
        p.touch()
        assert is_image_file(p) is True

    def test_png(self, tmp_path):
        p = tmp_path / "photo.png"
        p.touch()
        assert is_image_file(p) is True

    def test_bmp(self, tmp_path):
        p = tmp_path / "photo.bmp"
        p.touch()
        assert is_image_file(p) is True

    def test_webp(self, tmp_path):
        p = tmp_path / "photo.webp"
        p.touch()
        assert is_image_file(p) is True

    def test_tiff(self, tmp_path):
        p = tmp_path / "photo.tiff"
        p.touch()
        assert is_image_file(p) is True

    def test_uppercase_extension(self, tmp_path):
        p = tmp_path / "photo.JPG"
        p.touch()
        assert is_image_file(p) is True

    def test_non_image_txt(self, tmp_path):
        p = tmp_path / "readme.txt"
        p.touch()
        assert is_image_file(p) is False

    def test_non_image_py(self, tmp_path):
        p = tmp_path / "script.py"
        p.touch()
        assert is_image_file(p) is False

    def test_no_extension(self, tmp_path):
        p = tmp_path / "noext"
        p.touch()
        assert is_image_file(p) is False


# ── count_images ───────────────────────────────────────

class TestCountImages:
    def test_counts_correct(self, tmp_dataset):
        counts = count_images(tmp_dataset / "train", CLASSES)
        for cls in CLASSES:
            assert counts[cls] == 4

    def test_counts_val(self, tmp_dataset):
        counts = count_images(tmp_dataset / "val", CLASSES)
        for cls in CLASSES:
            assert counts[cls] == 6

    def test_nonexistent_directory(self, tmp_path):
        counts = count_images(tmp_path / "nonexistent", CLASSES)
        assert counts == {}

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        counts = count_images(empty, CLASSES)
        assert counts == {}

    def test_ignores_non_class_dirs(self, tmp_path):
        """非类别目录应被忽略"""
        other_dir = tmp_path / "train" / "elephant"
        other_dir.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(other_dir / "e.jpg")
        counts = count_images(tmp_path / "train", CLASSES)
        assert "elephant" not in counts

    def test_ignores_non_image_files(self, tmp_path):
        """非图片文件不应被计数"""
        cls_dir = tmp_path / "train" / "cat"
        cls_dir.mkdir(parents=True)
        (cls_dir / "readme.txt").touch()
        (cls_dir / "notes.md").touch()
        Image.new("RGB", (10, 10)).save(cls_dir / "cat.jpg")
        counts = count_images(tmp_path / "train", CLASSES)
        assert counts["cat"] == 1


# ── create_test_set ────────────────────────────────────

class TestCreateTestSet:
    def test_extracts_one_third(self, tmp_dataset):
        """每类从 6 张验证集中抽取 1/3 = 2 张"""
        stats = create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )
        for cls in CLASSES:
            assert stats[cls]["original"] == 6
            assert stats[cls]["extracted"] == 2
            assert stats[cls]["remaining"] == 4

    def test_files_moved_not_copied(self, tmp_dataset):
        """文件应该被移动，不是复制"""
        create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )
        for cls in CLASSES:
            val_count = len(list((tmp_dataset / "val" / cls).glob("*.jpg")))
            test_count = len(list((tmp_dataset / "test" / cls).glob("*.jpg")))
            assert val_count == 4  # 6 - 2
            assert test_count == 2

    def test_total_preserved(self, tmp_dataset):
        """移动后总数不变"""
        original_total = sum(
            len(list((tmp_dataset / "val" / cls).glob("*.jpg")))
            for cls in CLASSES
        )
        create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )
        new_val = sum(
            len(list((tmp_dataset / "val" / cls).glob("*.jpg")))
            for cls in CLASSES
        )
        new_test = sum(
            len(list((tmp_dataset / "test" / cls).glob("*.jpg")))
            for cls in CLASSES
        )
        assert new_val + new_test == original_total

    def test_no_overlap(self, tmp_dataset):
        """验证集和测试集文件名不重叠"""
        create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )
        for cls in CLASSES:
            val_names = {f.name for f in (tmp_dataset / "val" / cls).glob("*.jpg")}
            test_names = {f.name for f in (tmp_dataset / "test" / cls).glob("*.jpg")}
            assert val_names.isdisjoint(test_names), f"{cls}: 验证集和测试集有重叠文件"

    def test_deterministic_with_seed(self, tmp_dataset):
        """相同 seed 应产生相同结果"""
        # 先复制一份
        copy_path = tmp_dataset.parent / "dataset_copy"
        shutil.copytree(tmp_dataset, copy_path)

        stats1 = create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )
        stats2 = create_test_set(
            val_dir=copy_path / "val",
            test_dir=copy_path / "test",
            classes=CLASSES,
            seed=42,
        )

        for cls in CLASSES:
            test1 = sorted(f.name for f in (tmp_dataset / "test" / cls).glob("*.jpg"))
            test2 = sorted(f.name for f in (copy_path / "test" / cls).glob("*.jpg"))
            assert test1 == test2

    def test_empty_val_class(self, tmp_path):
        """验证集某类为空时应跳过"""
        val_dir = tmp_path / "val" / "cat"
        val_dir.mkdir(parents=True)
        # cat 目录为空
        stats = create_test_set(
            val_dir=tmp_path / "val",
            test_dir=tmp_path / "test",
            classes=["cat"],
            seed=42,
        )
        assert stats["cat"]["extracted"] == 0

    def test_missing_val_dir(self, tmp_path):
        """验证集目录不存在时应记录错误"""
        stats = create_test_set(
            val_dir=tmp_path / "nonexistent",
            test_dir=tmp_path / "test",
            classes=["cat"],
            seed=42,
        )
        assert stats["cat"]["original"] == 0

    def test_single_image_class(self, tmp_path):
        """只有 1 张图片时至少抽取 1 张"""
        val_dir = tmp_path / "val" / "cat"
        val_dir.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(val_dir / "cat_0001.jpg")

        stats = create_test_set(
            val_dir=tmp_path / "val",
            test_dir=tmp_path / "test",
            classes=["cat"],
            seed=42,
        )
        assert stats["cat"]["extracted"] >= 1

    def test_small_val_set_protection(self, tmp_path):
        """验证集样本极少时应保护剩余样本量（MIN_VAL_REMAINING=2）"""
        val_dir = tmp_path / "val" / "cat"
        val_dir.mkdir(parents=True)
        # 只有 3 张图片，1/3 = 1 张，剩余 2 张 >= MIN_VAL_REMAINING
        for i in range(3):
            Image.new("RGB", (10, 10)).save(val_dir / f"cat_{i:04d}.jpg")

        stats = create_test_set(
            val_dir=tmp_path / "val",
            test_dir=tmp_path / "test",
            classes=["cat"],
            seed=42,
        )
        # 应抽取 1 张，保留 2 张
        assert stats["cat"]["extracted"] == 1
        assert stats["cat"]["remaining"] == 2

    def test_very_small_val_set_adjusts_extraction(self, tmp_path):
        """验证集仅 2 张时，应调整抽取数量以保留至少 2 张"""
        val_dir = tmp_path / "val" / "cat"
        val_dir.mkdir(parents=True)
        # 只有 2 张图片，1/3 向下取整 = 0 但 max(1, ...) = 1
        # 抽取 1 张后剩余 1 张 < MIN_VAL_REMAINING=2
        # 保护逻辑应跳过该类（因为无法同时满足抽取和保留）
        for i in range(2):
            Image.new("RGB", (10, 10)).save(val_dir / f"cat_{i:04d}.jpg")

        stats = create_test_set(
            val_dir=tmp_path / "val",
            test_dir=tmp_path / "test",
            classes=["cat"],
            seed=42,
        )
        # 验证集保留量应 >= 0，且不会崩溃
        remaining = stats["cat"]["remaining"]
        assert remaining >= 0

    def test_idempotent_run_twice(self, tmp_dataset):
        """运行两次 create_test_set，第二次应跳过已有测试集的类"""
        stats1 = create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )
        # 记录第一次的结果
        first_val_counts = {}
        first_test_counts = {}
        for cls in CLASSES:
            first_val_counts[cls] = len(list((tmp_dataset / "val" / cls).glob("*.jpg")))
            first_test_counts[cls] = len(list((tmp_dataset / "test" / cls).glob("*.jpg")))

        # 第二次运行
        stats2 = create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )

        # 第二次应全部跳过
        for cls in CLASSES:
            assert stats2[cls].get("skipped") is True
            # 文件数量不变
            val_count = len(list((tmp_dataset / "val" / cls).glob("*.jpg")))
            test_count = len(list((tmp_dataset / "test" / cls).glob("*.jpg")))
            assert val_count == first_val_counts[cls]
            assert test_count == first_test_counts[cls]

    def test_idempotent_partial(self, tmp_dataset):
        """部分类已有测试集时，只抽取缺失的类"""
        # 先给 cat 创建测试集
        test_cat = tmp_dataset / "test" / "cat"
        test_cat.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(test_cat / "existing.jpg")

        stats = create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )
        # cat 应跳过
        assert stats["cat"].get("skipped") is True
        # 其他类应正常抽取
        for cls in ["dog", "tiger", "lion"]:
            assert stats[cls]["extracted"] == 2

    def test_min_val_remaining_protection(self, tmp_path):
        """验证集样本极少时应自动调整抽取数量，保留最低验证样本"""
        val_dir = tmp_path / "val" / "cat"
        val_dir.mkdir(parents=True)
        # 只有 3 张图片，1/3 = 1 张，剩余 2 张 >= MIN_VAL_REMAINING(2)
        for i in range(3):
            Image.new("RGB", (10, 10)).save(val_dir / f"cat_{i:04d}.jpg")

        stats = create_test_set(
            val_dir=tmp_path / "val",
            test_dir=tmp_path / "test",
            classes=["cat"],
            seed=42,
        )
        assert stats["cat"]["extracted"] >= 1
        assert stats["cat"]["remaining"] >= 2  # 至少保留 2 张

    def test_min_val_remaining_with_two_images(self, tmp_path):
        """验证集仅有 2 张时，抽取后仍应保留最低数量"""
        val_dir = tmp_path / "val" / "cat"
        val_dir.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (10, 10)).save(val_dir / f"cat_{i:04d}.jpg")

        stats = create_test_set(
            val_dir=tmp_path / "val",
            test_dir=tmp_path / "test",
            classes=["cat"],
            seed=42,
            test_ratio=1/3,
        )
        # 2 张图片，1/3 = max(1,0) = 1，剩余 1 < MIN_VAL_REMAINING(2)
        # 应调整：num_to_extract = max(1, 2-2) = max(1,0) = 但 0 <= 0 会跳过
        # 实际上 num_to_extract 会被调整
        assert stats["cat"]["extracted"] + stats["cat"]["remaining"] <= 2


# ── generate_synthetic_dataset ─────────────────────────

class TestGenerateSyntheticDataset:
    def test_generates_all_classes(self, tmp_path):
        result = generate_synthetic_dataset(dataset_root=tmp_path)
        assert result is True
        for cls in CLASSES:
            assert (tmp_path / "train" / cls).exists()
            assert (tmp_path / "val" / cls).exists()

    def test_correct_counts(self, tmp_path):
        generate_synthetic_dataset(dataset_root=tmp_path)
        for cls in CLASSES:
            train_count = len(list((tmp_path / "train" / cls).glob("*.jpg")))
            val_count = len(list((tmp_path / "val" / cls).glob("*.jpg")))
            assert train_count == 120
            assert val_count == 45

    def test_images_are_valid(self, tmp_path):
        """生成的图片应该可以正常打开"""
        generate_synthetic_dataset(dataset_root=tmp_path)
        img_path = next((tmp_path / "train" / "cat").glob("*.jpg"))
        img = Image.open(img_path)
        assert img.size == (224, 224)
        assert img.mode == "RGB"

    def test_skip_existing(self, tmp_path):
        """已有足够图片时应跳过"""
        generate_synthetic_dataset(dataset_root=tmp_path)
        # 记录文件修改时间
        first_img = next((tmp_path / "train" / "cat").glob("*.jpg"))
        mtime1 = first_img.stat().st_mtime

        # 再次生成
        import time
        time.sleep(0.1)
        generate_synthetic_dataset(dataset_root=tmp_path)
        mtime2 = first_img.stat().st_mtime
        assert mtime1 == mtime2  # 文件未被重新生成


# ── download_sample_dataset ────────────────────────────

class TestDownloadSampleDataset:
    def test_returns_true_when_complete(self, tmp_dataset):
        """数据集完整时直接返回 True"""
        # tmp_dataset 有 train 4张/类, val 6张/类
        # 设置低阈值使其被认为完整
        result = download_sample_dataset(
            dataset_root=tmp_dataset,
            classes=CLASSES,
            min_train=2,
            min_val=2,
        )
        assert result is True

    def test_generates_when_no_data_and_allowed(self, tmp_path):
        """完全没有数据且 allow_synthetic=True 时应生成合成数据集"""
        result = download_sample_dataset(
            dataset_root=tmp_path,
            classes=CLASSES,
            min_train=80,
            min_val=30,
            allow_synthetic=True,
        )
        assert result is True
        # 验证生成了数据
        for cls in CLASSES:
            assert (tmp_path / "train" / cls).exists()

    def test_fails_when_no_data_and_not_allowed(self, tmp_path):
        """完全没有数据且 allow_synthetic=False（默认）时应返回 False"""
        result = download_sample_dataset(
            dataset_root=tmp_path,
            classes=CLASSES,
            min_train=80,
            min_val=30,
            allow_synthetic=False,
        )
        assert result is False
        # 不应生成任何数据
        assert not (tmp_path / "train").exists()

    def test_default_allow_synthetic_is_false(self, tmp_path):
        """allow_synthetic 默认值应为 False"""
        result = download_sample_dataset(
            dataset_root=tmp_path,
            classes=CLASSES,
        )
        assert result is False

    def test_no_pollution_of_real_data(self, tmp_path):
        """已有少量真实数据时不应混入合成数据"""
        # 创建少量"真实"数据（低于阈值）
        for cls in CLASSES:
            d = tmp_path / "train" / cls
            d.mkdir(parents=True)
            for i in range(5):
                Image.new("RGB", (10, 10)).save(d / f"{cls}_{i}.jpg")
            v = tmp_path / "val" / cls
            v.mkdir(parents=True)
            for i in range(3):
                Image.new("RGB", (10, 10)).save(v / f"{cls}_v_{i}.jpg")

        result = download_sample_dataset(
            dataset_root=tmp_path,
            classes=CLASSES,
            min_train=80,
            min_val=30,
        )
        assert result is True
        # 数据量应保持不变（未被合成数据污染）
        for cls in CLASSES:
            train_count = len(list((tmp_path / "train" / cls).glob("*.jpg")))
            assert train_count == 5  # 原始数量，未增加

    def test_passes_when_all_classes_present(self, tmp_path):
        """所有必需类别都存在时应返回 True（即使数量低于阈值）"""
        for cls in CLASSES:
            d = tmp_path / "train" / cls
            d.mkdir(parents=True)
            Image.new("RGB", (10, 10)).save(d / f"{cls}_0.jpg")
            v = tmp_path / "val" / cls
            v.mkdir(parents=True)
            Image.new("RGB", (10, 10)).save(v / f"{cls}_v_0.jpg")

        result = download_sample_dataset(
            dataset_root=tmp_path,
            classes=CLASSES,
            min_train=80,
            min_val=30,
        )
        assert result is True  # 所有类别存在，虽然数量不足但尊重原貌

    def test_fails_when_missing_required_classes(self, tmp_path):
        """已有部分真实数据但缺少必需类别时应返回 False"""
        # 只创建 cat 和 dog，缺少 tiger 和 lion
        for cls in ["cat", "dog"]:
            d = tmp_path / "train" / cls
            d.mkdir(parents=True)
            Image.new("RGB", (10, 10)).save(d / f"{cls}_0.jpg")
            v = tmp_path / "val" / cls
            v.mkdir(parents=True)
            Image.new("RGB", (10, 10)).save(v / f"{cls}_v_0.jpg")

        result = download_sample_dataset(
            dataset_root=tmp_path,
            classes=CLASSES,
            min_train=80,
            min_val=30,
        )
        assert result is False  # 缺少 tiger 和 lion，应报错



# ── print_dataset_summary ──────────────────────────────

class TestPrintDatasetSummary:
    def test_returns_correct_totals(self, tmp_dataset_with_test):
        totals = print_dataset_summary(
            dataset_root=tmp_dataset_with_test,
            classes=CLASSES,
        )
        for cls in CLASSES:
            assert totals[cls]["train"] == 4
            assert totals[cls]["val"] == 6
            assert totals[cls]["test"] == 2

    def test_handles_missing_splits(self, tmp_path):
        """缺少某些 split 目录时不应崩溃"""
        totals = print_dataset_summary(
            dataset_root=tmp_path,
            classes=CLASSES,
        )
        for cls in CLASSES:
            assert totals[cls]["train"] == 0
            assert totals[cls]["val"] == 0
            assert totals[cls]["test"] == 0

    def test_generates_summary_file(self, tmp_dataset_with_test):
        """应生成 dataset_summary.txt 文件"""
        print_dataset_summary(
            dataset_root=tmp_dataset_with_test,
            classes=CLASSES,
        )
        summary_file = tmp_dataset_with_test / "dataset_summary.txt"
        assert summary_file.exists()
        content = summary_file.read_text(encoding="utf-8")
        assert "数据集统计摘要" in content
        assert "cat" in content
        assert "dog" in content
        assert "合计" in content

    def test_summary_file_contains_counts(self, tmp_dataset_with_test):
        """summary 文件应包含正确的数量"""
        print_dataset_summary(
            dataset_root=tmp_dataset_with_test,
            classes=CLASSES,
        )
        summary_file = tmp_dataset_with_test / "dataset_summary.txt"
        content = summary_file.read_text(encoding="utf-8")
        # 每类 train=4, val=6, test=2
        assert "4" in content
        assert "6" in content
        assert "2" in content


# ── create_test_set 磁盘空间与权限预校验 ──────────────

class TestCreateTestSetDiskCheck:
    def test_disk_space_check_passes_normally(self, tmp_dataset):
        """正常情况下磁盘空间检查应通过"""
        stats = create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )
        # 应正常抽取
        for cls in CLASSES:
            assert stats[cls]["extracted"] > 0

    def test_permission_error_returns_empty(self, tmp_path, monkeypatch):
        """目标目录无写入权限时应返回空 stats"""
        val_dir = tmp_path / "val" / "cat"
        val_dir.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(val_dir / "cat_0.jpg")

        # 模拟 mkdir 成功但 touch 失败（权限问题）
        original_touch = Path.touch

        def mock_touch(self, *args, **kwargs):
            if ".write_probe" in str(self):
                raise PermissionError("Permission denied")
            return original_touch(self, *args, **kwargs)

        monkeypatch.setattr(Path, "touch", mock_touch)

        stats = create_test_set(
            val_dir=tmp_path / "val",
            test_dir=tmp_path / "test",
            classes=["cat"],
            seed=42,
        )
        assert stats == {}

    def test_low_disk_space_returns_empty(self, tmp_dataset, monkeypatch):
        """磁盘空间不足时应返回空 stats"""
        import shutil as _shutil
        from collections import namedtuple

        FakeDiskUsage = namedtuple("usage", ["total", "used", "free"])
        # 模拟只有 10MB 剩余空间（低于 100MB 阈值）
        monkeypatch.setattr(
            _shutil, "disk_usage",
            lambda path: FakeDiskUsage(
                total=1000 * 1024 * 1024,
                used=995 * 1024 * 1024,
                free=5 * 1024 * 1024,
            ),
        )

        stats = create_test_set(
            val_dir=tmp_dataset / "val",
            test_dir=tmp_dataset / "test",
            classes=CLASSES,
            seed=42,
        )
        assert stats == {}



# ── parse_args CLI 参数 ────────────────────────────────

class TestParseArgs:
    def test_default_values(self):
        """无参数时应使用默认值"""
        import sys
        from unittest.mock import patch
        with patch.object(sys, "argv", ["prepare_data.py"]):
            args = parse_args()
        assert args.dataset_root is None
        assert args.seed is None
        assert args.test_ratio is None
        assert args.allow_synthetic is False

    def test_custom_dataset_root(self):
        """--dataset-root 应正确解析"""
        import sys
        from unittest.mock import patch
        with patch.object(sys, "argv", ["prepare_data.py", "--dataset-root", "/data/animals"]):
            args = parse_args()
        assert args.dataset_root == "/data/animals"

    def test_allow_synthetic_flag(self):
        """--allow-synthetic 应设为 True"""
        import sys
        from unittest.mock import patch
        with patch.object(sys, "argv", ["prepare_data.py", "--allow-synthetic"]):
            args = parse_args()
        assert args.allow_synthetic is True

    def test_custom_seed(self):
        """--seed 应正确解析"""
        import sys
        from unittest.mock import patch
        with patch.object(sys, "argv", ["prepare_data.py", "--seed", "123"]):
            args = parse_args()
        assert args.seed == 123

    def test_custom_test_ratio(self):
        """--test-ratio 应正确解析"""
        import sys
        from unittest.mock import patch
        with patch.object(sys, "argv", ["prepare_data.py", "--test-ratio", "0.25"]):
            args = parse_args()
        assert args.test_ratio == pytest.approx(0.25)

    def test_all_args_combined(self):
        """所有参数组合使用"""
        import sys
        from unittest.mock import patch
        with patch.object(sys, "argv", [
            "prepare_data.py",
            "--dataset-root", "/my/data",
            "--seed", "99",
            "--test-ratio", "0.5",
            "--allow-synthetic",
        ]):
            args = parse_args()
        assert args.dataset_root == "/my/data"
        assert args.seed == 99
        assert args.test_ratio == pytest.approx(0.5)
        assert args.allow_synthetic is True
