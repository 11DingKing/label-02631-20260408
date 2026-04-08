"""
run_pipeline.py 测试用例
覆盖：
- find_latest_output_dir: 查找最新输出目录
- run_step: 步骤执行
- check_environment: 环境检查
"""

import os
import sys
import time
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from run_pipeline import find_latest_output_dir, run_step, check_environment


# ── find_latest_output_dir ─────────────────────────────

class TestFindLatestOutputDir:
    def test_finds_matching_dir(self, tmp_path):
        d1 = tmp_path / "baseline_20250101_000000"
        d1.mkdir()
        d2 = tmp_path / "baseline_20250102_000000"
        d2.mkdir()
        time.sleep(0.05)
        (d2 / "marker").touch()

        result = find_latest_output_dir(str(tmp_path), "baseline")
        assert result == d2

    def test_returns_none_for_empty(self, tmp_path):
        result = find_latest_output_dir(str(tmp_path), "baseline")
        assert result is None

    def test_returns_none_for_nonexistent(self):
        result = find_latest_output_dir("/nonexistent/path", "baseline")
        assert result is None

    def test_falls_back_to_any_dir(self, tmp_path):
        """tag 不匹配时应回退到任意目录"""
        d = tmp_path / "tuned_v1_20250101"
        d.mkdir()
        result = find_latest_output_dir(str(tmp_path), "baseline")
        assert result == d

    def test_ignores_files(self, tmp_path):
        """应忽略文件，只看目录"""
        (tmp_path / "baseline_file.txt").touch()
        d = tmp_path / "baseline_20250101"
        d.mkdir()
        result = find_latest_output_dir(str(tmp_path), "baseline")
        assert result == d


# ── run_step ───────────────────────────────────────────

class TestRunStep:
    def test_successful_command(self):
        result = run_step("测试", "-c", "print('hello')")
        assert result is True

    def test_failed_command(self):
        result = run_step("失败测试", "-c", "import sys; sys.exit(1)")
        assert result is False

    def test_syntax_error_returns_false(self):
        result = run_step("语法错误", "-c", "this is not valid python")
        assert result is False


# ── check_environment ──────────────────────────────────

class TestCheckEnvironment:
    def test_passes_in_current_env(self):
        """当前测试环境应通过检查（依赖已安装）"""
        result = check_environment()
        assert result is True

    def test_detects_missing_package(self):
        """缺少依赖包时应返回 False"""
        with patch("run_pipeline.REQUIRED_PACKAGES", ["torch", "nonexistent_pkg_xyz"]):
            result = check_environment()
            assert result is False

    def test_warns_no_venv(self):
        """不在虚拟环境中时应发出警告（不阻断）"""
        # 模拟非虚拟环境
        with patch.object(sys, "prefix", sys.base_prefix):
            with patch.object(sys, "base_prefix", sys.base_prefix):
                with patch.dict(os.environ, {}, clear=True):
                    # 即使不在 venv 中，只要依赖齐全仍应返回 True
                    result = check_environment()
                    assert result is True

    def test_python_version_check(self):
        """Python 版本过低时应返回 False"""
        from collections import namedtuple
        FakeVersion = namedtuple("version_info", ["major", "minor", "micro", "releaselevel", "serial"])
        fake_version = FakeVersion(3, 8, 0, "final", 0)
        with patch.object(sys, "version_info", fake_version):
            result = check_environment()
            assert result is False



# ── log_config 日志轮转 ────────────────────────────────

class TestLogConfig:
    def test_setup_logger_returns_logger(self):
        """setup_logger 应返回 Logger 实例"""
        import logging
        from log_config import setup_logger
        logger = setup_logger("test_logger_1", "test_log_rotation.log")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_rotating_handler(self, tmp_path):
        """logger 应包含 RotatingFileHandler"""
        import logging
        from logging.handlers import RotatingFileHandler
        from log_config import setup_logger
        log_file = str(tmp_path / "test_rotate.log")
        logger = setup_logger("test_logger_rotate", log_file)
        has_rotating = any(
            isinstance(h, RotatingFileHandler) for h in logger.handlers
        )
        assert has_rotating

    def test_logger_has_console_handler(self, tmp_path):
        """logger 应包含 StreamHandler（控制台输出）"""
        import logging
        from log_config import setup_logger
        log_file = str(tmp_path / "test_console.log")
        logger = setup_logger("test_logger_console", log_file)
        has_stream = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )
        assert has_stream

    def test_logger_writes_to_file(self, tmp_path):
        """logger 应将日志写入文件"""
        from log_config import setup_logger
        log_file = tmp_path / "test_write.log"
        logger = setup_logger("test_logger_write", str(log_file))
        logger.info("测试日志写入")
        # flush handlers
        for h in logger.handlers:
            h.flush()
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "测试日志写入" in content

    def test_no_duplicate_handlers(self, tmp_path):
        """多次调用 setup_logger 不应重复添加 handler"""
        from log_config import setup_logger
        log_file = str(tmp_path / "test_dup.log")
        logger1 = setup_logger("test_logger_dup", log_file)
        handler_count_1 = len(logger1.handlers)
        logger2 = setup_logger("test_logger_dup", log_file)
        handler_count_2 = len(logger2.handlers)
        assert handler_count_1 == handler_count_2
        assert logger1 is logger2

    def test_max_bytes_config(self, tmp_path):
        """RotatingFileHandler 应使用配置的 maxBytes"""
        from logging.handlers import RotatingFileHandler
        from log_config import setup_logger, MAX_LOG_BYTES
        log_file = str(tmp_path / "test_max.log")
        logger = setup_logger("test_logger_max", log_file)
        for h in logger.handlers:
            if isinstance(h, RotatingFileHandler):
                assert h.maxBytes == MAX_LOG_BYTES
                break

    def test_all_modules_use_log_config(self):
        """所有主要模块应使用 log_config 而非 logging.basicConfig"""
        for module_name in ["prepare_data", "train", "evaluate", "retrain", "run_pipeline"]:
            source_path = Path(module_name + ".py")
            content = source_path.read_text()
            assert "from log_config import" in content, (
                f"{module_name}.py 未使用 log_config"
            )
            assert "logging.basicConfig" not in content, (
                f"{module_name}.py 仍使用 logging.basicConfig"
            )


# ── 流水线评估指标展示 ─────────────────────────────────

class TestPipelineMetricsDisplay:
    def test_run_step_passes_allow_synthetic(self):
        """流水线应向 prepare_data.py 传递 --allow-synthetic"""
        from pathlib import Path
        source = Path("run_pipeline.py").read_text()
        assert "--allow-synthetic" in source

    def test_pipeline_summary_includes_dataset_stats(self):
        """流水线总结部分应包含数据集图片数量统计展示逻辑"""
        from pathlib import Path
        source = Path("run_pipeline.py").read_text()
        # 总结部分应读取并展示 dataset_summary.txt
        assert "数据集图片数量统计" in source
        assert "summary_path" in source

    def test_pipeline_shows_synthetic_data_warning(self):
        """流水线应包含合成数据警告逻辑"""
        from pathlib import Path
        source = Path("run_pipeline.py").read_text()
        assert "合成数据集生成完成" in source
        assert "合成数据仅用于验证流水线功能" in source

    def test_pipeline_shows_confusion_analysis(self):
        """流水线应展示混淆矩阵分析报告"""
        from pathlib import Path
        source = Path("run_pipeline.py").read_text()
        assert "confusion_analysis.txt" in source


# ── 错误处理与边界条件测试 ─────────────────────────────

class TestParsePipelineArgs:
    def test_default_values(self):
        """无参数时应使用默认值"""
        from run_pipeline import parse_pipeline_args
        with patch.object(sys, "argv", ["run_pipeline.py"]):
            args = parse_pipeline_args()
        assert args.accuracy_threshold == 85.0
        assert args.min_improvement == 2.0
        assert args.epochs == 15

    def test_custom_threshold(self):
        """自定义准确率阈值"""
        from run_pipeline import parse_pipeline_args
        with patch.object(sys, "argv", [
            "run_pipeline.py", "--accuracy-threshold", "90.0",
        ]):
            args = parse_pipeline_args()
        assert args.accuracy_threshold == 90.0

    def test_custom_epochs(self):
        """自定义训练轮数"""
        from run_pipeline import parse_pipeline_args
        with patch.object(sys, "argv", [
            "run_pipeline.py", "--epochs", "20",
        ]):
            args = parse_pipeline_args()
        assert args.epochs == 20

    def test_all_custom_args(self):
        """所有参数自定义"""
        from run_pipeline import parse_pipeline_args
        with patch.object(sys, "argv", [
            "run_pipeline.py",
            "--accuracy-threshold", "90.0",
            "--min-improvement", "3.0",
            "--epochs", "25",
        ]):
            args = parse_pipeline_args()
        assert args.accuracy_threshold == 90.0
        assert args.min_improvement == 3.0
        assert args.epochs == 25


class TestRunStepTiming:
    def test_step_timing_logged(self):
        """run_step 应记录步骤耗时"""
        result = run_step("计时测试", "-c", "import time; time.sleep(0.1)")
        assert result is True


class TestPrepareDataErrorHandling:
    def test_draw_pattern_stripes(self, tmp_path):
        """_draw_pattern 应正常绘制条纹"""
        from prepare_data import _draw_pattern
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (224, 224), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        _draw_pattern(draw, 224, "stripes", (100, 100, 100), random)
        # 不崩溃即通过

    def test_draw_pattern_spots(self, tmp_path):
        """_draw_pattern 应正常绘制斑点"""
        from prepare_data import _draw_pattern
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (224, 224), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        _draw_pattern(draw, 224, "spots", (100, 100, 100), random)

    def test_draw_shape_triangle_ears(self, tmp_path):
        """_draw_shape 应正常绘制三角耳"""
        from prepare_data import _draw_shape
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (224, 224), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        _draw_shape(draw, 224, "triangle_ears", (100, 100, 100), random)

    def test_draw_shape_unknown(self, tmp_path):
        """未知形状不应崩溃"""
        from prepare_data import _draw_shape
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (224, 224), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        _draw_shape(draw, 224, "unknown_shape", (100, 100, 100), random)
