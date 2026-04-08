"""
retrain.py 测试用例
覆盖：
- load_baseline_accuracy: 基线准确率加载
- TUNING_STRATEGIES: 策略配置完整性
- retrain_with_strategy: 策略执行
- main 中 >= 85% 提前终止逻辑
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

import pytest

from retrain import (
    load_baseline_accuracy,
    retrain_with_strategy,
    TUNING_STRATEGIES,
)
from train import get_default_config


# ── load_baseline_accuracy ─────────────────────────────

class TestLoadBaselineAccuracy:
    def test_from_test_metrics(self, tmp_path):
        """从 test_metrics.json 加载"""
        metrics = {"accuracy": 0.82}
        with open(tmp_path / "test_metrics.json", "w") as f:
            json.dump(metrics, f)

        acc = load_baseline_accuracy(str(tmp_path))
        assert acc == pytest.approx(82.0)

    def test_from_history(self, tmp_path):
        """test_metrics 不存在时从 history.json 加载"""
        history = {"val_acc": [50.0, 60.0, 75.0, 70.0]}
        with open(tmp_path / "history.json", "w") as f:
            json.dump(history, f)

        acc = load_baseline_accuracy(str(tmp_path))
        assert acc == pytest.approx(75.0)

    def test_returns_zero_when_no_files(self, tmp_path):
        """无文件时返回 0"""
        acc = load_baseline_accuracy(str(tmp_path))
        assert acc == 0.0

    def test_prefers_test_metrics_over_history(self, tmp_path):
        """test_metrics.json 优先于 history.json"""
        with open(tmp_path / "test_metrics.json", "w") as f:
            json.dump({"accuracy": 0.90}, f)
        with open(tmp_path / "history.json", "w") as f:
            json.dump({"val_acc": [50.0]}, f)

        acc = load_baseline_accuracy(str(tmp_path))
        assert acc == pytest.approx(90.0)


# ── TUNING_STRATEGIES ──────────────────────────────────

class TestTuningStrategies:
    def test_at_least_one_strategy(self):
        assert len(TUNING_STRATEGIES) >= 1

    def test_strategies_have_required_fields(self):
        for strategy in TUNING_STRATEGIES:
            assert "name" in strategy
            assert "description" in strategy
            assert "changes" in strategy
            assert isinstance(strategy["changes"], dict)

    def test_strategies_have_tag(self):
        """每个策略应有唯一的 tag"""
        tags = [s["changes"].get("tag") for s in TUNING_STRATEGIES]
        assert all(t is not None for t in tags)
        assert len(set(tags)) == len(tags)  # 唯一

    def test_strategies_modify_hyperparams(self):
        """每个策略应至少修改一个超参数"""
        for strategy in TUNING_STRATEGIES:
            changes = strategy["changes"]
            non_tag = {k: v for k, v in changes.items() if k != "tag"}
            assert len(non_tag) > 0

    def test_strategies_compatible_with_default_config(self):
        """策略的 key 应该是默认配置中存在的"""
        default = get_default_config()
        valid_keys = set(default.keys()) | {"tag"}
        for strategy in TUNING_STRATEGIES:
            for key in strategy["changes"]:
                assert key in valid_keys, (
                    f"策略 {strategy['name']} 包含无效配置项: {key}"
                )


# ── 达标提前终止逻辑 ──────────────────────────────────

class TestEarlyTermination:
    def test_skips_tuning_when_above_85(self, tmp_path):
        """baseline >= 85% 时 main() 应直接返回，不调用 retrain_with_strategy"""
        # 准备 baseline 目录
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "test_metrics.json", "w") as f:
            json.dump({"accuracy": 0.90}, f)  # 90% >= 85%
        with open(baseline_dir / "config.json", "w") as f:
            json.dump(get_default_config(), f)

        with patch("retrain.retrain_with_strategy") as mock_retrain:
            test_args = [
                "retrain.py",
                "--baseline-dir", str(baseline_dir),
            ]
            with patch.object(sys, "argv", test_args):
                from retrain import main
                main()

            # retrain_with_strategy 不应被调用
            mock_retrain.assert_not_called()

    def test_proceeds_when_below_85(self, tmp_path):
        """baseline < 85% 时应进入调优循环"""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "test_metrics.json", "w") as f:
            json.dump({"accuracy": 0.70}, f)  # 70% < 85%
        with open(baseline_dir / "config.json", "w") as f:
            json.dump(get_default_config(), f)

        mock_return = (75.0, 5.0, True, tmp_path / "tuned")
        (tmp_path / "tuned").mkdir()

        with patch("retrain.retrain_with_strategy", return_value=mock_return) as mock_retrain:
            test_args = [
                "retrain.py",
                "--baseline-dir", str(baseline_dir),
                "--max-attempts", "1",
            ]
            with patch.object(sys, "argv", test_args):
                from retrain import main
                main()

            # retrain_with_strategy 应被调用
            assert mock_retrain.call_count >= 1

    def test_boundary_85_exactly(self, tmp_path):
        """baseline 恰好 85% 时应直接返回"""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "test_metrics.json", "w") as f:
            json.dump({"accuracy": 0.85}, f)  # 恰好 85%
        with open(baseline_dir / "config.json", "w") as f:
            json.dump(get_default_config(), f)

        with patch("retrain.retrain_with_strategy") as mock_retrain:
            test_args = [
                "retrain.py",
                "--baseline-dir", str(baseline_dir),
            ]
            with patch.object(sys, "argv", test_args):
                from retrain import main
                main()

            mock_retrain.assert_not_called()

    def test_exits_nonzero_when_all_strategies_fail(self, tmp_path):
        """所有策略均未达到 2% 提升目标时应以 sys.exit(1) 退出"""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "test_metrics.json", "w") as f:
            json.dump({"accuracy": 0.60}, f)  # 60% < 85%
        with open(baseline_dir / "config.json", "w") as f:
            json.dump(get_default_config(), f)

        # 模拟策略执行：提升 1%（未达到 2% 目标）
        mock_return = (61.0, 1.0, False, tmp_path / "tuned")
        (tmp_path / "tuned").mkdir()

        with patch("retrain.retrain_with_strategy", return_value=mock_return):
            test_args = [
                "retrain.py",
                "--baseline-dir", str(baseline_dir),
                "--max-attempts", "1",
            ]
            with patch.object(sys, "argv", test_args):
                from retrain import main
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_succeeds_when_improvement_met_but_below_85(self, tmp_path):
        """提升 >= 2% 但最终准确率仍 < 85% 时，应视为调优成功（不报错退出）"""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "test_metrics.json", "w") as f:
            json.dump({"accuracy": 0.60}, f)  # 60%
        with open(baseline_dir / "config.json", "w") as f:
            json.dump(get_default_config(), f)

        # 模拟策略执行：提升 5%（达到 2% 目标），但最终 65% < 85%
        mock_return = (65.0, 5.0, True, tmp_path / "tuned")
        (tmp_path / "tuned").mkdir()

        with patch("retrain.retrain_with_strategy", return_value=mock_return) as mock_retrain:
            test_args = [
                "retrain.py",
                "--baseline-dir", str(baseline_dir),
                "--max-attempts", "1",
            ]
            with patch.object(sys, "argv", test_args):
                from retrain import main
                # 不应抛出 SystemExit
                main()
            assert mock_retrain.call_count >= 1

    def test_exits_with_error_when_all_strategies_fail(self, tmp_path):
        """所有策略均未达到目标提升幅度时应 sys.exit(1)"""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "test_metrics.json", "w") as f:
            json.dump({"accuracy": 0.70}, f)  # 70% < 85%
        with open(baseline_dir / "config.json", "w") as f:
            json.dump(get_default_config(), f)

        # 模拟所有策略都只提升 1%（未达到 2% 目标）
        mock_return = (71.0, 1.0, False, tmp_path / "tuned")
        (tmp_path / "tuned").mkdir()

        with patch("retrain.retrain_with_strategy", return_value=mock_return):
            test_args = [
                "retrain.py",
                "--baseline-dir", str(baseline_dir),
                "--max-attempts", "1",
            ]
            with patch.object(sys, "argv", test_args):
                from retrain import main
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


# ── retrain_with_strategy (集成测试) ───────────────────

class TestRetrainWithStrategy:
    def test_strategy_execution(self, tmp_dataset_with_test):
        """测试单个策略执行（快速版本）"""
        config = get_default_config()
        config["dataset_root"] = str(tmp_dataset_with_test)
        config["output_dir"] = str(tmp_dataset_with_test / "outputs")
        config["num_epochs"] = 1
        config["batch_size"] = 4
        config["num_workers"] = 0
        config["pretrained"] = False

        strategy = {
            "name": "test_strategy",
            "description": "测试策略",
            "changes": {
                "learning_rate": 0.0005,
                "num_epochs": 1,
                "tag": "test_tune",
            },
        }

        test_acc, improvement, success, output_dir = retrain_with_strategy(
            strategy=strategy,
            baseline_config=config,
            baseline_acc=0.0,
            target_improvement=0.0,
        )

        assert isinstance(test_acc, float)
        assert isinstance(improvement, float)
        assert isinstance(success, bool)
        assert output_dir.exists()


# ── 错误处理与边界条件测试 ─────────────────────────────

class TestLoadBaselineAccuracyErrorHandling:
    def test_corrupted_test_metrics_json(self, tmp_path):
        """损坏的 test_metrics.json 应回退到 history.json"""
        with open(tmp_path / "test_metrics.json", "w") as f:
            f.write("this is not valid json")
        with open(tmp_path / "history.json", "w") as f:
            json.dump({"val_acc": [50.0, 60.0]}, f)

        acc = load_baseline_accuracy(str(tmp_path))
        assert acc == pytest.approx(60.0)

    def test_corrupted_both_files(self, tmp_path):
        """两个文件都损坏时应返回 0"""
        with open(tmp_path / "test_metrics.json", "w") as f:
            f.write("bad json")
        with open(tmp_path / "history.json", "w") as f:
            f.write("also bad json")

        acc = load_baseline_accuracy(str(tmp_path))
        assert acc == 0.0

    def test_missing_accuracy_key(self, tmp_path):
        """test_metrics.json 缺少 accuracy 字段时应回退"""
        with open(tmp_path / "test_metrics.json", "w") as f:
            json.dump({"precision": 0.9}, f)
        with open(tmp_path / "history.json", "w") as f:
            json.dump({"val_acc": [70.0]}, f)

        acc = load_baseline_accuracy(str(tmp_path))
        assert acc == pytest.approx(70.0)


class TestRetrainMaxAttempts:
    def test_max_attempts_limits_strategies(self, tmp_path):
        """max_attempts 应限制尝试的策略数量"""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "test_metrics.json", "w") as f:
            json.dump({"accuracy": 0.50}, f)
        with open(baseline_dir / "config.json", "w") as f:
            json.dump(get_default_config(), f)

        call_count = 0
        def mock_retrain(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return (51.0, 1.0, False, tmp_path / "tuned")

        (tmp_path / "tuned").mkdir()

        with patch("retrain.retrain_with_strategy", side_effect=mock_retrain):
            test_args = [
                "retrain.py",
                "--baseline-dir", str(baseline_dir),
                "--max-attempts", "2",
            ]
            with patch.object(sys, "argv", test_args):
                from retrain import main
                with pytest.raises(SystemExit):
                    main()

        assert call_count == 2
