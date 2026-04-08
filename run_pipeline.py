"""
VGG16 动物图像分类 - 完整流水线
一键运行：数据准备 -> 训练 -> 评估 -> 调优（如需要）
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

from log_config import setup_logger

logger = setup_logger(__name__, "pipeline.log")

ACCURACY_THRESHOLD = 85.0
MIN_IMPROVEMENT = 2.0

REQUIRED_PACKAGES = ["torch", "torchvision", "PIL", "numpy", "matplotlib", "sklearn", "seaborn"]


def parse_pipeline_args():
    """解析流水线命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="VGG16 动物图像分类 - 完整流水线")
    parser.add_argument(
        "--accuracy-threshold", type=float, default=ACCURACY_THRESHOLD,
        help=f"准确率达标阈值 (默认 {ACCURACY_THRESHOLD}%%)",
    )
    parser.add_argument(
        "--min-improvement", type=float, default=MIN_IMPROVEMENT,
        help=f"调优最低提升幅度 (默认 {MIN_IMPROVEMENT}%%)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="基线训练轮数 (默认 15)",
    )
    return parser.parse_args()


def check_environment():
    """
    检查运行环境：Python 版本、虚拟环境、依赖包。
    不强制终止，但给出明确警告。
    """
    import sys

    # Python 版本检查
    py_version = sys.version_info
    if py_version < (3, 9):
        logger.error(
            f"Python 版本过低: {py_version.major}.{py_version.minor}，需要 >= 3.9"
        )
        return False

    logger.info(f"Python 版本: {py_version.major}.{py_version.minor}.{py_version.micro}")

    # 虚拟环境检查（Docker 容器内跳过）
    in_docker = os.environ.get("IN_DOCKER") == "1" or os.path.exists("/.dockerenv")
    in_venv = (
        hasattr(sys, "real_prefix")  # virtualenv
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)  # venv
        or os.environ.get("CONDA_DEFAULT_ENV") is not None  # conda
        or os.environ.get("VIRTUAL_ENV") is not None  # venv env var
    )
    if in_docker:
        logger.info("检测到 Docker 容器环境，跳过虚拟环境检查。")
    elif not in_venv:
        logger.warning(
            "未检测到虚拟环境。强烈建议在 venv/conda 中运行以避免依赖冲突："
        )
        logger.warning("   python3 -m venv venv && source venv/bin/activate")

    # 依赖包检查
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(f"缺少依赖包: {', '.join(missing)}")
        logger.error("请运行: pip install -r requirements.txt")
        return False

    logger.info("依赖检查通过。")
    return True


def find_latest_output_dir(base_dir: str = "outputs", tag: str = "baseline") -> Path:
    """找到最新的输出目录"""
    base = Path(base_dir)
    if not base.exists():
        return None

    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(tag)]
    if not dirs:
        # 尝试找任何目录
        dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        return None

    return max(dirs, key=lambda d: d.stat().st_mtime)


def run_step(description: str, module_name: str, *args):
    """运行一个步骤"""
    import time as _time

    logger.info(f"\n{'=' * 60}")
    logger.info(f"步骤: {description}")
    logger.info(f"{'=' * 60}")

    cmd = [sys.executable, module_name] + list(args)
    logger.info(f"执行: {' '.join(cmd)}")

    step_start = _time.time()
    result = subprocess.run(cmd, capture_output=False)
    step_time = _time.time() - step_start

    if result.returncode != 0:
        logger.error(f"步骤失败: {description} (返回码: {result.returncode}, 耗时: {step_time:.1f}s)")
        return False

    logger.info(f"步骤完成: {description} (耗时: {step_time:.1f}s)")
    return True


def main():
    args = parse_pipeline_args()
    accuracy_threshold = args.accuracy_threshold
    min_improvement = args.min_improvement
    epochs = args.epochs

    logger.info("=" * 60)
    logger.info("VGG16 动物图像分类 - 完整流水线")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"准确率阈值: {accuracy_threshold}%, 最低提升: {min_improvement}%, 训练轮数: {epochs}")
    logger.info("=" * 60)

    # ── Step 0: 环境检查 ──
    if not check_environment():
        logger.error("环境检查未通过，终止流水线。")
        sys.exit(1)

    # ── Step 1: 数据准备（流水线模式下允许合成数据） ──
    if not run_step("数据准备", "prepare_data.py", "--allow-synthetic"):
        logger.error("数据准备失败，终止流水线。")
        sys.exit(1)

    # ── 合成数据警告：检测是否使用了合成数据 ──
    summary_path = Path("dataset") / "dataset_summary.txt"
    data_prepare_log = Path("data_prepare.log")
    if data_prepare_log.exists():
        log_content = data_prepare_log.read_text(encoding="utf-8", errors="ignore")
        if "合成数据集生成完成" in log_content:
            logger.warning("")
            logger.warning("!" * 60)
            logger.warning("⚠️  警告：当前使用的是合成数据集（几何图形），非真实动物图片！")
            logger.warning("⚠️  合成数据仅用于验证流水线功能可运行，不代表真实分类效果。")
            logger.warning("⚠️  如需真实动物分类，请将真实图片放入 dataset/train/ 和 dataset/val/")
            logger.warning("⚠️  各类别子目录后重新运行（不带 --allow-synthetic 参数）。")
            logger.warning("!" * 60)
            logger.warning("")

    # ── 在控制台展示数据集图片数量统计（Prompt 要求：说明每个新文件夹的图片数量） ──
    if summary_path.exists():
        logger.info("")
        logger.info("=" * 60)
        logger.info("各文件夹图片数量统计")
        logger.info("=" * 60)
        with open(summary_path, encoding="utf-8") as f:
            for line in f:
                logger.info(f"  {line.rstrip()}")
        logger.info("=" * 60)
        logger.info("")

    # ── Step 2: 基线训练 ──
    if not run_step(
        f"基线训练 (VGG16, {epochs} epochs)",
        "train.py",
        "--epochs", str(epochs),
        "--tag", "baseline",
    ):
        logger.error("训练失败，终止流水线。")
        sys.exit(1)

    # ── Step 3: 评估 ──
    output_dir = find_latest_output_dir("outputs", "baseline")
    if output_dir is None:
        logger.error("未找到训练输出目录。")
        sys.exit(1)

    model_path = output_dir / "best_model.pth"
    if not model_path.exists():
        model_path = output_dir / "final_model.pth"

    if not run_step(
        "测试集评估",
        "evaluate.py",
        "--model", str(model_path),
        "--test-dir", "dataset/test",
        "--output-dir", str(output_dir),
    ):
        logger.error("评估失败，终止流水线。")
        sys.exit(1)

    # 读取测试准确率
    metrics_path = output_dir / "test_metrics.json"
    if not metrics_path.exists():
        logger.error("未找到评估指标文件。")
        sys.exit(1)

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        test_accuracy = metrics["accuracy"] * 100
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"评估指标文件解析失败: {e}")
        sys.exit(1)

    # ── 在控制台直观展示关键评估指标 ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("测试集评估结果摘要")
    logger.info("=" * 60)
    logger.info(f"  准确率 (Accuracy):    {test_accuracy:.2f}%")
    logger.info(f"  精确率 (Precision):   {metrics['precision_macro']*100:.2f}%")
    logger.info(f"  召回率 (Recall):      {metrics['recall_macro']*100:.2f}%")
    logger.info(f"  F1 分数 (F1-macro):   {metrics['f1_macro']*100:.2f}%")
    logger.info("-" * 60)
    if "per_class" in metrics:
        logger.info(f"  {'类别':<10} {'精确率':>8} {'召回率':>8} {'F1':>8}")
        for cls_name, cls_metrics in metrics["per_class"].items():
            logger.info(
                f"  {cls_name:<10} "
                f"{cls_metrics['precision']*100:>7.2f}% "
                f"{cls_metrics['recall']*100:>7.2f}% "
                f"{cls_metrics['f1']*100:>7.2f}%"
            )
    logger.info("=" * 60)

    # ── 在控制台展示混淆矩阵分析报告 ──
    analysis_path = output_dir / "confusion_analysis.txt"
    if analysis_path.exists():
        logger.info("")
        with open(analysis_path, encoding="utf-8") as f:
            analysis_text = f.read()
        logger.info(analysis_text)
    else:
        logger.warning("未找到混淆矩阵分析报告文件 (confusion_analysis.txt)")

    # ── Step 4: 判断是否需要调优 ──
    if test_accuracy >= accuracy_threshold:
        logger.info(f"✅ 准确率 {test_accuracy:.2f}% >= {accuracy_threshold}%，无需调优！")
    else:
        logger.info(f"⚠️ 准确率 {test_accuracy:.2f}% < {accuracy_threshold}%，开始调优...")

        if not run_step(
            "参数调优重训练",
            "retrain.py",
            "--baseline-dir", str(output_dir),
            "--target-improvement", str(min_improvement),
        ):
            logger.warning("调优过程出现问题，请检查日志。")

    # ── 总结 ──
    logger.info("\n" + "=" * 60)
    logger.info("流水线执行完毕")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)

    # ── 在总结中再次展示数据集图片数量统计（Prompt 要求：说明每个新文件夹的图片数量） ──
    if summary_path.exists():
        logger.info("")
        logger.info("数据集图片数量统计:")
        logger.info("-" * 40)
        with open(summary_path, encoding="utf-8") as f:
            for line in f:
                logger.info(f"  {line.rstrip()}")
        logger.info("-" * 40)

    logger.info("生成的文件:")
    logger.info("  - dataset/dataset_summary.txt : 数据集统计")
    logger.info("  - training_curves.png         : 训练曲线")
    logger.info("  - confusion_matrix.png        : 混淆矩阵")
    logger.info("  - per_class_metrics.png       : 每类指标对比")
    logger.info("  - test_metrics.json           : 评估指标 + 混淆分析")
    logger.info("  - confusion_analysis.txt      : 混淆矩阵分析报告")
    logger.info("  - errors/                     : 错误分类样本（按类别归档）")
    logger.info("  - gradcam/                    : Grad-CAM 热力图（误判原因可视化）")
    logger.info("  - best_model.pth              : 最佳模型")


if __name__ == "__main__":
    main()
