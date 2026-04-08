"""
统一日志配置模块
使用 RotatingFileHandler 进行日志轮转管理，防止单个日志文件过大。
"""

import logging
from logging.handlers import RotatingFileHandler

# 日志轮转配置
MAX_LOG_BYTES = 10 * 1024 * 1024  # 单个日志文件最大 10MB
BACKUP_COUNT = 3                   # 保留 3 个历史日志文件
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    创建带轮转功能的 logger。

    Args:
        name: logger 名称
        log_file: 日志文件路径
        level: 日志级别

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（模块被多次导入时）
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 文件 handler：带轮转
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
