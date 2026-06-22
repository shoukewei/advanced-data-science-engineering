import logging
from pathlib import Path
import pytest
from modules.logging_utils import get_pipeline_logger


def test_get_pipeline_logger_writes_file_and_returns_logger(tmp_path):
    log_file = tmp_path / "pipeline.log"
    logger = get_pipeline_logger("tests.logger", log_file=str(log_file), level="INFO")
    logger.info("hello-world")

    # file created and contains message
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "hello-world" in content


def test_get_pipeline_logger_idempotent_handlers(tmp_path):
    log_file = tmp_path / "pipeline2.log"
    logger1 = get_pipeline_logger("tests.logger2", log_file=str(log_file))
    logger2 = get_pipeline_logger("tests.logger2", log_file=str(log_file))
    # calling twice should not duplicate handlers
    assert len(logger1.handlers) == len(logger2.handlers)
