import os
import re
from logging import Logger

from graphtool.run_logging import configure_run_logger


def test_configure_run_logger_creates_directory_and_writes_message(tmp_path):
    logs_dir = tmp_path / "logs"

    logger = configure_run_logger(logs_dir)
    try:
        logger.info("Loaded 2 markdown documents")
        _flush_logger(logger)

        log_files = list(logs_dir.glob("graphtool-*.log"))
        assert logs_dir.exists()
        assert len(log_files) == 1
        assert re.fullmatch(
            r"graphtool-\d{8}-\d{6}(?:-\d{3})?\.log",
            log_files[0].name,
        )
        assert "Loaded 2 markdown documents" in log_files[0].read_text()
    finally:
        _close_logger(logger)


def test_configure_run_logger_prunes_oldest_logs(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    old_logs = []
    for index in range(3):
        path = logs_dir / f"graphtool-20260101-00000{index}.log"
        path.write_text(f"log {index}")
        os.utime(path, (index + 1, index + 1))
        old_logs.append(path)

    logger = configure_run_logger(logs_dir, max_log_files=3)
    try:
        log_files = list(logs_dir.glob("graphtool-*.log"))

        assert len(log_files) == 3
        assert old_logs[0] not in log_files
        assert old_logs[1] in log_files
        assert old_logs[2] in log_files
    finally:
        _close_logger(logger)


def _flush_logger(logger: Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def _close_logger(logger: Logger) -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
