import logging
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "graphtool.run"
LOG_FILE_PATTERN = "graphtool-*.log"


def configure_run_logger(logs_dir: Path, max_log_files: int = 3) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = _new_log_path(logs_dir)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    _prune_old_logs(logs_dir, max_log_files)
    return logger


def _new_log_path(logs_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = logs_dir / f"graphtool-{timestamp}.log"
    if not path.exists():
        return path

    index = 1
    while True:
        path = logs_dir / f"graphtool-{timestamp}-{index:03d}.log"
        if not path.exists():
            return path
        index += 1


def _prune_old_logs(logs_dir: Path, max_log_files: int) -> None:
    log_paths = sorted(
        logs_dir.glob(LOG_FILE_PATTERN),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    for path in log_paths[:-max_log_files]:
        path.unlink()
