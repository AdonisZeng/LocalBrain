import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    log_file = log_path / f"localbrain_{today}_{time_str}.txt"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.handlers.clear()
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    for name in ["uvicorn", "uvicorn.access", "httpx", "httpcore", "chromadb", "sentence_transformers"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    logger = logging.getLogger("app.core.logging_config")
    logger.info(f"Logging system initialized: log_file={log_file}, level={log_level}")


def get_logger(name: str = None):
    return logging.getLogger(name)
