# modules/logging_utils.py

import sys
import logging
from pathlib import Path
from datetime import datetime

def get_pipeline_logger(
    name: str,
    log_file: str = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Create and configure a named logger for pipeline use.

    Creates a logger with:
    - A console handler (writes to stdout with coloured level prefix)
    - An optional file handler (writes to a .log file)
    - ISO 8601 timestamps on every message

    Parameters
    ----------
    name : str
        Logger name. Use the module name (e.g. 'data_io', 'pipeline').
    log_file : str, optional
        Path to a log file. If None, only console logging is active.
    level : str, optional
        Minimum log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
        Default is 'INFO'.

    Returns
    -------
    logging.Logger
        Configured logger ready for use.

    Examples
    --------
    >>> logger = get_pipeline_logger("data_io", log_file="logs/data_io.log")
    >>> logger.info("Dataset loaded: 200 rows")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger
    logger.propagate = False

    return logger