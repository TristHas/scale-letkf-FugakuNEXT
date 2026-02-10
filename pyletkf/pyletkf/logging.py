import logging, os, time
from pathlib import Path

import logging
import time

class RankLogger(logging.Logger):
    def time(self, msg: str, *, level: int = logging.INFO):
        return _LogTiming(self, msg, level)

class _LogTiming:
    __slots__ = ("logger", "msg", "level", "t0")

    def __init__(self, logger: logging.Logger, msg: str, level: int):
        self.logger = logger
        self.msg = msg
        self.level = level

    def __enter__(self):
        self.logger.log(self.level, "%s...", self.msg)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type is None:
            self.logger.log(self.level, "TIMING %s done in %.3f s", self.msg, dt)
        else:
            self.logger.error("%s failed after %.3f s", self.msg, dt)
        return False

def configure_rank_file_logging(
        rank,
        logger_name: str,
        log_dir = "/tmp/letkf",
        level: int = logging.INFO,
    ) -> logging.Logger:
    """
        Minimal per-rank file logging.
        - One file per rank: <prefix>.rank00012.log
        - No QueueListener/async machinery
        - Minimal format: timestamp level message
    """
    log_dir = Path(log_dir) / logger_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"rank{rank:05d}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
               for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a", delay=True)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
        
    logger.info(f"logging started (rank={rank})")
    return logger

logging.setLoggerClass(RankLogger)
