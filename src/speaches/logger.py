import logging
import logging.config
from collections import deque


class RingBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        self.buffer = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(self.format(record))

    def get_logs(self, lines: int = 200) -> list[str]:
        return list(self.buffer)[-lines:]


ring_buffer_handler = RingBufferHandler()


def setup_logger(log_level: str) -> None:
    assert log_level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], log_level
    ring_buffer_handler.setFormatter(logging.Formatter(
        "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s"
    ))
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s"},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "root": {
                "level": log_level.upper(),
                "handlers": ["stdout"],
            },
            "PIL": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "httpx": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "python_multipart": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "httpcore": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "aiortc": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "aioice": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "numba.core": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "openai": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
        },
    }

    logging.config.dictConfig(logging_config)
    logging.getLogger().addHandler(ring_buffer_handler)
