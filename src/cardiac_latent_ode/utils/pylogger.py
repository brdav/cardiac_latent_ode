import logging
from typing import ClassVar

from rich.console import Console
from rich.logging import RichHandler


class RichLogger:
    """A Rich-enhanced logger that writes to terminal."""

    _loggers: ClassVar[dict] = {}  # keep one logger per module

    def __new__(cls, name: str):
        if name in cls._loggers:
            return cls._loggers[name]

        # Create main logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = (
            False  # prevent duplicate output if root logger is configured
        )

        # ========== Handlers ==========
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            markup=True,
            log_time_format="[%X]",
        )
        rich_handler.setLevel(logging.INFO)

        # ========== Attach Handlers ==========
        logger.addHandler(rich_handler)

        cls._loggers[name] = logger
        return logger