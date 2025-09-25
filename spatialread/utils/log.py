import warnings
import os
import inspect
import traceback
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Callable

from colorama import Fore, Style
from pprint import pprint as pp

from spatialread.config import get_log_config


def get_log_dir() -> Path:
    """Get configured log directory.

    Returns:
        Path object to log directory

    Raises:
        RuntimeError: If config not initialized
    """
    config = get_log_config()
    log_dir = Path(config.get("log_dir", "./logs")).absolute()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_logger(filename: str) -> logging.Logger:
    """Create and configure a logger with a file handler."""
    if not filename.endswith(".log"):
        filename += ".log"
    log_dir = get_log_dir()
    logger = logging.getLogger(filename)
    logger.setLevel(get_log_config().get("level", logging.INFO))

    # Avoid adding duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / filename)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


def get_terminal_width(default: int = 50) -> int:
    """Get terminal width with fallback to default value."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


TERMINAL_WIDTH = get_terminal_width()


def title(sentence: str, length: int = TERMINAL_WIDTH, char: str = "=") -> None:
    """Print a centered title with colored formatting."""
    print(
        "\n"
        + Fore.YELLOW
        + (" SpatialRead: " + sentence.upper() + " ").center(length, char)
        + Style.RESET_ALL
    )


def err(log: str) -> None:
    """Print error message with red color."""
    print(Fore.RED + "ERROR: " + log + Style.RESET_ALL)


def warn(log: str) -> None:
    """Print warning message with yellow color."""
    print(Fore.YELLOW + "WARNING: " + log + Style.RESET_ALL)


def end(log: str) -> None:
    """Print end message with blue color."""
    print(Fore.BLUE + "END: " + log + Style.RESET_ALL)


def start(log: str) -> None:
    """Print start message with cyan color."""
    print(Fore.CYAN + "START: " + log + Style.RESET_ALL)


def param(**params: Any) -> None:
    """Pretty print parameters."""
    pp(params)


def log_errors(
    reraise: bool = False,
    include_traceback: bool = True,
    max_traceback_lines: int = 20,
) -> Callable:
    """Decorator to log errors to separate files for each decorated function."""

    def decorator(func: Callable) -> Callable:
        # Create unique logger name based on function location and name
        file_path = inspect.getfile(func)
        file_name = os.path.basename(file_path)
        module_name = os.path.splitext(file_name)[0]
        logger_name = f"{module_name}.{func.__name__}"

        # Create log directory structure
        log_dir = get_log_dir()
        log_subdir = log_dir / module_name
        log_subdir.mkdir(parents=True, exist_ok=True)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log_suffix = kwargs.pop("log_suffix", None)
            log_file = (
                f"{func.__name__}_{log_suffix}.log"
                if log_suffix
                else f"{func.__name__}.log"
            )
            log_path = log_subdir / log_file

            # Get or create logger
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)

            # Add file handler if none exists
            if not logger.handlers:
                handler = logging.FileHandler(log_path)
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                logger.addHandler(handler)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__} from {file_name}: {str(e)}"

                if include_traceback:
                    tb = traceback.format_exc().splitlines()[-max_traceback_lines:]
                    error_msg += "\nTraceback (last {} lines):\n{}".format(
                        max_traceback_lines, "\n".join(tb)
                    )

                logger.error(error_msg)

                if reraise:
                    raise
                return None

        return wrapper

    return decorator


class Log:
    def __init__(self):
        super().__init__()
        # NOTE here we do not initialize logger due to we want dynamically load loogger_config from user config file
        self.logger = None

    def _setup_logger(self) -> None:
        """Configure the logger based on the configuration."""
        self.logger_config = get_log_config()
        if not self.logger_config.get("enabled", False):
            return

        # Create logger
        self.logger = logging.getLogger(self.logger_config["filename"])
        self.logger.setLevel(
            getattr(logging, self.logger_config["level"].upper(), logging.INFO)
        )

        # Avoid duplicate handlers
        if self.logger.handlers:
            return

        # Create formatter
        formatter = logging.Formatter(self.logger_config["format"])

        # Create file handler if specified
        if self.logger_config.get("filename"):
            log_file_name = self.logger_config["filename"]
            if not log_file_name.endswith(".log"):
                log_file_name += ".log"
            log_file = Path(log_file_name)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log(
        self,
        message: str,
        level: Literal["debug", "info", "warning", "error", "critical"] = "info",
    ) -> None:
        """
        Log a message with the specified level.

        Args:
            message: The message to log
            level: One of 'debug', 'info', 'warning', 'error', 'critical'

        Raises:
            ValueError: If invalid log level is provided
        """
        if self.logger is None:
            self._setup_logger()

        log_method = getattr(self.logger, level.lower(), None)
        if log_method is None:
            raise ValueError(
                f"Invalid log level '{level}'. "
                "Expected one of: debug, info, warning, error, critical"
            )

        message = f"{self.__class__.__name__}: {message}"

        log_method(message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(message, "debug")

    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, "info")

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, "warning")
        warnings.warn(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, "error")

    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.log(message, "critical")
        raise RuntimeError(message)
