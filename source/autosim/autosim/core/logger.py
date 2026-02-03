import logging


class AutoSimLogger:
    """Logger class for AutoSim pipeline debugging and monitoring."""

    def __init__(self, name: str, level: int = logging.INFO):
        self._name = name
        self._level = level
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(self._name)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(self._level)
            self._logger.propagate = False
        return self._logger

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
