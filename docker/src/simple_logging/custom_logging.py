import logging
import os
import errno
import types


def log_newline(self, how_many_lines=1):
    file_handler = None
    if self.handlers:
        file_handler = self.handlers[0]

    # Switch formatter, output a blank line
    file_handler.setFormatter(self.blank_formatter)
    for i in range(how_many_lines):
        self.info('')

    # Switch back
    file_handler.setFormatter(self.default_formatter)


def setup_custom_logger(name, logging_level, flog=None,
                        log_format='%(asctime)s - %(levelname)s - %(process)d - '
                                   '%(module)s.%(funcName)s:%(lineno)d\t%(message)s'):

    if flog is None:
        raise TypeError("setup_custom_logger::Argument flog cannot be None")

    if not os.path.exists(os.path.dirname(flog)):
        try:
            os.makedirs(os.path.dirname(flog))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    formatter = logging.Formatter(fmt=log_format)

    fhandler = logging.FileHandler(flog)
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging_level)

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    logger.addHandler(fhandler)
    logger.default_formatter = formatter
    logger.blank_formatter = logging.Formatter(fmt="")
    logger.newline = types.MethodType(log_newline, logger)
    return logger