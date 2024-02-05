"""Set up logging for the ISF package."""
import logging, sys

class LastPartFilter(logging.Filter):
    """
    Logging add-on that adds a field that only shows the last part of the module name.
    Useful for logger stream handlers, when the logger name can be quite long and muddles output.
    """
    def filter(self, record):
        record.name_last = record.name.rsplit('.', 1)[-1]
        return True

# All loggers will inherit the root logger's level and handlers
logger = logging.getLogger("ISF")
# Redirect warnings to the logging system. This will format them accordingly.
logging.captureWarnings(True)
logger_stream_handler = logging.StreamHandler(stream=sys.stdout)
logger_stream_handler.name = "ISF_logger_stream_handler"
logger_stream_handler.setFormatter(
    logging.Formatter("[%(levelname)s] %(name_last)s: %(message)s"))
logger_stream_handler.addFilter(LastPartFilter())
logger.handlers = [logger_stream_handler]  
logger.setLevel(logging.INFO)  # Set back to WARNING at the end