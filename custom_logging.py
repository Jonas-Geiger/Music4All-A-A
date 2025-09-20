import logging

def setup_logging():
    """
    Sets up the logging configuration for the data fetching process.

    :return: A tuple of the info_logger and error_logger.
    """
    # Set up the logger for INFO messages
    info_logger = logging.getLogger('info_logger')
    info_logger.setLevel(logging.INFO)

    # Create file handler for logging to a file
    info_file_handler = logging.FileHandler('data_quality.log', encoding='utf-8')
    info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    info_file_handler.setFormatter(info_formatter)
    info_logger.addHandler(info_file_handler)

    # Create stream handler for printing to console
    info_stream_handler = logging.StreamHandler()
    info_stream_handler.setFormatter(info_formatter)
    info_logger.addHandler(info_stream_handler)

    # Set up the logger for ERROR messages
    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler('errors.log', encoding='utf-8')
    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    error_handler.setFormatter(error_formatter)
    error_logger.addHandler(error_handler)

    # Prevent the error_logger from propagating messages up to the root logger
    error_logger.propagate = False

    return info_logger, error_logger
