import json
import shutil
from typing import Union
from pathlib import Path, PosixPath
import requests
import logging


LOG_FORMAT = logging.Formatter(
    '%(name)s: %(asctime)s,%(msecs)d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s')

logger = logging.getLogger()


class SlackLoggingHandler(logging.StreamHandler):
    def __init__(self, webhook_url, stream=None):
        super(SlackLoggingHandler, self).__init__(stream)
        self.url = webhook_url

    def emit(self, record):
        message = super(SlackLoggingHandler, self).format(record)
        requests.post(self.url, json.dumps({'text': message}))


def configure_logger(logger_name: str,
                     log_format: str = LOG_FORMAT,
                     log_dir: Union[str, Path, PosixPath, None] = None,
                     webhook_url: Union[str, None] = None):
    # get root logger
    logger = logging.getLogger()
    logger.name = logger_name

    # slack post
    if webhook_url is not None:
        slack_handler = SlackLoggingHandler(webhook_url)
        slack_handler.setLevel(logging.ERROR)
        slack_handler.setFormatter(log_format)
        logger.addHandler(slack_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True)
        log_filename = str(log_dir.joinpath('text_log.txt'))
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    # stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)
