# logger.py
import logging
import os
from datetime import datetime

def setup_logger(name='autopilot_service', log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Файл
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
