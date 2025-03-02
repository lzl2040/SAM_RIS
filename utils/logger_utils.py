import logging

def get_logger(name):
    # ---------------- start writing logs ----------------------#
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_hander = logging.FileHandler(name, mode='a+')
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)
    console_hander = logging.StreamHandler()
    console_hander.setFormatter(formatter)
    logger.addHandler(console_hander)
    # ---------------- end writing logs ----------------------#
    return logger