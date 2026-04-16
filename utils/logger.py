import torch.distributed as dist
import logging
import os
from datetime import datetime


def _is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def create_logger(logging_dir, log_filename=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if _is_main_process():  # real logger
        handlers = [logging.StreamHandler()]
        run_log_path = None
        if logging_dir:
            os.makedirs(logging_dir, exist_ok=True)
            if not log_filename:
                run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"log_{run_stamp}.txt"
            run_log_path = os.path.join(logging_dir, log_filename)
            handlers.append(logging.FileHandler(run_log_path))

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
            force=True,
        )
        logger = logging.getLogger(__name__)
        if run_log_path:
            logger.info(f"Run log file: {run_log_path}")
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger