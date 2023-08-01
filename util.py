import logging as log
from torch.cuda import is_available, get_device_name, device_count
import configparser


log.basicConfig(
    level=log.INFO, format="[%(levelname)s] [%(module) - %(funcName)] %(message)s"
)


def get_device():
    device = "cuda" if is_available() else "cpu"
    if device == "cuda":
        log.info(f"Using computation device: {get_device_name()} * {device_count()}")
    else:
        log.info("Using computation device: cpu")
    return device


class Config:
    def __init__(self, path="./config.ini"):
        self._config_parser = configparser.ConfigParser()
        self._config_parser.read(path)

    def get(self, *args):
        return self._config_parser.get(*args, fallback=None)
