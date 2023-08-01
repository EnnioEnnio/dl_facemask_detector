import configparser


class Config:
    def __init__(self, path="./config.ini"):
        self._config_parser = configparser.ConfigParser()
        self._config_parser.read(path)

    def get(self, *args):
        return self._config_parser.get(*args, fallback=None)
