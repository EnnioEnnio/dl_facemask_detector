from torch.cuda import is_available, get_device_name, device_count
import argparse
import configparser
import gdown
import logging as log
import os
import shutil
import sys
import tempfile
import zipfile

log.basicConfig(level=log.INFO, format="[%(levelname)s] [%(module)s] %(message)s")


def get_device():
    """
    Returns the device to be used for computation during training / evaluation.
    """
    device = "cuda" if is_available() else "cpu"
    if device == "cuda":
        log.info(f"Using computation device: {get_device_name()} * {device_count()}")
    else:
        log.info("Using computation device: cpu")
    return device


class Config:
    """
    Wrapper class for configparser.ConfigParser.
    """

    def __init__(self, path="./config.ini"):
        self._config_parser = configparser.ConfigParser()
        self._config_parser.read(path)

    def get(self, *args):
        return self._config_parser.get(*args, fallback=None)


def download_model(url, out_path=None):
    """
    Downloads a pretrained model from the specified url to the specified path.
    """
    out = out_path or os.getenv("OUT") or "./model.pt"
    out = os.path.abspath(out)
    if os.path.exists(out):
        log.warn("model already exists, overwriting")
    log.info(f"downloading model to {out}")
    gdown.download(url, out, quiet=False)


def download_dataset(url, out_path=None):
    """
    Downloads a dataset from the specified url to the specified path.
    """
    log.info("creating temporary directory for dataset")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = os.path.join(tmpdir, "dataset.zip")
        log.info(f"downloading dataset to {tmpdir}")
        gdown.download(url, tmp_out, quiet=False)
        log.info("unzipping dataset")
        with zipfile.ZipFile(tmp_out, "r") as zip_ref:
            out = os.path.abspath(out_path or os.getenv("OUT") or "./dataset")
            if os.path.exists(out):
                log.warn(f"path {out} already exists, overwriting")
                shutil.rmtree(out, ignore_errors=True)
            log.info(f"extracting dataset to {out}")
            zip_ref.extractall(out)
        log.info("removing temporary directory")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    # root parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="url to download from")
    parser.add_argument(
        "--out",
        type=str,
        help="path to download to (default: $OUT)",
        default=os.environ.get("OUT"),
    )

    subparsers = parser.add_subparsers(help="commands", dest="command")

    parser_model = subparsers.add_parser("model")
    parser_dataset = subparsers.add_parser("dataset")

    args = parser.parse_args()
    command = args.command

    if command == "model":
        # last updated: 2021-09-15 (sk)
        model_url = "https://drive.google.com/uc?id=1i_GS0o_2evh_K8Iivt0S-i089sE7P_ud"
        download_model(args.url or model_url, args.out)
    elif command == "dataset":
        # last updated: 2021-09-15 (sk)
        dataset_url = "https://drive.google.com/uc?id=1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp"
        download_dataset(args.url or dataset_url, args.out)
    else:
        parser.print_usage()

    sys.exit(0)
