#!/usr/bin/env python3

import gdown
from util import log
import os
import sys
import tempfile
import zipfile
import shutil


def download_dataset(url, out_path=None):
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
    # last updated: 2021-09-15 (sk)
    url = "https://drive.google.com/uc?id=1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp"
    download_dataset(url)
    sys.exit(0)
