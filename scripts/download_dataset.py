#!/usr/bin/env python3

import gdown
import logging as log
import os
import sys
import tempfile
import zipfile
import shutil

if __name__ == "__main__":
    url = "https://drive.google.com/uc?id=1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp"
    log.basicConfig(level=log.INFO, format="[%(levelname)s] %(message)s")
    log.info("creating temporary directory for dataset")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = os.path.join(tmpdir, "dataset.zip")
        gdown.download(url, tmp_out, quiet=False)
        log.info("unzipping dataset")
        with zipfile.ZipFile(tmp_out, "r") as zip_ref:
            zip_ref.extractall(os.getenv("OUT") or ".")
        log.info("removing temporary directory")
        shutil.rmtree(tmpdir, ignore_errors=True)
    sys.exit(0)
