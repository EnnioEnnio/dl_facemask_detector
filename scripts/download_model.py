#!/usr/bin/env python3

import gdown
from util import log
import os
import sys


def download_model(url, out_path=None):
    out = out_path or os.getenv("OUT") or "./model.pt"
    out = os.path.abspath(out)
    if os.path.exists(out):
        log.warn("model already exists, overwriting")
    log.info(f"downloading model to {out}")
    gdown.download(url, out, quiet=False)


if __name__ == "__main__":
    # last updated: 2021-09-15 (sk)
    url = "https://drive.google.com/uc?id=1i_GS0o_2evh_K8Iivt0S-i089sE7P_ud"
    download_model(url)
    sys.exit(0)
