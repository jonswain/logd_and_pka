"""Functions to set up project"""

import os
from logd_pka_calculator.utils.paths import make_dir_function as mdf


def set_up() -> None:
    """Creates required data directories"""
    os.makedirs(mdf("data/external")(), exist_ok=True)
    os.makedirs(mdf("data/interim")(), exist_ok=True)
    os.makedirs(mdf("data/processed")(), exist_ok=True)
    os.makedirs(mdf("models")(), exist_ok=True)
