"""Config file for logD and pKa calculators"""

from logd_pka_calculator.utils.paths import make_dir_function as mdf

RAW_DATA_PATH = "data/raw/Random100K.sdf"
DATA_PATH = str(mdf(RAW_DATA_PATH)())
