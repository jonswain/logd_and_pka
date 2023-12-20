"""Functions for generating features for classical ML models."""

import pandas as pd
from molfeat.calc import FPCalculator, get_calculator
from molfeat.trans import MoleculeTransformer


def all_descriptors(dataframe, molcol="ROMol") -> pd.DataFrame:
    """Generates Morgan count fingerprints and all RDKit 2D descriptors.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe with a column of canonical smiles.

    Returns:
        Pandas dataframe: The combined features dataframe.
    """
    params = {
        "radius": 3,
        "nBits": 2048,
        "useChirality": True,
        "useFeatures": True,
    }

    fp_calc = FPCalculator("ecfp-count", **params)
    fp_transf = MoleculeTransformer(fp_calc, n_jobs=-1)

    rdkit_calc = get_calculator("desc2d")
    rdkit_transf = MoleculeTransformer(rdkit_calc, n_jobs=-1)

    df_desc = pd.DataFrame(fp_transf(dataframe[molcol]), columns=fp_calc.columns)
    df_rdkit = pd.DataFrame(rdkit_transf(dataframe[molcol]), columns=rdkit_calc.columns)
    df_rdkit.drop("Alerts", axis=1, inplace=True)
    desc_df = pd.concat([df_desc, df_rdkit], axis=1)
    desc_df.index = dataframe.SMILES
    desc_df = desc_df.fillna(0)

    return desc_df
