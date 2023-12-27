"""Functions to make datasets for LogD calculator"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.model_selection import GroupShuffleSplit
from logd_pka_calculator.utils.paths import make_dir_function as mdf


def load_logd_data(file_path) -> pd.DataFrame:
    """Loads a sdf and extracts logD data

    Args:
        file_path (Path): Path to sdf file

    Returns:
        pd.DataFrame:
    """
    df = PandasTools.LoadSDF(file_path, molColName="ROMol").set_index("ID")
    df["SMILES"] = df.ROMol.apply(Chem.MolToSmiles)
    PandasTools.AddMurckoToFrame(df)
    print(df.columns)
    df = df[["ROMol", "SMILES", "Murcko_SMILES", "logd", "apka", "bpka"]]
    df = df.drop_duplicates(subset="SMILES", keep="first")
    df.logd = df.logd.astype(float)
    return df


def train_test_split(data, train_frac=0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into training and test fractions based on scaffolds

    Args:
        data (pd.DataFrame): A dataframe containing a column of Murko Scaffolds
        and the data to be split
        train_frac (float, optional): Training set fraction of total. Defaults to 0.9.

    Returns:
        Tuple[pd.DataFrame]: The data split into training and test splits
    """
    gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=0)
    train_index, test_index = list(gss.split(data, groups=data.Murcko_SMILES))[0]
    train_data = data.iloc[train_index].reset_index(drop=True)
    test_data = data.iloc[test_index].reset_index(drop=True)
    return (train_data, test_data)


def chemprop_singletask_data() -> None:
    """Puts data into correct shape to be used by chemprop"""
    for dataset in ["train", "test"]:
        df = pd.read_csv(mdf(f"data/processed/{dataset}_data.csv")())
        df[["SMILES", "logd"]].to_csv(
            mdf(f"data/processed/{dataset}_data_chemprop.csv")(), index=False
        )


def chemprop_multitask_data() -> None:
    """Puts data into correct shape to be used by chemprop"""
    for dataset in ["train", "test"]:
        df = pd.read_csv(mdf(f"data/processed/{dataset}_data.csv")())
        df[["SMILES", "logd", "apka", "bpka"]].to_csv(
            mdf(f"data/processed/{dataset}_data_chemprop_multitask.csv")(),
            index=False,
        )
