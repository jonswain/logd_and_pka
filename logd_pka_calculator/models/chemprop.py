"""Functions for training models"""

import numpy as np
import chemprop
from logd_pka_calculator.data.make_dataset import chemprop_singletask_data
from logd_pka_calculator.utils.paths import make_dir_function as mdf


def single_task_model_train() -> None:
    """Trains a single-task chemprop model for logD.
    Includes chemprop RDKit 2D descriptors.
    Saves the model in models/
    """
    chemprop_singletask_data()

    arguments = [
        "--data_path",
        str(mdf("data/processed/train_data_chemprop.csv")()),
        "--dataset_type",
        "regression",
        "--save_dir",
        str(mdf("models/chemprop_singletask")()),
        "--features_generator",
        "rdkit_2d_normalized",
        "--no_features_scaling",
        "--split_type",
        "scaffold_balanced",
        "--num_folds",
        "5",
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    _, _ = chemprop.train.cross_validate(
        args=args, train_func=chemprop.train.run_training
    )


def single_task_model_predict(smiles: list) -> np.array:
    """Uses the chemprop model to make predictions on test data.

    Args:
        smiles (list): A list of SMILES string to predict

    Returns:
        np.array: An array of predicted properties
    """
    smiles = [[s] for s in smiles]
    arguments = [
        "--test_path",
        "/dev/null",
        "--preds_path",
        "/dev/null",
        "--checkpoint_dir",
        str(mdf("models/chemprop_singletask")()),
        "--features_generator",
        "rdkit_2d_normalized",
        "--no_features_scaling",
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args, smiles=smiles)
    return np.array(preds)
