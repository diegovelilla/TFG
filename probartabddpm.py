import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lib

def make_dataset_from_df(
    df: pd.DataFrame,
    cat_features: list[str],
    target_col: str,
    T: lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_state: int = 42
):
    """
    Build a lib.Dataset from a single DataFrame (instead of loading .npy from disk).

    Arguments:
        df            : pd.DataFrame containing all features + target column.
        cat_features  : list of column-names in df that are categorical.
        target_col    : name of the column in df that holds y.
        T             : a lib.Transformations object (same as before).
        num_classes   : int; if > 0, we're doing classification, else regression.
        is_y_cond     : bool; whether the model is y-conditional (affects how y is concatenated).
        change_val    : bool; if True, call lib.change_val(D) before transforming.
        split_ratios  : 3-tuple of floats summing to 1.0, e.g. (0.8,0.1,0.1) for train/val/test.
        random_state  : int seed for reproducibility.

    Returns:
        A lib.Dataset object, after applying transforms T.
    """
    # 1) Split df into train / val / test
    #    We do one pass for train vs. temp (val+test), then split temp into val/test.
    train_frac, val_frac, test_frac = split_ratios
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("split_ratios must sum to 1.0")

    # First split: train vs. temp
    df_train, df_temp = train_test_split(
        df, test_size=(1.0 - train_frac), random_state=random_state, shuffle=True
    )
    # Compute val / test relative to temp
    relative_val = val_frac / (val_frac + test_frac)
    df_val, df_test = train_test_split(
        df_temp, test_size=(1.0 - relative_val), random_state=random_state, shuffle=True
    )

    # Create a dict-of-DataFrames so we can loop over ['train','val','test']
    splits = {
        "train": df_train.reset_index(drop=True),
        "val":   df_val.reset_index(drop=True),
        "test":  df_test.reset_index(drop=True),
    }

    # 2) Pre-allocate X_num, X_cat, y dictionaries
    #    Mimic the original code's logic about “only create X_cat if it exists on disk or not is_y_cond,” etc.
    #    Here we just allocate if there is at least one categorical column or (num_classes>0 and not is_y_cond).
    X_cat = {} if len(cat_features) > 0 else None
    # For numeric: any column that is not in cat_features or target_col
    # → But if is_y_cond == True, we might defer concatenating y into X.
    all_features = [c for c in df.columns if c != target_col]
    num_features = [c for c in all_features if c not in cat_features]
    X_num = {} if len(num_features) > 0 else None

    y_dict = {}

    # 3) Fill in X_num, X_cat, y for each split
    for split_name, df_split in splits.items():
        # Extract y
        y_split = df_split[target_col].values
        y_dict[split_name] = y_split

        # Build X_cat_split (as a NumPy array or DataFrame – here we keep it as DataFrame, 
        # because lib.transform_dataset likely handles pandas vs numpy)
        if X_cat is not None:
            X_cat_split = df_split[cat_features].copy()
            if not is_y_cond and num_classes > 0:
                # For classification + not y_cond: concat y back into X_cat
                X_cat_split = pd.concat(
                    [X_cat_split.reset_index(drop=True),
                     pd.Series(y_split, name=target_col).reset_index(drop=True)],
                    axis=1
                )
            X_cat[split_name] = X_cat_split

        # Build X_num_split
        if X_num is not None:
            X_num_split = df_split[num_features].copy()
            if not is_y_cond and num_classes == 0:
                # For regression + not y_cond: concat y into X_num
                X_num_split = pd.concat(
                    [X_num_split.reset_index(drop=True),
                     pd.Series(y_split, name=target_col).reset_index(drop=True)],
                    axis=1
                )
            X_num[split_name] = X_num_split

    # 4) Load info (since we no longer have info.json, we can build it on the fly)
    #    If you have some metadata (e.g. n_classes), you can pass it directly here.
    #    Otherwise, mimic the old behavior:
    #    - task_type: "classification" if num_classes>0 else "regression"
    #    - n_classes: num_classes or None
    task_type_str = "classification" if num_classes > 0 else "regression"
    y_info = {}  # (you can populate with anything, but original used {})

    D = lib.Dataset(
        X_num=X_num,
        X_cat=X_cat,
        y=y_dict,
        y_info=y_info,
        task_type=lib.TaskType(task_type_str),
        n_classes=(num_classes if num_classes > 0 else None)
    )

    # 5) Optionally swap val/test within the Dataset
    if change_val:
        D = lib.change_val(D)

    # 6) Finally, apply the transformations
    return lib.transform_dataset(D, T, None)
