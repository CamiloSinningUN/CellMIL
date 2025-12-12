import pandas as pd
from typing import cast
from sklearn.model_selection import train_test_split  # type: ignore
from cellmil.utils import logger


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.8,
    test: bool = False,
    stratify: bool = True,
    label: str = "label",
    random_state: int = 42,
):
    assert train_size <= 1.0 and train_size > 0.0, "train_size must be between 0 and 1"

    val_size = 1 - train_size if not test else (1 - train_size) / 2
    test_size = 1 - train_size - val_size

    df["SPLIT"] = "train"

    if test:
        _df, test_df = cast(
            tuple[pd.DataFrame, pd.DataFrame],
            train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df[label] if stratify and label in df.columns else None,
            ),
        )

        df.loc[test_df.index, "SPLIT"] = "test"
        logger.info(f"Test size: {len(test_df) if test else 0}")
    else:
        _df = df

    train_df, val_df = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            _df,
            test_size=val_size,
            random_state=random_state,
            stratify=_df[label] if stratify and label in _df.columns else None,
        ),
    )

    df.loc[val_df.index, "SPLIT"] = "val"

    logger.info(f"Validation size: {len(val_df)}")
    logger.info(f"Train size: {len(train_df)}")

    return df

def complementary_frequencies(df: pd.DataFrame, label: str = "label") -> dict[int, float]:
    label_counts = df[label].value_counts().sort_index()  # type: ignore
    total_samples = len(df)
    complementary_freqs: dict[int, float] = {}
    for label_val in label_counts.index:
        count = cast(int, label_counts[label_val])
        complementary_freqs[label_val] = (total_samples - count) / total_samples
    return complementary_freqs


