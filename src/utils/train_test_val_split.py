from pathlib import Path
from sklearn.model_selection import train_test_split

RANDOM_STATE = 339


def train_test_val_split(data, dir):
    # Initial split: "train"/test (90% vs 10%)
    all_train_df, test_df = train_test_split(
        data, test_size=0.1, random_state=RANDOM_STATE, shuffle=True
    )

    print(f"Total train data duration: {all_train_df['duration'].sum()}")

    # Second split: train/valid (80% / 10% (of the whole dataset))
    train_df, val_df = train_test_split(
        all_train_df,
        test_size=0.1 / 0.9,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print(f"Original data shape: {data.shape}")
    print(f"Train split shape: {train_df.shape}")
    print(f"Validation split shape: {val_df.shape}")
    print(f"Test split shape: {test_df.shape}")

    train_df.to_json(
        Path(dir) / "train_manifest.json",
        orient="records",
        lines=True,
        default_handler=str,
    )

    val_df.to_json(
        Path(dir) / "valid_manifest.json",
        orient="records",
        lines=True,
        default_handler=str,
    )

    test_df.to_json(
        Path(dir) / "test_manifest.json",
        orient="records",
        lines=True,
        default_handler=str,
    )
