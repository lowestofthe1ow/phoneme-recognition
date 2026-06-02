import subprocess
import sys
from pathlib import Path

import questionary

from src.utils.cli_utils import get_checkpoints, get_manifest_files


def get_manifest_files(base_dir=DEFAULT_DATA_DIR):
    path = Path(base_dir)
    if not path.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        sys.exit(1)

    # Recursively check all subdirectories for .json files
    manifests = [str(p) for p in path.rglob("*.json") if p.is_file()]

    if not manifests:
        print(f"Error: No .json files found in '{base_dir}'.")
        sys.exit(1)

    return sorted(manifests)


available_manifests = get_manifest_files()

# Prompt user to select the test manifest
test_manifest = questionary.select(
    "Select test data manifest:", choices=available_manifests, use_indicator=True
).ask()

if not test_manifest:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select the train manifest
train_manifest = questionary.select(
    "Select train data manifest:", choices=available_manifests, use_indicator=True
).ask()

if not train_manifest:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select the validation manifest
valid_manifest = questionary.select(
    "Select validation data manifest:", choices=available_manifests, use_indicator=True
).ask()

if not valid_manifest:
    print("Cancelled by user.")
    sys.exit(0)

# Build the command execution array for uv
cmd = [
    "uv",
    "run",
    "python3",
    "-m",
    "src.scripts.kotone_train",
    "--test-manifest-path",
    test_manifest,
    "--train-manifest-path",
    train_manifest,
    "--valid-manifest-path",
    valid_manifest,
]

print(f"\nRunning command: {' '.join(cmd)}\n")

subprocess.run(cmd, check=True)
