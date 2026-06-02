from pathlib import Path

DEFAULT_CHECKPOINTS_PATH = "models/checkpoints"
DEFAULT_DATA_DIR = "data"


def get_checkpoints(base_dir=DEFAULT_CHECKPOINTS_PATH):
    path = Path(base_dir)
    if not path.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        sys.exit(1)

    # Recursively check all checkpoint subdirectories
    checkpoints = [
        str(p) for p in path.glob("*/*") if p.is_dir() and "checkpoint" in p.name
    ]

    return sorted(checkpoints)


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
