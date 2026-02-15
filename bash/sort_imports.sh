# Uses isort to sort imports across all Python files in the repo, except those
# from NVIDIA NeMo.
#
# NOTE: The isort package is not a requirement, so install it first

isort --skip src/scripts/nemo --skip .venv . --profile black 
