# Phoneme extraction research project

Main Git repository for phoneme extraction undergraduate research project.

## Project structure

The repository is structured as follows. Directories in parentheses are ignored.

```
phoneme-extraction
├── bash                      Bash scripts for certain tasks
├── config                    YAML configuration files
├── (data)                    Dataset files
├── (models)                  Model checkpoint files (.nemo, .ckpt, etc.)
└── src
    ├── datasets              Python code for managing datasets
    ├── scripts               Python scripts for certain tasks
    │   └── nemo              Python scripts from NVIDIA NeMo repository
    └── utils                 Miscellaneous utility scripts
```

## How to run

> [!NOTE]
> **Last updated**: 25 Jan 2026. Current approach is to fine-tune NVIDIA's
> `parakeet-tdt-0.6b-v3` model with a phoneme-based
> [SentencePiece](https://github.com/google/sentencepiece) tokenizer.

1. Clone the repository and setup a virtual environment with `uv`. **Currently
   uses Python 3.13**.
2. Install dependencies with `uv sync`
3. Set up datasets... (WIP)
4. Train the SentencePiece tokenizer with `source bash/train_tokenizer.sh`.
5. Fine-tune the `parakeet-tdt-0.6b-v3` model with
   `source bash/finetune_parakeet.sh`.
