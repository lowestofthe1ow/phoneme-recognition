# Phoneme extraction research project

Main Git repository for phoneme extraction undergraduate research project.

## Project structure

The repository is structured as follows. Directories in parentheses are ignored
by the .gitignore file, so make them yourself.

```
phoneme-extraction
├── bash                      Bash scripts for certain tasks
├── config                    YAML configuration files
├── (data)                    Dataset files
│   ├── nexdata
│   │   └── filipino_822
├── (models)                  Model checkpoint files (.nemo, .ckpt, etc.)
│   ├── checkpoints
│   ├── facebook
│   ├── nvidia
│   └── tokenizers
└── src
    ├── datasets              Python code for managing datasets
    ├── scripts               Python scripts for certain tasks
    │   └── nemo              Python scripts from NVIDIA NeMo repository
    └── utils                 Miscellaneous utility scripts
```

## How to run

> [!NOTE]
> **Last updated**: 27 Jan 2026. Current approach is to fine-tune NVIDIA's
> `stt_en_conformer_ctc_small` model with a phoneme-based
> [SentencePiece](https://github.com/google/sentencepiece) tokenizer.

1. Clone the repository and setup a virtual environment with `uv`. **Currently
   uses Python 3.13**.
2. Install dependencies with `uv sync`

> [!NOTE]
Running Epitran for English G2P requires `flite` and its `lex_lookup`. Follow
installation instructions [here](https://pypi.org/project/epitran/).

3. Set up datasets... (WIP)
4. Train the SentencePiece tokenizer with `source bash/train_tokenizer.sh`.
5. Fine-tune the `stt_en_conformer_ctc_small` model with
   `source bash/finetune_stt_en_conformer.sh`.

> [!NOTE]
> **Regarding `parakeet-tdt-0.6b-v3`**: Currently investigating how to handle
> this model. Current experiments use `stt_en_conformer_ctc_small`, which
> shows rather promising results despite being orders of magnitude smaller.

> [!WARNING]
> Dataset is currently missing a lot of data. Turns out the HuggingFace dataset
> doesn't have all 822 hours... leaving the train split with about 1 hour of
> data only. The `stt_en_conformer_ctc_small` finetune seems to perform rather
> well (?) all things considered.

## Experimental results

Some interesting results from initial experiments...

### `stt_en_conformer_ctc_small`

Training configurations are as in the YAML config and batch scripts. We trained
for 100 epochs.

**Actual sample transcriptions**
- ʔaŋ talumpati aj isaŋ uɾi ŋ kompetiʃon ŋ mɡa paɡbasa.
- pandikit na tape aj inilalaɡaj sa manila papeɾ upaŋ mabasa ŋ lahat aŋ
  nakasulat.
- siɾa aŋ numeɾo ŋ plaka kaja hindi makapaɡ maneho dahil sa kontɾol ŋ tɾapiko sa
  ɾuɾok na oɾas.
- actual d͡ʒunioɾ hiɡh st͡ʃool aj aŋ paɡ-aaɾal ŋ mɡa baɡaj na dapat nilaŋ
  matutunan sa senioɾ hiɡh st͡ʃool.

**Predicted transcriptions**
- ʔaŋ talumpataj isaŋ uɾi ŋ kompitʃon ŋ mɡa paɡbasa.
- padikit na tejpaj nila laɡaj sa manilapeipeɾ upaŋ mabasa ŋ lahat aŋ nakasulat.
- siɾa aŋ numeɾo ŋ plaka kajahindi makapaɡ maneho dahil sa kontɾol ŋ tɾapiko sa
  ɾuɾok na oɾas.
- ʔaŋ dinjoɾa hajs kol aj aŋ paɡ-aaɾal ŋ mɡa baɡa a dapat nal aŋ mututunan sa
  siɲo hajs kol.
