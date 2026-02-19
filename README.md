<div align="center">

<h1>Phoneme recognition of Filipino speech with deep learning methods</h1>

</div>

This serves as the main Git repository for a phoneme recognition undergraduate
research project.

## Project structure

The repository is structured as follows. Directories in parentheses are ignored
by the .gitignore file, so make them yourself.

```
phoneme-recognition
├── bash                      Bash scripts for certain tasks
├── config                    YAML configuration files
├── (data)                    Dataset files
│   ├── magichub
│   │   └── asr-sfdusc
│   ├── nexdata
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

## Setting up

### Installing dependencies

This project uses [`uv`](https://docs.astral.sh/uv/) to manage packages.

1. Create a Python 3.13 virtual environment with `uv venv --python 3.13`.
2. Run `uv sync` to install dependencies.

### Set up `pre-commit`

1. `pre-commit` should have been installed as a development dependency. Check
   with `pre-commit --version`.
2. Install the hook scripts with `pre-commit install`.
3. Run `pre-commit run --all-files` to run the pre-commit hooks on all files.

### Creating a `.env` file

Create a `.env` file at the root directory to run the scripts in `bash/`. It
should contain the following:

```
CHECKPOINT_PATH="models/checkpoints/Speech_To_Text_Finetuning_2026-01-30_18-24-21.nemo"
BASE_MODEL_PATH="models/nvidia/stt_en_conformer_ctc_small.nemo"
TEST_MANIFEST_PATH="data/nexdata/filipino_822/test_manifest.json"
TRAIN_MANIFEST_PATH="data/nexdata/filipino_822/train_manifest.json"
VALID_MANIFEST_PATH="data/nexdata/filipino_822/valid_manifest.json"
```

Modify the file as needed when changing models or datasets.

### Download datasets

1. **Nexdata "822" hours dataset**: Download the dataset from
   [here](https://huggingface.co/datasets/Nexdata/822-Hours-Tagalog-the-Philippines-Scripted-Monologue-Smartphone-speech-dataset)
2. **MagicHub ASR-SFDuSC**: Download the dataset from
   [here](https://magichub.com/datasets/filipino-scripted-speech-corpus-daily-use-sentence/)
3. Follow the directory structure shown above with the downloaded files.
4. Run the `make_manifest.py` scripts in the respective dataset folders in
   `src/datasets/scripts` for all downloaded datasets

## Model training

### Fine-tuning

> [!NOTE]
> **Last updated**: 27 Jan 2026. Current approach is to fine-tune NVIDIA's
> `stt_en_conformer_ctc_small` model with a phoneme-based
> [SentencePiece](https://github.com/google/sentencepiece) tokenizer.

1. Set up datasets... (WIP)
    4. Train the SentencePiece tokenizer with `source bash/train_tokenizer.sh`.
2. Fine-tune the `stt_en_conformer_ctc_small` model with
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

### Evaluation

Run `source bash/evaluate_phoneme_recognition.sh`

## Experimental results

Some interesting results from initial experiments...

### `stt_en_conformer_ctc_small` trained on Nexdata corpus

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
