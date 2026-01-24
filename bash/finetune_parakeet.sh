uv run src/scripts/nemo/speech_to_text_finetune.py \
    --config-path="$(pwd)/config" \
    --config-name="speech_to_text_finetune" \
    model.train_ds.manifest_filepath="data/nexdata/filipino_822/train_manifest.json" \
    model.validation_ds.manifest_filepath="data/nexdata/filipino_822/valid_manifest.json" \
    model.tokenizer.update_tokenizer=True \
    model.tokenizer.dir=models/tokenizers/ipa_tokenizer/tokenizer_spe_char_v120 \
    model.tokenizer.type=bpe \
    trainer.devices=-1 \
    trainer.accelerator='gpu' \
    trainer.max_epochs=50 \
    +init_from_nemo_model="models/nvidia/parakeet-tdt-0.6b-v3.nemo"
