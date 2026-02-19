source .env

uv run python3 -m src.scripts.augmented_ctc_finetune \
    --config-path="$(pwd)/config" \
    --config-name="speech_to_text_finetune" \
    model.train_ds.manifest_filepath="$TRAIN_MANIFEST_PATH" \
    model.validation_ds.manifest_filepath="$VALID_MANIFEST_PATH" \
    model.tokenizer.update_tokenizer=True \
    model.tokenizer.dir=models/tokenizers/ipa_tokenizer/tokenizer_spe_char_v34 \
    model.tokenizer.type=bpe \
    trainer.devices=-1 \
    trainer.accelerator='gpu' \
    trainer.max_epochs=200 \
    +init_from_nemo_model="$BASE_MODEL_PATH"
