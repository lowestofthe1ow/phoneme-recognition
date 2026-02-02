uv run src/scripts/nemo/speech_to_text_finetune.py \
    --config-path="$(pwd)/config" \
    --config-name="speech_to_text_finetune" \
    model.train_ds.manifest_filepath="data/magichub/asr-sfdusc/train_manifest.json" \
    model.validation_ds.manifest_filepath="data/magichub/asr-sfdusc/valid_manifest.json" \
    model.tokenizer.update_tokenizer=True \
    model.tokenizer.dir=models/tokenizers/ipa_tokenizer/tokenizer_spe_char_v120 \
    model.tokenizer.type=bpe \
    trainer.devices=-1 \
    trainer.accelerator='gpu' \
    trainer.max_epochs=200 \
    +init_from_nemo_model="models/nvidia/stt_en_conformer_ctc_small.nemo"
