python src/scripts/nemo/process_asr_text_tokenizer.py \
  --manifest=data/nexdata/filipino_822/train_manifest.json \
  --data_root=models/tokenizers/ipa_tokenizer \
  --vocab_size=120 \
  --tokenizer=spe \
  --spe_type=char \
  --spe_character_coverage=1.0 \
  --no_lower_case
