# Trains the SentencePiece tokenizer on some data

# Note we use a vocab size of 34 as that's the alphabet size for this data
# (At least I think so)

source .env

uv run src/scripts/nemo/process_asr_text_tokenizer.py \
    --manifest="$TRAIN_MANIFEST_PATH" \
    --data_root="models/tokenizers/ipa_tokenizer" \
    --vocab_size=34 \
    --tokenizer=spe \
    --spe_type=char \
    --spe_character_coverage=1.0 \
    --no_lower_case
