# Evaluates a model on a test dataset then shows WER/CER

source .env

uv run src/scripts/nemo/speech_to_text_eval.py \
    model_path="$CHECKPOINT_PATH" \
    dataset_manifest="$TEST_MANIFEST_PATH" \
    batch_size=16 \
    amp=True \
    use_cer=True
