source .env

python3 src/scripts/nemo_eval.py \
    --model_dir "$CHECKPOINT_PATH" \
    --dataset_manifest "$TEST_MANIFEST_PATH" \
    --tokenizer_dir "$TOKENIZER_PATH" \
    --output_filename "data/evaluation_transcripts.json" \
    --batch_size 16 \
    --tolerance 0.25 \
    --scores_per_sample
