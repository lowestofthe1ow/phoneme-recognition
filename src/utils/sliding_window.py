import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


def predict_sliding_window_batch(
    sentences, model, tokenizer, window_size=11, max_batch_size=32
):
    """
    Processes a batch of sentences using a sliding window for G2P inference.
    Flattens all windows across the batch to maximize GPU throughput.
    """
    device = model.device
    all_windows = []
    window_metadata = []

    for sent_idx, sentence in enumerate(sentences):
        words = sentence.split()
        N = len(words)

        if N <= window_size:
            all_windows.append(sentence)
            window_metadata.append((sent_idx, True, -1))
        else:
            half_window = window_size // 2

            for i in range(N):
                start = i - half_window
                end = start + window_size

                if start < 0:
                    start = 0
                    end = window_size
                if end > N:
                    end = N
                    start = N - window_size

                window_words = words[start:end]
                window_text = " ".join(window_words).capitalize()

                all_windows.append(window_text)
                window_metadata.append((sent_idx, False, i - start))

    all_decoded = []
    for i in range(0, len(all_windows), max_batch_size):
        batch_windows = all_windows[i : i + max_batch_size]
        inputs = tokenizer(
            batch_windows, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
            )
        all_decoded.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    final_predictions = [""] * len(sentences)
    reconstructed_words = {i: [] for i in range(len(sentences))}

    for decoded_text, (sent_idx, is_whole, target_idx) in zip(
        all_decoded, window_metadata
    ):
        if is_whole:
            final_predictions[sent_idx] = decoded_text
        else:
            phoneme_words = decoded_text.split()
            if phoneme_words:
                safe_idx = min(target_idx, len(phoneme_words) - 1)
                reconstructed_words[sent_idx].append(phoneme_words[safe_idx])
            else:
                reconstructed_words[sent_idx].append("")

    for sent_idx in range(len(sentences)):
        if not final_predictions[sent_idx]:
            final_predictions[sent_idx] = " ".join(reconstructed_words[sent_idx])

    return final_predictions
