from transformers import TrainerCallback


class PreviewCallback(TrainerCallback):
    """Shows a sample evaluation at the end of a training epoch"""

    def __init__(self, val_dataset, data_collator, tokenizer):
        super().__init__()
        self.val_dataset = val_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        # Take 3 samples from the validation set
        keys_to_keep = ["audio_values", "labels"]
        batch_list = [
            {k: v for k, v in self.val_dataset[i].items() if k in keys_to_keep}
            for i in range(3)
        ]

        inputs = self.data_collator(batch_list)

        audio = inputs["audio_values"].to(args.device)
        attention_mask = (audio != 0).long()

        model_inputs = {
            "audio_values": audio,
            "attention_mask": attention_mask,
        }

        generated_tokens = kwargs["model"].generate(**model_inputs, max_new_tokens=50)

        preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        labels = inputs["labels"].clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        print("=" * 40)
        print(f"Epoch: {state.epoch}")
        print("-" * 40)
        for p, l in zip(preds, labels):
            print(f"Target:  {l}")
            print(f"Predict: {p}")
        print("=" * 40)
