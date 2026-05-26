import io
from pathlib import Path

import pandas as pd
import soundfile as sf


def main():
    target_idx = 12615
    base_dir = Path("data/filipinospeechcorpus/data")
    output_wav = f"worst_sample_{target_idx}.wav"

    print(f"1. Loading evaluation data for index {target_idx}...")
    try:
        df_eval = pd.read_json("evaluation_transcripts.json", lines=True)
        row_eval = df_eval.loc[target_idx]
    except KeyError:
        print(f"Error: Index {target_idx} not found in evaluation_transcripts.json.")
        return
    except FileNotFoundError:
        print("Error: evaluation_transcripts.json not found.")
        return

    virtual_path = row_eval["audio_filepath"]
    print(f"   Found Virtual Path: {virtual_path}")

    # 2. Parse the fsc:// virtual path
    # Example: fsc://train-00022-of-00139.parquet/sample_000426.wav
    path_parts = virtual_path.replace("fsc://", "").split("/")
    parquet_filename = path_parts[0]
    row_idx = int(path_parts[1].split("_")[1].split(".")[0])

    print(f"\n2. Opening Parquet file: {parquet_filename}")
    print(f"   Targeting row index: {row_idx}")

    parquet_path = base_dir / parquet_filename
    if not parquet_path.exists():
        print(f"Error: Could not find parquet file at {parquet_path}")
        return

    df_parquet = pd.read_parquet(parquet_path)
    parquet_row = df_parquet.iloc[row_idx]

    # 3. Extract and save the audio bytes
    print(f"\n3. Extracting audio binary and saving to {output_wav}...")
    audio_bytes = parquet_row["audio"]["bytes"]
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    sf.write(output_wav, arr, sr)
    print("   Audio extraction successful!")

    # 4. Print textual data
    print("\n" + "=" * 60)
    print(" EVALUATION DATA (PHONEMES)")
    print("=" * 60)
    print(f"Ground Truth (text): {row_eval['text']}")
    print(f"Predicted Text     : {row_eval['pred_text']}")
    print(f"CER Score          : {row_eval['cer']:.2f}")

    print("\n" + "=" * 60)
    print(" ORIGINAL PARQUET DATA (GRAPHEMES & METADATA)")
    print("=" * 60)
    # Dynamically print all columns from the Parquet row (except the massive audio byte array)
    for col in df_parquet.columns:
        if col != "audio":
            print(f"{col.capitalize():<15}: {parquet_row[col]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
