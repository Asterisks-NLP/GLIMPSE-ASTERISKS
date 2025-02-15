import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import datetime
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

GENERATION_CONFIGS = {
    "top_p_sampling": {
        "max_new_tokens": 200,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 1.0,
        "num_return_sequences": 8,
        "num_beams": 1,
    },
    **{
        f"sampling_topp_{str(topp).replace('.', '')}": {
            "max_new_tokens": 200,
            "do_sample": True,
            "num_return_sequences": 8,
            "top_p": 0.95,
        }
        for topp in [0.5, 0.8, 0.95, 0.99]
    },
}

# Add base.csv config to all configs
for key, value in GENERATION_CONFIGS.items():
    GENERATION_CONFIGS[key] = {
        "min_length": 0,
        "early_stopping": True,
        **value,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset_path", type=Path, default="data/processed/all_reviews_2017.csv")
    parser.add_argument("--decoding_config", type=str, default="top_p_sampling", choices=GENERATION_CONFIGS.keys())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trimming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default="data/candidates")
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()

def prepare_dataset(dataset_path) -> Dataset:
    try:
        dataset = pd.read_csv(dataset_path)
    except:
        raise ValueError(f"Unknown dataset {dataset_path}")

    # Drop rows with missing or None text
    dataset = dataset.dropna(subset=["text"])

    # Verify dataset integrity
    assert dataset["text"].notna().all(), "Dataset contains None or NaN values in 'text'."

    return Dataset.from_pandas(dataset)

def collate_fn(batch):
    # Filter out examples with None text
    batch = [example for example in batch if example["text"] is not None]
    if not batch:
        raise ValueError("All examples in the batch are invalid (None text).")
    return {key: [d[key] for d in batch] for key in batch[0].keys()}

def evaluate_summarizer(
    model, tokenizer, dataset: Dataset, decoding_config, batch_size: int, device: str, trimming: bool
) -> Dataset:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=trimming, collate_fn=collate_fn)

    summaries = []
    print("Generating summaries...")

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        try:
            text = batch["text"]

            # Remove empty strings or None values
            text = [t for t in text if t]
            if not text:
                print(f"Skipping empty batch {batch_idx}.")
                continue

            inputs = tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model.generate(**inputs, **decoding_config)
            outputs = outputs.reshape(len(text), -1, outputs.shape[-1])

            for b in range(len(text)):
                summaries.append([
                    tokenizer.decode(outputs[b, i], skip_special_tokens=True)
                    for i in range(outputs.shape[1])
                ])
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            raise

    if trimming:
        if len(summaries) > len(dataset):
            summaries = summaries[:len(dataset)]
            print("Summaries truncated to match dataset size.")

        dataset = dataset.select(range(len(summaries)))

    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})

    return dataset

def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")

def main():
    args = parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    model = model.to(args.device)

    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_path)

    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    dataset = evaluate_summarizer(
        model,
        tokenizer,
        dataset,
        GENERATION_CONFIGS[args.decoding_config],
        args.batch_size,
        args.device,
        args.trimming,
    )

    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.explode('summary').reset_index()
    df_dataset['id_candidate'] = df_dataset.groupby(['index']).cumcount()

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = sanitize_model_name(args.model_name)
    padding_status = "trimmed" if args.trimming else "padded"
    output_path = (
        Path(args.output_dir)
        / f"{model_name}-_-{args.dataset_path.stem}-_-{args.decoding_config}-_-{padding_status}-_-{date}.csv"
    )

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_dataset.to_csv(output_path, index=False, encoding="utf-8")

    if args.scripted_run:
        print(output_path)

if __name__ == "__main__":
    main()
