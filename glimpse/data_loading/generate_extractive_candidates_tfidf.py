import argparse
import datetime
from pathlib import Path
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import nltk
nltk.download('punkt_tab')
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_path", type=Path, default="data/processed/all_reviews_2017.csv")
    parser.add_argument("--output_dir", type=str, default="data/candidates")
    
    # if ran in a scripted way, the output path will be printed
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)

    # limit the number of samples to generate
    parser.add_argument("--limit", type=int, default=None)

    # number of top sentences to select based on TF-IDF score
    parser.add_argument("--num_top_sentences", type=int, default=5)

    args = parser.parse_args()

    return args


def prepare_dataset(dataset_path) -> Dataset:
    try:
        dataset = pd.read_csv(dataset_path)
    except:
        raise ValueError(f"Unknown dataset {dataset_path}")
    
    # make a dataset from the dataframe
    dataset = Dataset.from_pandas(dataset)

    return dataset


def evaluate_summarizer_with_tfidf(dataset: Dataset, num_sentences=3) -> Dataset:
    """
    Generate extractive summaries using a TF-IDF based approach.

    @param dataset: A dataset with the text
    @return: The same dataset with the summaries added
    """
    print("Loading SentenceTransformer model...")

    summaries = []
    print("Generating summaries with TF-IDF based approach...")

    # (tqdm library for progress bar) 
    for sample in tqdm(dataset):
        text = sample["text"]

        text = text.replace('-----', '\n')
        sentences = nltk.sent_tokenize(text)

        # Remove empty sentences
        sentences = [s for s in sentences if s.strip()] 
        
        # important check to solve some problems
        if len(sentences) < num_sentences:
            # If the number of sentences is less than clusters, use all sentences
            summaries.append(sentences)
            continue

        # Use TF-IDF Vectorizer to compute sentence importance
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        # For each cluster (here based on the number of sentences you want to select)
        # We will select the top 'num_sentences' sentences based on TF-IDF scores

        # Sum the TF-IDF scores for each sentence
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        # Sort sentences by their TF-IDF score in descending order
        sorted_indices = np.argsort(sentence_scores)[::-1]

        # Select the top 'num_sentences' sentences based on TF-IDF score
        selected_sentences = [sentences[i] for i in sorted_indices[:num_sentences]]

        summaries.append(selected_sentences)

    # Add summaries to the huggingface dataset
    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})

    return dataset

def main():
    args = parse_args()
    # load the dataset
    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_path)

    # limit the number of samples
    if args.limit is not None:
        _lim = min(args.limit, len(dataset))
        dataset = dataset.select(range(_lim))

    # generate summaries with tf-idf
    dataset = evaluate_summarizer_with_tfidf(dataset, args.num_top_sentences)

    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.explode("summary")
    df_dataset = df_dataset.reset_index()
    # add an idx with the id of the summary for each example
    df_dataset["id_candidate"] = df_dataset.groupby(["index"]).cumcount()

    # save the dataset
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    output_path = (
        Path(args.output_dir)
        / f"extractive_sentences-_-{args.dataset_path.stem}-_-top_sentences-{args.num_top_sentences}-_-{date}.csv"
    )

    # create output dir if it doesn't exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_dataset.to_csv(output_path, index=False, encoding="utf-8")
    
    # in case of scripted run, print the output path
    if args.scripted_run: print(output_path)


if __name__ == "__main__":
    main()
