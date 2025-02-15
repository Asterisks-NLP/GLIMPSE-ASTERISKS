import argparse
import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

import nltk
import os

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_path", type=Path, default="data/processed/all_reviews_2017.csv")
    parser.add_argument("--output_dir", type=str, default="data/candidates")

    # if ran in a scripted way, the output path will be printed
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)

    # limit the number of samples to generate
    parser.add_argument("--limit", type=int, default=None)

    # in this extension one of the argument (--num_clusters) indicates the number of clusters
    parser.add_argument("--num_clusters", type=int, default=3)

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


def evaluate_summarizer_with_clustering(dataset: Dataset, model_name="all-MiniLM-L6-v2", num_clusters=3) -> Dataset:
    """
    Generate extractive summaries using clustering-based approach.

    @param dataset: A dataset with the text
    @return: The same dataset with the summaries added
    """
    print("Loading SentenceTransformer model...")
    # SentenceTransformer is used to have a vectorial rapresentation of sentences
    model = SentenceTransformer(model_name)

    summaries = []
    print("Generating summaries with clustering-based approach...")

    # (tqdm library for progress bar) 
    for sample in tqdm(dataset):
        text = sample["text"]

        text = text.replace('-----', '\n')
        sentences = nltk.sent_tokenize(text)
        
        # Remove empty sentences
        sentences = [s for s in sentences if s.strip()] 
        
        # here in the original version there is a generic skip to summaries.append(sentences)

        # important check to solve some problems
        if len(sentences) < num_clusters:
            # If the number of sentences is less than clusters, use all sentences
            summaries.append(sentences)
            continue

        # Encode sentences with SentenceTransformer
        embeddings = model.encode(sentences) # transform each sentence in a vector

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Select the most representative sentence for each cluster
        selected_sentences = []
        for cluster_id in range(num_clusters):
            # now we work separately on each cluster, we get all the indeces...
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            # ... to get all the embeddings
            cluster_embeddings = [embeddings[i] for i in cluster_indices]

            center = cluster_centers[cluster_id]

            # for each cluster we select just one sentence, the closest one to the center
            closest_index = cluster_indices[
                min(range(len(cluster_embeddings)), key=lambda i: ((cluster_embeddings[i] - center) ** 2).sum())
            ]
            selected_sentences.append(sentences[closest_index])

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

    # generate summaries with clustering extension
    dataset = evaluate_summarizer_with_clustering(dataset, num_clusters=args.num_clusters)

    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.explode("summary")
    df_dataset = df_dataset.reset_index()
    # add an idx with  the id of the summary for each example
    df_dataset["id_candidate"] = df_dataset.groupby(["index"]).cumcount()

    # save the dataset
    # add unique date in name
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    output_path = (
        Path(args.output_dir)
        / f"extractive_sentences-_-{args.dataset_path.stem}-_-clustering-{args.num_clusters}-_-{date}.csv"
    )

    # create output dir if it doesn't exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_dataset.to_csv(output_path, index=False, encoding="utf-8")
    
    # in case of scripted run, print the output path
    if args.scripted_run:
        print(output_path)


if __name__ == "__main__":
    main()
