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
    parser.add_argument("--th", type=float, default=0.5)

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

def evaluate_summarizer_with_tfidf(dataset: Dataset, th = 0.5) -> Dataset:
    """
    Generate extractive summaries using a dynamic TF-IDF based approach.
    For each text, a different number of sentences is selected based on a
    dynamic threshold computed on the TF-IDF scores.
    @param dataset: A dataset with the text.
    @return: The same dataset with the summaries added.
    """
    print("Generating summaries with dynamic TF-IDF based approach...")
    summaries = []
    
    for sample in tqdm(dataset):
        text = sample["text"]
        text = text.replace('-----', '\n')
        sentences = nltk.sent_tokenize(text)
        
        # Remove empty sentences
        sentences = [s for s in sentences if s.strip()] 
        

        if len(sentences) == 0:
            summaries.append([])
            continue
        if len(sentences) < 3:
            summaries.append(sentences)
            continue

        # Compute TF-IDF scores for each sentence
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        # Compute a dynamic threshold based on the mean and standard deviation
        mean_score = np.mean(sentence_scores)
        std_score = np.std(sentence_scores)
        threshold = mean_score + th * std_score  # il fattore 0.5 può essere regolato

        # Seleziona le frasi con punteggio superiore alla soglia
        selected_indices = np.where(sentence_scores >= threshold)[0]
        # Se nessuna frase supera la soglia, seleziona la frase con il punteggio massimo
        if len(selected_indices) == 0:
            selected_indices = [np.argmax(sentence_scores)]
        
        # Ordina le frasi selezionate in base al punteggio in ordine decrescente
        sorted_selected = sorted(selected_indices, key=lambda i: sentence_scores[i], reverse=True)
        selected_sentences = [sentences[i] for i in sorted_selected]

        summaries.append(selected_sentences)
    
    # Add summaries to the HuggingFace dataset
    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})
    return dataset

def main():
    args = parse_args()
    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_path)

    if args.limit is not None:
        _lim = min(args.limit, len(dataset))
        dataset = dataset.select(range(_lim))

    # generate summaries with dynamic TF-IDF selection
    dataset = evaluate_summarizer_with_tfidf(dataset, args.th)

    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.explode("summary")
    df_dataset = df_dataset.reset_index()
    df_dataset["id_candidate"] = df_dataset.groupby(["index"]).cumcount()

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    output_path = (
        Path(args.output_dir)
        / f"extractive_sentences-_-{args.dataset_path.stem}-_-dynamic_tfidf_{args.th}-_-{date}.csv"
    )
    
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_dataset.to_csv(output_path, index=False, encoding="utf-8")
    
    if args.scripted_run:
        print(output_path)

if __name__ == "__main__":
    main()