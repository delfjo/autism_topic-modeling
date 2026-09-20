"""
install dependencies:
pip install bertopic sentence-transformers scikit-learn pandas matplotlib

run with:
    python topicmodeling.py \
        --file autism:autism_posts.json \
        --file autisminwomen:autisminwomen_posts.json \
        --outdir results
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from umap import UMAP


# custom stopwords because otherwise they're everywhere
CUSTOM_STOPWORDS = [
    "like",
    "just",
]

STOPWORDS = set(ENGLISH_STOP_WORDS) | set(CUSTOM_STOPWORDS)


# load and clean

REMOVED_MARKERS = {"[removed]", "[deleted]", ""}
URL_RE = re.compile(r"http\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
MARKDOWN_RE = re.compile(r"[*_~`>#\[\]()]")


def clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MARKDOWN_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_posts(path: str, subreddit_label: str, min_words: int = 5) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for post in raw:
        title = (post.get("title") or "").strip()
        body = (post.get("body") or "").strip()

        if body in REMOVED_MARKERS:
            body = ""

        text = clean_text(f"{title}. {body}" if body else title)

        if len(text.split()) < min_words:
            continue

        rows.append(
            {
                "id": post.get("id"),
                "subreddit": subreddit_label,
                "date": post.get("date"),
                "text": text,
            }
        )

    df = pd.DataFrame(rows)
    print(f"[{subreddit_label}] loaded {len(raw)} posts -> {len(df)} usable after cleaning/filtering")
    return df


# topic modeling

def fit_topic_model(
    docs: list[str],
    nr_topics="auto",
    min_topic_size: int = 20,
    min_samples: int = 5,
    cluster_selection_method: str = "leaf",
    n_components: int = 10,
    n_neighbors: int = 15,
) -> BERTopic:

    vectorizer_model = CountVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=5,
    )

    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.3),
    }

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )

    topic_model.fit_transform(docs)
    return topic_model, vectorizer_model, representation_model, ctfidf_model


# compare

def build_comparison_table(df: pd.DataFrame, topic_model: BERTopic) -> pd.DataFrame:
    subreddits = df["subreddit"].unique().tolist()
    a, b = subreddits

    counts = (
        df.groupby(["topic", "subreddit"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=subreddits, fill_value=0)
    )

    props = counts.div(counts.sum(axis=0), axis=1)  # share within each subreddit

    info = topic_model.get_topic_info().set_index("Topic")
    label_col = "Representation" if "Representation" in info.columns else "Name"

    table = pd.DataFrame(
        {
            "label": info.loc[counts.index, label_col].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x
            ),
            f"count_{a}": counts[a],
            f"count_{b}": counts[b],
            f"share_{a}": props[a],
            f"share_{b}": props[b],
        }
    )
    table["share_diff"] = table[f"share_{a}"] - table[f"share_{b}"]
    table["log_ratio"] = (
        (table[f"share_{a}"] + 1e-6).apply(lambda x: x)
        .div(table[f"share_{b}"] + 1e-6)
        .apply(lambda x: __import__("math").log2(x))
    )
    table = table.sort_values("share_diff", ascending=False)
    return table, a, b


def plot_top_differences(table: pd.DataFrame, a: str, b: str, outdir: Path, top_n: int = 15):
    plot_df = table[table.index != -1].copy()
    top = pd.concat([plot_df.head(top_n), plot_df.tail(top_n)])
    top = top.sort_values("share_diff")

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(top))))
    colors = ["#4C72B0" if v < 0 else "#DD8452" for v in top["share_diff"]]
    ax.barh(top["label"], top["share_diff"] * 100, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"% share difference  ( {b}  <---  0  --->  {a} )")
    ax.set_title(f"Topics most over/under-represented: r/{a} vs r/{b}")
    fig.tight_layout()
    out_path = outdir / "topic_share_differences.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def strip_stopwords_from_labels(topic_info: pd.DataFrame) -> pd.DataFrame:
    def clean_words(words):
        return [w for w in words if w.lower() not in STOPWORDS]

    def clean_name(name: str) -> str:
        parts = name.split("_")
        prefix, words = parts[0], parts[1:]
        kept = clean_words(words)
        return "_".join([prefix] + kept) if kept else name

    topic_info = topic_info.copy()
    if "Name" in topic_info.columns:
        topic_info["Name"] = topic_info["Name"].apply(clean_name)
    if "Representation" in topic_info.columns:
        topic_info["Representation"] = topic_info["Representation"].apply(
            lambda words: clean_words(words) if isinstance(words, list) else words
        )
    return topic_info


# subclustering the super-topic

def subcluster_topic(
    df: pd.DataFrame,
    topic_id: int,
    outdir: Path,
    min_topic_size: int,
    min_samples: int,
    cluster_selection_method: str,
    n_components: int,
    n_neighbors: int,
):

    sub_df = df[df["topic"] == topic_id].reset_index(drop=True)
    print(f"\n=== Sub-clustering topic {topic_id} ({len(sub_df)} documents) ===")

    if len(sub_df) < min_topic_size * 3:
        print(
            f"skipping: only {len(sub_df)} documents, too few to sub-cluster "
            f"meaningfully with min_topic_size={min_topic_size}."
        )
        return

    sub_outdir = outdir / f"subcluster_topic_{topic_id}"
    sub_outdir.mkdir(parents=True, exist_ok=True)

    sub_topic_model, sub_vectorizer_model, sub_representation_model, sub_ctfidf_model = fit_topic_model(
        sub_df["text"].tolist(),
        nr_topics="auto",
        min_topic_size=min_topic_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        n_components=n_components,
        n_neighbors=n_neighbors,
    )
    sub_df["topic"] = sub_topic_model.topics_

    n_outliers = (sub_df["topic"] == -1).sum()
    print(f"sub-cluster outliers: {n_outliers}/{len(sub_df)}")
    print("sub-cluster topic sizes:")
    print(sub_df["topic"].value_counts().head(15))

    new_sub_topics = sub_topic_model.reduce_outliers(
        sub_df["text"].tolist(),
        sub_topic_model.topics_,
        strategy="embeddings",
        threshold=0.3,
    )
    sub_topic_model.update_topics(
        sub_df["text"].tolist(),
        topics=new_sub_topics,
        vectorizer_model=sub_vectorizer_model,
        ctfidf_model=sub_ctfidf_model,
        representation_model=sub_representation_model,
    )
    sub_df["topic"] = new_sub_topics

    sub_topic_info = sub_topic_model.get_topic_info()
    sub_topic_info = strip_stopwords_from_labels(sub_topic_info)
    sub_topic_info.to_csv(sub_outdir / "topic_info.csv", index=False)
    sub_df.to_csv(sub_outdir / "doc_topic_assignments.csv", index=False)

    if sub_df["subreddit"].nunique() == 2:
        sub_comparison, sub_a, sub_b = build_comparison_table(sub_df, sub_topic_model)
        sub_comparison.to_csv(sub_outdir / "topic_comparison.csv")
        plot_top_differences(sub_comparison, sub_a, sub_b, sub_outdir)
    else:
        print("only one subreddit present in this topic, skipping comparison table.")

    print(f"sub-cluster results saved to {sub_outdir}")


# main

def parse_file_arg(value: str):
    label, path = value.split(":", 1)
    return label, path


def main():
    parser = argparse.ArgumentParser(description="Compare topics across two subreddits with BERTopic")
    parser.add_argument(
        "--file",
        action="append",
        type=parse_file_arg,
        required=True,
        help="label:path.json — pass twice, once per subreddit",
    )
    parser.add_argument("--outdir", default="results", help="Directory to write outputs to")
    parser.add_argument("--min-topic-size", type=int, default=20)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="HDBSCAN min_samples. Lower = fewer outliers but higher chaining risk (one dominant topic).",
    )
    parser.add_argument(
        "--cluster-selection-method",
        default="leaf",
        choices=["leaf", "eom"],
        help='"leaf" favors several evenly-sized clusters; "eom" allows fewer, larger, uneven ones.',
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="UMAP dimensions retained before clustering. Higher preserves more structure (default 10; BERTopic default is 5).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors. Lower = more local structure; higher = more global structure.",
    )
    parser.add_argument("--nr-topics", default="auto", help='"auto" or an integer to force a number of topics')
    parser.add_argument("--min-words", type=int, default=5, help="Drop posts shorter than this many words")
    parser.add_argument(
        "--outlier-strategy",
        default="embeddings",
        choices=["embeddings", "c-tf-idf", "distributions"],
        help="How to reassign HDBSCAN outliers (-1) to real topics",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=0.3,
        help="Minimum similarity required to reassign an outlier; below this it stays -1",
    )
    parser.add_argument(
        "--no-subcluster",
        action="store_true",
        help="Skip automatically sub-clustering the largest topic",
    )
    parser.add_argument(
        "--subcluster-min-topic-size",
        type=int,
        default=15,
        help="min_topic_size used when sub-clustering the largest topic (usually smaller than the main run's)",
    )
    args = parser.parse_args()

    if len(args.file) != 2:
        parser.error("Pass exactly two --file arguments, one per subreddit")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load
    dfs = [load_posts(path, label, min_words=args.min_words) for label, path in args.file]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Combined corpus: {len(df)} documents")

    # fit shared model
    nr_topics = args.nr_topics
    if nr_topics != "auto":
        nr_topics = int(nr_topics)

    topic_model, vectorizer_model, representation_model, ctfidf_model = fit_topic_model(
        df["text"].tolist(),
        nr_topics=nr_topics,
        min_topic_size=args.min_topic_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
    )
    df["topic"] = topic_model.topics_

    n_outliers_before = (df["topic"] == -1).sum()
    print(f"outliers before reduction: {n_outliers_before}/{len(df)}")

    print("\ntopic sizes BEFORE outlier reassignment (top 10):")
    print(df["topic"].value_counts().head(10))

    new_topics = topic_model.reduce_outliers(
        df["text"].tolist(),
        topic_model.topics_,
        strategy=args.outlier_strategy,
        threshold=args.outlier_threshold,
    )
    topic_model.update_topics(
        df["text"].tolist(),
        topics=new_topics,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
    )
    df["topic"] = new_topics

    n_outliers_after = (df["topic"] == -1).sum()
    print(f"\noutliers after reduction: {n_outliers_after}/{len(df)}")

    print("\ntopic sizes AFTER outlier reassignment (top 10):")
    print(df["topic"].value_counts().head(10))

    topic_info = topic_model.get_topic_info()
    topic_info = strip_stopwords_from_labels(topic_info)
    topic_info.to_csv(outdir / "topic_info.csv", index=False)
    df.to_csv(outdir / "doc_topic_assignments.csv", index=False)

    comparison, a, b = build_comparison_table(df, topic_model)
    comparison.to_csv(outdir / "topic_comparison.csv")
    plot_top_differences(comparison, a, b, outdir)

    topics_per_class = topic_model.topics_per_class(df["text"].tolist(), classes=df["subreddit"].tolist())
    topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=20).write_html(
        str(outdir / "topics_per_subreddit.html")
    )

    topic_model.save(str(outdir / "bertopic_model"), serialization="safetensors", save_ctfidf=True)

    if not args.no_subcluster:
        largest_topic_id = df.loc[df["topic"] != -1, "topic"].value_counts().idxmax()
        subcluster_topic(
            df,
            topic_id=largest_topic_id,
            outdir=outdir,
            min_topic_size=args.subcluster_min_topic_size,
            min_samples=args.min_samples,
            cluster_selection_method=args.cluster_selection_method,
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
        )

    print("\nDone. Key outputs:")
    print(f"  {outdir / 'topic_info.csv'}                 - all topics + top keywords")
    print(f"  {outdir / 'topic_comparison.csv'}            - per-topic share in each subreddit + difference")
    print(f"  {outdir / 'topic_share_differences.png'}     - bar chart of biggest differences")
    print(f"  {outdir / 'topics_per_subreddit.html'}       - interactive comparison view")
    print(f"  {outdir / 'doc_topic_assignments.csv'}       - every post with its assigned topic")
    if not args.no_subcluster:
        print(f"  {outdir / f'subcluster_topic_{largest_topic_id}'}/          - structure found inside the largest topic")


if __name__ == "__main__":
    main()
