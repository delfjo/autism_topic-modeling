# need new graphs because previous ones are not as readable (replace keywords by self-assigned labels, etc.)
# run with:
# python visualize.py --comparison-csv results/topic_comparison.csv --outdir plots

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TOPIC_LABELS = {
    1: "Music hyperfixations",
    2: "Special interests and favorite objects",
    3: "Clothing sensory sensitivity and comfort",
    4: "Hygiene challenges",
    5: "Noise sensitivity and hearing protection",
    6: "Restricted eating and safe foods",
    7: "Stimming and repetitive behaviors",
    8: "Community and social connection",
    9: "Hair sensory sensitivity",
    10: "Driving anxiety and sensory overload",
    11: "Feeling confused or excluded in social events",
    12: "Special interests and hyperfixations",
    13: "Uncertainty in relationships and social interaction",
    14: "Loud noise sensitivity",
    15: "Skin and tactile sensitivity",
    16: "Smell sensitivity",
    17: "Overstimulation",
    18: "Fidget toys",
    19: "Temperature sensitivity",
    20: "Meltdowns and frustrations",
    21: "Communication",
}

ALWAYS_EXCLUDED_TOPICS = {-1, 0}

SUBREDDIT_COLORS = {
    "autism": "#CCEDAB",
    "autisminwomen": "#BEABED",
}


def find_share_columns(df: pd.DataFrame) -> tuple[str, str]:

    share_cols = [c for c in df.columns if c.startswith("share_") and c != "share_diff"]
    return share_cols[0], share_cols[1]


def plot_topic_shares(comparison_csv: Path, outdir: Path):
    df = pd.read_csv(comparison_csv, index_col=0)
    df.index.name = "Topic"

    share_a_col, share_b_col = find_share_columns(df)
    subreddit_a = share_a_col.replace("share_", "")
    subreddit_b = share_b_col.replace("share_", "")

    color_a = SUBREDDIT_COLORS[subreddit_a]
    color_b = SUBREDDIT_COLORS[subreddit_b]

    df = df.drop(index=[t for t in ALWAYS_EXCLUDED_TOPICS if t in df.index])

    df["custom_label"] = [f"{t}. {TOPIC_LABELS[t]}" for t in df.index]

    df["pct_a"] = df[share_a_col] * 100
    df["pct_b"] = df[share_b_col] * 100

    df = df.sort_values("share_diff", ascending=True)

    n_topics = len(df)
    fig_height = max(8, 0.5 * n_topics)  # scale figure height to topic count
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = range(n_topics)
    bar_height = 0.38

    ax.barh(
        [i + bar_height / 2 for i in y],
        df["pct_a"],
        height=bar_height,
        label=subreddit_a,
        color=color_a,
    )
    ax.barh(
        [i - bar_height / 2 for i in y],
        df["pct_b"],
        height=bar_height,
        label=subreddit_b,
        color=color_b,
    )

    for i, (pa, pb) in enumerate(zip(df["pct_a"], df["pct_b"])):
        ax.text(pa + 0.15, i + bar_height / 2, f"{pa:.1f}%", va="center", fontsize=9)
        ax.text(pb + 0.15, i - bar_height / 2, f"{pb:.1f}%", va="center", fontsize=9)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["custom_label"], fontsize=10)
    ax.set_xlabel("Share of subreddit's posts (%)", fontsize=11)
    ax.set_title(f"Topic share by subreddit: r/{subreddit_a} vs r/{subreddit_b}", fontsize=13)

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    max_val = max(df["pct_a"].max(), df["pct_b"].max())
    ax.set_xlim(0, max_val * 1.15)

    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    out_path = outdir / "topic_share_by_subreddit.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot topic share per subreddit with custom labels")
    parser.add_argument("--comparison-csv", default="results/topic_comparison.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_topic_shares(comparison_csv=Path(args.comparison_csv), outdir=outdir)


if __name__ == "__main__":
    main()
