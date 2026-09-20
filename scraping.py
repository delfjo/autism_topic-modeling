import requests
import json
import time
import random
import os
from datetime import datetime, timezone

SUBREDDITS = ["autism", "autisminwomen"]
TARGET = 5000


def save_checkpoint(posts, before, checkpoint_file):
    checkpoint = {
        "posts": posts,
        "before": before
    }

    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(
            checkpoint,
            f,
            indent=2,
            ensure_ascii=False
        )


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        print(f"resuming from {len(checkpoint['posts'])} posts")

        return (
            checkpoint["posts"],
            checkpoint["before"]
        )

    return [], None


session = requests.Session()


for SUBREDDIT in SUBREDDITS:

    CHECKPOINT_FILE = f"{SUBREDDIT}_checkpoint.json"
    OUTPUT_FILE = f"{SUBREDDIT}_posts.json"

    # load previous progress
    posts, before = load_checkpoint(CHECKPOINT_FILE)

    # track IDs already downloaded to avoid duplicates
    seen_ids = {
        post["id"]
        for post in posts
        if "id" in post
    }

    while len(posts) < TARGET:

        url = "https://api.pullpush.io/reddit/search/submission/"

        params = {
            "subreddit": SUBREDDIT,
            "size": 100
        }

        if before:
            params["before"] = before

        # request with retry handling, getting rate limited the whole time
        while True:

            try:
                response = session.get(
                    url,
                    params=params,
                    timeout=30
                )

                if response.status_code == 429:
                    wait = random.randint(180, 300)

                    print(
                        f"rate limited. waiting {wait // 60} minutes..."
                    )

                    time.sleep(wait)
                    continue

                response.raise_for_status()
                break

            except requests.exceptions.RequestException as e:

                print("request failed:", e)
                print("retrying in 60 seconds...")

                time.sleep(60)

        data = response.json()["data"]

        if not data:
            print("no more posts available.")
            break

        new_posts = 0

        for post in data:

            reddit_id = post.get("id")

            # skip duplicates
            if reddit_id in seen_ids:
                continue

            created = post.get("created_utc")

            if created:
                date = datetime.fromtimestamp(
                    created,
                    tz=timezone.utc
                ).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )

            else:
                date = None

            posts.append(
                {
                    "id": reddit_id,
                    "title": post.get("title", ""),
                    "body": post.get("selftext", ""),
                    "date": date
                }
            )

            seen_ids.add(reddit_id)
            new_posts += 1

        # move pagination backwards
        before = data[-1]["created_utc"]

        # save checkpoint after every batch just in case
        save_checkpoint(
            posts,
            before,
            CHECKPOINT_FILE
        )

        print(
            f"downloaded {len(posts)}/{TARGET} posts "
            f"from r/{SUBREDDIT} "
            f"(+{new_posts} new)"
        )

        # slow down requests, otherwise issues
        time.sleep(
            random.uniform(20, 40)
        )

    posts = posts[:TARGET]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            posts,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"finished r/{SUBREDDIT}. "
        f"saved {len(posts)} posts to {OUTPUT_FILE}"
    )
