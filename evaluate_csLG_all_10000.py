import csv
import numpy as np

PAPER_FILE = "data/processed/papers_clean_10000.csv"
EMBEDDING_FILE = "data/processed/paper_embeddings_10000.npy"

K_VALUES = [1, 3, 5, 10, 20]

print("=" * 60)
print("FULL cs.LG TOP-K EVALUATION - 10,000 PAPERS")
print("=" * 60)

# Load papers
print("\nLoading papers...")
with open(PAPER_FILE, "r", encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

# Load embeddings
print("Loading embeddings...")
embeddings = np.load(EMBEDDING_FILE)

print(f"Papers: {len(papers)}")
print(f"Embeddings: {embeddings.shape}")

if len(papers) != len(embeddings):
    raise ValueError("Dataset and embeddings are not aligned!")

# Normalize
print("\nNormalizing embeddings...")
embeddings = embeddings / np.linalg.norm(
    embeddings, axis=1, keepdims=True
)

# Find all cs.LG papers
lg_indices = [
    i for i, paper in enumerate(papers)
    if "cs.LG" in set(paper["categories"].split(";"))
]

print(f"cs.LG papers: {len(lg_indices)}")

# Results
success = {k: 0 for k in K_VALUES}
total_matches = {k: 0 for k in K_VALUES}

print("\nRunning evaluation...")

for count, i in enumerate(lg_indices, 1):

    # Cosine similarity against all 10,000 papers
    scores = embeddings @ embeddings[i]

    # Don't recommend the paper itself
    scores[i] = -1

    ranked = np.argsort(scores)[::-1]

    for k in K_VALUES:

        top_k = ranked[:k]

        matches = sum(
            "cs.LG" in set(papers[j]["categories"].split(";"))
            for j in top_k
        )

        total_matches[k] += matches

        if matches > 0:
            success[k] += 1

    if count % 100 == 0 or count == len(lg_indices):
        print(
            f"Processed {count}/{len(lg_indices)} cs.LG papers"
        )

print("\n" + "=" * 60)
print("FULL cs.LG RESULTS")
print("=" * 60)

for k in K_VALUES:

    success_rate = 100 * success[k] / len(lg_indices)

    match_rate = (
        100 * total_matches[k]
        / (len(lg_indices) * k)
    )

    print(f"\nTOP-{k}")
    print("-" * 40)
    print(
        f"Targets with >=1 cs.LG match: "
        f"{success[k]}/{len(lg_indices)}"
    )
    print(
        f"Success rate: {success_rate:.2f}%"
    )
    print(
        f"Individual cs.LG match rate: "
        f"{match_rate:.2f}%"
    )

print("\n" + "=" * 60)
print("FINAL INTERPRETATION")
print("=" * 60)

best_k = max(
    K_VALUES,
    key=lambda k: success[k] / len(lg_indices)
)

print(
    f"Best success rate: Top-{best_k}"
)
print(
    f"Success rate: "
    f"{100 * success[best_k] / len(lg_indices):.2f}%"
)

print("\nFull cs.LG evaluation complete.")