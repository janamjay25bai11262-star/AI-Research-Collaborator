import csv
import numpy as np
from pathlib import Path

PROFILE_FILE = Path("data/processed/researcher_profiles_clean.csv")
PAPER_FILE = Path("data/processed/papers_clean_10000.csv")
RESEARCHER_EMBEDDING_FILE = Path("data/processed/researcher_embeddings.npy")
PAPER_EMBEDDING_FILE = Path("data/processed/paper_embeddings_10000.npy")

K_VALUES = [1, 3, 5, 10, 20]


def get_categories(text):
    return set(
        x.strip()
        for x in text.split(";")
        if x.strip()
    )


print("=" * 60)
print("ALL-RESEARCHER -> PAPER TOP-K EVALUATION")
print("=" * 60)

print("\nLoading researchers...")
with open(PROFILE_FILE, "r", encoding="utf-8") as file:
    researchers = list(csv.DictReader(file))

print("Researchers:", len(researchers))

print("\nLoading papers...")
with open(PAPER_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

print("Papers:", len(papers))

print("\nLoading embeddings...")
researcher_embeddings = np.load(RESEARCHER_EMBEDDING_FILE)
paper_embeddings = np.load(PAPER_EMBEDDING_FILE)

print("Researcher embeddings:", researcher_embeddings.shape)
print("Paper embeddings:", paper_embeddings.shape)

# Safety checks
assert len(researchers) == researcher_embeddings.shape[0], \
    "Researcher count and embeddings do not match."

assert len(papers) == paper_embeddings.shape[0], \
    "Paper count and embeddings do not match."

print("\nDataset and embeddings are aligned.")

# Normalize embeddings
print("\nNormalizing embeddings...")

researcher_embeddings = researcher_embeddings / (
    np.linalg.norm(
        researcher_embeddings,
        axis=1,
        keepdims=True
    ) + 1e-12
)

paper_embeddings = paper_embeddings / (
    np.linalg.norm(
        paper_embeddings,
        axis=1,
        keepdims=True
    ) + 1e-12
)

print("Embeddings normalized.")

# Counters
success_counts = {k: 0 for k in K_VALUES}
match_counts = {k: 0 for k in K_VALUES}
total_recommendations = {k: 0 for k in K_VALUES}

researcher_count = len(researchers)

print("\n" + "=" * 60)
print("RUNNING EVALUATION")
print("=" * 60)

for r in range(researcher_count):

    researcher = researchers[r]

    researcher_name = researcher.get(
        "researcher_name",
        f"Researcher {r + 1}"
    )

    researcher_categories = get_categories(
        researcher.get("categories", "")
    )

    # Similarity against all 10,000 papers
    scores = paper_embeddings @ researcher_embeddings[r]

    ranked_indices = np.argsort(scores)[::-1]

    for k in K_VALUES:

        top_indices = ranked_indices[:k]

        matches = 0

        for paper_index in top_indices:

            paper_categories = get_categories(
                papers[paper_index].get("categories", "")
            )

            if researcher_categories & paper_categories:
                matches += 1

        # At least one relevant category match
        if matches > 0:
            success_counts[k] += 1

        match_counts[k] += matches
        total_recommendations[k] += k

    # Progress every 25 researchers
    if (r + 1) % 25 == 0 or r + 1 == researcher_count:
        print(
            f"Processed {r + 1}/{researcher_count} researchers"
        )

print("\n" + "=" * 60)
print("OVERALL TOP-K RESULTS")
print("=" * 60)

for k in K_VALUES:

    success_rate = (
        success_counts[k] /
        researcher_count *
        100
    )

    individual_match_rate = (
        match_counts[k] /
        total_recommendations[k] *
        100
    )

    print(f"\nTOP-{k}")
    print("-" * 40)

    print(
        f"Researchers with >=1 match: "
        f"{success_counts[k]}/{researcher_count}"
    )

    print(
        f"Top-{k} success rate: "
        f"{success_rate:.2f}%"
    )

    print(
        f"Individual category match rate: "
        f"{individual_match_rate:.2f}%"
    )

print("\n" + "=" * 60)
print("FINAL INTERPRETATION")
print("=" * 60)

best_k = max(
    K_VALUES,
    key=lambda k: success_counts[k]
)

best_rate = (
    success_counts[best_k] /
    researcher_count *
    100
)

print(
    f"\nBest Top-K by success rate: "
    f"Top-{best_k}"
)

print(
    f"Success rate: {best_rate:.2f}%"
)

if best_rate >= 80:
    print(
        "Strong researcher-to-paper "
        "category retrieval performance."
    )
elif best_rate >= 60:
    print(
        "Moderate researcher-to-paper "
        "category retrieval performance."
    )
else:
    print(
        "Weak researcher-to-paper "
        "category retrieval performance."
    )

print("\n" + "=" * 60)
print("ALL-RESEARCHER TOP-K EVALUATION COMPLETE")
print("=" * 60)