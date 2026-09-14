import csv
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

PAPERS_FILE = Path("data/processed/papers_clean_10000.csv")
EMBEDDINGS_FILE = Path("data/processed/paper_embeddings_10000.npy")

print("=" * 60)
print("10,000-PAPER RECOMMENDER EVALUATION")
print("=" * 60)

# Load papers
with open(PAPERS_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

# Load embeddings
embeddings = np.load(EMBEDDINGS_FILE)

print(f"\nPapers: {len(papers)}")
print(f"Embeddings: {embeddings.shape}")

if len(papers) != len(embeddings):
    raise ValueError("Number of papers and embeddings do not match.")

# Select 20 papers spread through the dataset
test_indices = np.linspace(
    0, len(papers) - 1, 20, dtype=int
)

category_match_counts = []
top1_scores = []
top5_scores = []

print("\nEvaluating 20 papers...")
print("=" * 60)

for target_index in test_indices:

    target = papers[target_index]

    target_categories = set(
        target["categories"].split(";")
    )

    similarities = cosine_similarity(
        embeddings[target_index].reshape(1, -1),
        embeddings
    )[0]

    # Remove the paper itself
    similarities[target_index] = -1

    top_indices = np.argsort(similarities)[::-1][:5]

    matches = 0

    for index in top_indices:
        recommended_categories = set(
            papers[index]["categories"].split(";")
        )

        if target_categories & recommended_categories:
            matches += 1

    category_match_counts.append(matches)

    top1_scores.append(similarities[top_indices[0]])
    top5_scores.append(np.mean(similarities[top_indices]))

    print(f"\nTarget #{target_index + 1}: {target['title']}")
    print(f"Categories: {target['categories']}")

    print("\nTop 5:")
    for rank, index in enumerate(top_indices, 1):
        shared = target_categories & set(
            papers[index]["categories"].split(";")
        )

        print(
            f"{rank}. {papers[index]['title']}"
        )
        print(
            f"   Similarity: {similarities[index]:.4f}"
        )
        print(
            f"   Shared categories: "
            f"{', '.join(shared) if shared else 'None'}"
        )

# Calculate results
total_recommendations = len(test_indices) * 5
category_matches = sum(category_match_counts)

category_match_rate = (
    category_matches / total_recommendations * 100
)

average_top1 = np.mean(top1_scores)
average_top5 = np.mean(top5_scores)

print("\n" + "=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print(f"\nPapers evaluated: {len(test_indices)}")
print(f"Recommendations checked: {total_recommendations}")

print(
    f"Category-matching recommendations: "
    f"{category_matches}"
)

print(
    f"Category match rate: "
    f"{category_match_rate:.2f}%"
)

print(
    f"Average Top-1 similarity: "
    f"{average_top1:.4f}"
)

print(
    f"Average Top-5 similarity: "
    f"{average_top5:.4f}"
)

print("\nEvaluation complete.")