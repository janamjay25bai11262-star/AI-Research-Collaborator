import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PAPER_FILE = "data/processed/papers_clean.csv"

# Categories that are too broad to be useful for this evaluation
IGNORED_CATEGORIES = {"cs.AI"}

# Load papers
with open(PAPER_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

print("=" * 60)
print("SPECIFIC CATEGORY SIMILARITY EVALUATION")
print("=" * 60)

print(f"\nNumber of papers: {len(papers)}")

# Load model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Title + abstract
paper_texts = [
    f"{paper['title']}. {paper['abstract']}"
    for paper in papers
]

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(
    paper_texts,
    show_progress_bar=True
)

# Similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Test 10 papers
test_indices = np.linspace(
    0,
    len(papers) - 1,
    10,
    dtype=int
)

total_matches = 0
total_comparisons = 0

for paper_index in test_indices:

    target_categories = (
        set(papers[paper_index]["categories"].split(";"))
        - IGNORED_CATEGORIES
    )

    scores = similarity_matrix[paper_index]
    similar_indices = np.argsort(scores)[::-1]

    matches = 0
    checked = 0

    print("\n" + "=" * 60)
    print(f"TARGET: {papers[paper_index]['title']}")
    print("=" * 60)

    print(
        f"\nSpecific categories: "
        f"{', '.join(target_categories) if target_categories else 'None'}"
    )

    for index in similar_indices:

        if index == paper_index:
            continue

        similar_categories = (
            set(papers[index]["categories"].split(";"))
            - IGNORED_CATEGORIES
        )

        shared_categories = target_categories & similar_categories

        checked += 1
        total_comparisons += 1

        if shared_categories:
            matches += 1
            total_matches += 1

        print(
            f"\n{checked}. {papers[index]['title']}"
        )

        print(
            f"   Similarity: {scores[index]:.4f}"
        )

        print(
            f"   Specific categories: "
            f"{', '.join(similar_categories) if similar_categories else 'None'}"
        )

        print(
            f"   Shared specific categories: "
            f"{', '.join(shared_categories) if shared_categories else 'None'}"
        )

        if checked == 3:
            break

    match_rate = matches / 3

    print(
        f"\nSpecific-category match rate: "
        f"{match_rate:.2%}"
    )

print("\n" + "=" * 60)
print("OVERALL EVALUATION")
print("=" * 60)

overall_score = total_matches / total_comparisons

print(f"\nComparisons checked: {total_comparisons}")
print(f"Specific-category matches: {total_matches}")
print(
    f"Overall specific-category match rate: "
    f"{overall_score:.2%}"
)

print("\nEvaluation complete.")