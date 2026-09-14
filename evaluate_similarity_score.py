import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PAPER_FILE = "data/processed/papers_clean.csv"

# Load papers
with open(PAPER_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

print("=" * 60)
print("SIMILARITY CATEGORY EVALUATION")
print("=" * 60)

print(f"\nNumber of papers: {len(papers)}")

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create text from title + abstract
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

# Calculate similarity
similarity_matrix = cosine_similarity(embeddings)

# Evaluate 10 papers
test_indices = np.linspace(
    0,
    len(papers) - 1,
    10,
    dtype=int
)

total_matches = 0
total_comparisons = 0

for paper_index in test_indices:

    target_categories = set(
        papers[paper_index]["categories"].split(";")
    )

    scores = similarity_matrix[paper_index]

    similar_indices = np.argsort(scores)[::-1]

    matches = 0
    checked = 0

    print("\n" + "=" * 60)
    print(f"TARGET: {papers[paper_index]['title']}")
    print("=" * 60)

    for index in similar_indices:

        if index == paper_index:
            continue

        similar_categories = set(
            papers[index]["categories"].split(";")
        )

        shared_categories = (
            target_categories & similar_categories
        )

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
            f"   Shared categories: "
            f"{', '.join(shared_categories) if shared_categories else 'None'}"
        )

        if checked == 3:
            break

    accuracy = matches / 3

    print(
        f"\nCategory match rate for this paper: "
        f"{accuracy:.2%}"
    )

print("\n" + "=" * 60)
print("OVERALL EVALUATION")
print("=" * 60)

overall_score = total_matches / total_comparisons

print(f"\nComparisons checked: {total_comparisons}")
print(f"Category matches: {total_matches}")
print(f"Overall category match rate: {overall_score:.2%}")

print("\nEvaluation complete.")