import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PAPER_FILE = "data/processed/papers_clean.csv"

# Load papers
with open(PAPER_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

print("=" * 60)
print("MULTI-PAPER EMBEDDING EVALUATION")
print("=" * 60)

print(f"\nNumber of papers: {len(papers)}")

# Load model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Combine title + abstract
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

# Calculate pairwise similarity
similarity_matrix = cosine_similarity(embeddings)

# Test 10 papers
test_indices = np.linspace(
    0,
    len(papers) - 1,
    10,
    dtype=int
)

for paper_index in test_indices:

    print("\n" + "=" * 60)
    print(f"TARGET PAPER #{paper_index + 1}")
    print("=" * 60)

    print("\nTitle:")
    print(papers[paper_index]["title"])

    print("\nCategories:")
    print(papers[paper_index]["categories"])

    scores = similarity_matrix[paper_index]

    similar_indices = np.argsort(scores)[::-1]

    print("\nTop 3 similar papers:")

    count = 0

    for index in similar_indices:

        if index == paper_index:
            continue

        print(
            f"\n{count + 1}. {papers[index]['title']}"
        )
        print(
            f"   Similarity: {scores[index]:.4f}"
        )
        print(
            f"   Categories: {papers[index]['categories']}"
        )

        count += 1

        if count == 3:
            break

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)