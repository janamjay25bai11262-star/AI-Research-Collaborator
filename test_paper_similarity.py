import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PAPER_FILE = "data/processed/papers_clean.csv"

# Load papers
with open(PAPER_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

print("=" * 60)
print("PAPER SIMILARITY TEST")
print("=" * 60)

print(f"\nNumber of papers: {len(papers)}")

# Load model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Use title + abstract as the paper representation
paper_texts = [
    f"{paper['title']}. {paper['abstract']}"
    for paper in papers
]

# Generate embeddings
print("Generating paper embeddings...")

embeddings = model.encode(
    paper_texts,
    show_progress_bar=True
)

# Calculate similarities
similarity_matrix = cosine_similarity(embeddings)

# Test the first paper
paper_index = 0

target_paper = papers[paper_index]

print("\n" + "=" * 60)
print("TARGET PAPER")
print("=" * 60)

print("\nTitle:")
print(target_paper["title"])

# Find most similar papers
scores = similarity_matrix[paper_index]

similar_indices = np.argsort(scores)[::-1]

print("\n" + "=" * 60)
print("TOP 5 SIMILAR PAPERS")
print("=" * 60)

count = 0

for index in similar_indices:

    # Don't compare the paper with itself
    if index == paper_index:
        continue

    print(f"\n{count + 1}. {papers[index]['title']}")
    print(f"Similarity: {scores[index]:.4f}")
    print(f"Categories: {papers[index]['categories']}")

    count += 1

    if count == 5:
        break